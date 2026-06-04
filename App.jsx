import React, { useEffect, useRef, useState, useCallback } from "react";

const WS_URL = "ws://localhost:8000/ws";
const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 0.5;
const CHUNK_SIZE = SAMPLE_RATE * CHUNK_SECONDS;

function MetricTile({ label, value, color = "rgb(37,164,117)", dim = false }) {
  return (
    <div style={m.tile}>
      <span style={m.tileLabel}>{label}</span>
      <span style={{ ...m.tileValue, color: dim ? "rgb(45,45,51)" : color }}>
        {value ?? "—"}
      </span>
    </div>
  );
}

function HistoryItem({ item, index }) {
  const confPct = item.confidence != null
    ? Math.round(Math.exp(item.confidence) * 100)
    : null;

  const confColor =
    confPct == null ? "rgb(100,116,139)" :
    confPct >= 90   ? "rgb(37,164,117)" :
    confPct >= 70   ? "rgb(251,191,36)" : "rgb(239,68,68)";

  return (
    <div style={m.histItem}>
      <div style={m.histHeader}>
        <span style={m.histTime}>{item.time}</span>
        {confPct != null && (
          <span style={{ ...m.histConf, color: confColor }}>{confPct}%</span>
        )}
      </div>
      <span style={m.histEn}>{item.en}</span>
      {item.sv && <span style={m.histSv}>{item.sv}</span>}
      <span style={m.histMeta}>
        {item.asr_ms > 0 && `${item.asr_ms}ms asr`}
        {item.mt_ms > 0 && ` · ${item.mt_ms}ms mt`}
        {item.revisions > 0 && ` · ${item.revisions} rev`}
        {item.stability != null && ` · ${Math.round(item.stability * 100)}% stab`}
      </span>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────
export default function RealtimeTranslator() {
  const [stableEn, setStableEn]         = useState("");
  const [stableSv, setStableSv]         = useState("");
  const [liveSv,   setLiveSv]           = useState("");
  const [history,  setHistory]          = useState([]);
  const [isListening, setIsListening]   = useState(false);
  const [wsStatus, setWsStatus]         = useState("connecting");

  const [networkLatency,  setNetworkLatency]  = useState(null);
  const [asrLatency,      setAsrLatency]      = useState(null);
  const [mtLatency,       setMtLatency]       = useState(null);
  const [totalServerMs,   setTotalServerMs]   = useState(null);
  const [sessionStability,setSessionStability]= useState(null);
  const [lastRevised,     setLastRevised]     = useState(false);
  const [lastConfidence,  setLastConfidence]  = useState(null);

  const wsRef           = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef    = useRef(null);
  const histEndRef      = useRef(null);
  const mainEndRef      = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.binaryType = "arraybuffer";
    ws.onopen  = () => setWsStatus("open");
    ws.onclose = () => setWsStatus("closed");

    ws.onmessage = (event) => {
      const now = performance.now();
      const msg = JSON.parse(event.data);

      if (msg.t_sent != null) setNetworkLatency(Math.round(now - msg.t_sent));
      if (msg.asr_ms  != null) setAsrLatency(msg.asr_ms);
      if (msg.mt_ms   != null) setMtLatency(msg.mt_ms);
      if (msg.total_ms != null) setTotalServerMs(msg.total_ms);

      if (msg.is_final) {
        if (msg.translation) setStableEn(p => (p + " " + msg.translation).trim());
        if (msg.text)        setStableSv(p => (p + " " + msg.text).trim());
        setLiveSv("");
        setLastRevised(false);
        if (msg.session_stability != null) setSessionStability(msg.session_stability);
        if (msg.confidence != null) setLastConfidence(msg.confidence);

        if (msg.text || msg.translation) {
          setHistory(prev => [...prev, {
            sv:        msg.text        || "",
            en:        msg.translation || "",
            time:      new Date().toLocaleTimeString(),
            asr_ms:    msg.asr_ms              || 0,
            mt_ms:     msg.mt_ms               || 0,
            confidence: msg.confidence         ?? null,
            revisions:  msg.utterance_revisions || 0,
            stability:  msg.utterance_stability ?? null,
          }]);
        }
      } else {
        setLiveSv(msg.text || "");
        setLastRevised(!!msg.revised);
      }
    };

    return () => ws.close();
  }, []);

  // Auto-scroll
  useEffect(() => {
    histEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);
  useEffect(() => {
    mainEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [stableEn, liveSv]);

  const buildFrame = useCallback((chunk, tSent) => {
    const frame = new ArrayBuffer(8 + chunk.byteLength);
    new DataView(frame).setFloat64(0, tSent, true);
    new Float32Array(frame, 8).set(chunk);
    return frame;
  }, []);

  const startListening = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true,
               autoGainControl: true, sampleRate: SAMPLE_RATE, channelCount: 1 },
    });
    const audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    audioContextRef.current = audioContext;
    const source = audioContext.createMediaStreamSource(stream);

    const workletCode = `
      class AudioProcessor extends AudioWorkletProcessor {
        process(inputs) { const ch = inputs[0]?.[0]; if (ch) this.port.postMessage(ch); return true; }
      }
      registerProcessor("audio-processor", AudioProcessor);
    `;
    const blob = new Blob([workletCode], { type: "application/javascript" });
    const url  = URL.createObjectURL(blob);
    await audioContext.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);

    const node = new AudioWorkletNode(audioContext, "audio-processor");
    processorRef.current = node;
    let buffer = [];
    node.port.onmessage = (event) => {
      buffer.push(...event.data);
      if (buffer.length >= CHUNK_SIZE) {
        const chunk = new Float32Array(buffer.slice(0, CHUNK_SIZE));
        buffer = buffer.slice(CHUNK_SIZE);
        if (wsRef.current?.readyState === WebSocket.OPEN)
          wsRef.current.send(buildFrame(chunk, performance.now()));
      }
    };
    source.connect(node);
    setIsListening(true);
  }, [buildFrame]);

  const stopListening = useCallback(() => {
    processorRef.current?.disconnect();
    audioContextRef.current?.close();
    setIsListening(false);
  }, []);

  const exportHistory = useCallback(() => {
    if (!history.length) return;
    const ts   = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const data = {
      exported_at: new Date().toISOString(),
      total_utterances: history.length,
      utterances: history.map((item, i) => ({
        index: i + 1, time: item.time,
        swedish: item.sv, english: item.en,
        asr_ms: item.asr_ms, mt_ms: item.mt_ms,
        confidence: item.confidence, revisions: item.revisions, stability: item.stability,
      })),
      full_swedish: stableSv,
      full_english: stableEn,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = `translation_${ts}.json`; a.click();
    URL.revokeObjectURL(url);
  }, [history, stableSv, stableEn]);

  // Derived display values
  const wsConnected  = wsStatus === "open";
  const livePhrases  = liveSv
    ? liveSv.split(/\s+/).filter(Boolean).reduce((acc, w, i) => {
        const gi = Math.floor(i / 3);
        acc[gi] = acc[gi] ? acc[gi] + " " + w : w;
        return acc;
      }, [])
    : [];

  const stabPct   = sessionStability != null ? Math.round(sessionStability * 100) : null;
  const stabColor = sessionStability == null ? "rgb(100,116,139)"
    : sessionStability > 0.8 ? "rgb(37,164,117)"
    : sessionStability > 0.5 ? "rgb(251,191,36)" : "rgb(239,68,68)";

  const confPct   = lastConfidence != null ? Math.round(Math.exp(lastConfidence) * 100) : null;
  const confColor = confPct == null ? "rgb(100,116,139)"
    : confPct >= 90 ? "rgb(37,164,117)"
    : confPct >= 70 ? "rgb(251,191,36)" : "rgb(239,68,68)";

  const statusDot = wsStatus === "open" ? "rgb(37,164,117)"
    : wsStatus === "connecting"         ? "rgb(251,191,36)" : "rgb(239,68,68)";

  return (
    <div style={m.shell}>
      <style>{CSS}</style>

      <header style={m.topbar}>
        <span style={m.brand}>SVEN</span>
        <div style={m.topCenter}>
          <span style={m.langBadge}>SV-EN</span>
        </div>
        <div style={m.topRight}>
          <span style={{ ...m.statusPill, borderColor: wsConnected ? "rgba(37,164,117,0.3)" : "rgb(45,45,51)" }}>
            <span style={{ ...m.statusDot, background: statusDot }} />
            <span style={{ ...m.statusText, color: wsConnected ? "rgb(37,164,117)" : "rgb(100,116,139)" }}>
              {wsStatus === "open" ? "CONNECTED" : wsStatus === "connecting" ? "CONNECTING" : "OFFLINE"}
            </span>
          </span>
        </div>
      </header>

      <main style={m.main}>

        <div style={m.leftCol}>

          <div style={m.controlBar}>
            <div style={m.controlLeft}>
              <button
                onClick={isListening ? stopListening : startListening}
                disabled={!wsConnected}
                style={{
                  ...m.btnPrimary,
                  background: isListening ? "rgb(220,38,38)" : "rgb(37,99,235)",
                  opacity: wsConnected ? 1 : 0.3,
                }}
                className="btn-hover"
              >
                <span style={m.btnMic}>{isListening ? "■" : "●"}</span>
                {isListening ? "STOP SESSION" : "START SESSION"}
              </button>

              {isListening && (
                <div style={m.liveChip}>
                  <span style={m.liveDot} className="pulse" />
                  <span style={m.liveText}>LIVE</span>
                </div>
              )}
            </div>

            <button
              onClick={exportHistory}
              disabled={!history.length}
              style={{ ...m.btnSecondary, opacity: history.length ? 1 : 0.3 }}
              className="btn-hover"
            >
              <span style={m.exportIcon}>↓</span>
              EXPORT
            </button>
          </div>

          <div style={m.outputBox}>
            <div style={m.outputScroll}>
              <p style={m.outputText}>
                {stableEn && <span>{stableEn} </span>}
                {livePhrases.map((phrase, i) => (
                  <span
                    key={`p${i}`}
                    className={lastRevised && i === livePhrases.length - 1 ? "revise-pulse" : "phrase-in"}
                    style={{
                      color: "rgba(229,225,228,0.35)",
                      fontStyle: "italic",
                      animationDelay: `${i * 55}ms`,
                    }}
                  >
                    {phrase}{" "}
                  </span>
                ))}
                {!stableEn && !livePhrases.length && (
                  <span style={m.placeholder}>Waiting for speech…</span>
                )}
              </p>
              <div ref={mainEndRef} />
            </div>

            <div style={m.svStrip}>
              <span style={m.svLabel}>SOURCE: SV</span>
              <span style={m.svText}>
                {stableSv && <span>{stableSv} </span>}
                {liveSv && <span style={{ color: "rgba(195,198,215,0.5)" }}>{liveSv}</span>}
                {!stableSv && !liveSv && <span style={{ color: "rgb(45,45,51)" }}>—</span>}
              </span>
            </div>
          </div>
        </div>

        <div style={m.rightCol}>

          <div style={m.metricsPanel}>
            <div style={m.panelHeader}>
              <span style={m.panelTitle}>SYSTEM METRICS</span>
              <span style={m.expertBadge}>
                <span style={{ ...m.statusDot, background: wsConnected ? "rgb(37,164,117)" : "rgb(45,45,51)", width: 6, height: 6 }} />
                EXPERT
              </span>
            </div>
            <div style={m.metricsGrid}>
              <MetricTile label="ASR LATENCY"   value={asrLatency   != null ? `${asrLatency} ms`   : null} dim={asrLatency == null} />
              <MetricTile label="MT LATENCY"    value={mtLatency    != null && mtLatency > 0 ? `${mtLatency} ms` : null} dim={!mtLatency} />
              <MetricTile label="TOTAL TIME"    value={totalServerMs != null && totalServerMs > 0 ? `${totalServerMs} ms` : null} dim={!totalServerMs} />
              <MetricTile label="CONFIDENCE"    value={confPct != null ? `${confPct}%` : null} color={confColor} dim={confPct == null} />
              <MetricTile label="NET RTT"       value={networkLatency != null ? `${networkLatency} ms` : null} dim={networkLatency == null} />
              <MetricTile label="STABILITY"     value={stabPct != null ? `${stabPct}%` : null} color={stabColor} dim={stabPct == null} />
            </div>
          </div>

          <div style={m.historyPanel}>
            <div style={m.panelHeader}>
              <span style={m.panelTitle}>RECENT TRANSLATIONS</span>
              {history.length > 0 && (
                <span style={m.histCount}>{history.length}</span>
              )}
            </div>
            <div style={m.histScroll}>
              {history.length === 0 ? (
                <div style={m.histEmpty}>No translations yet</div>
              ) : (
                history.map((item, i) => (
                  <HistoryItem key={i} item={item} index={i} />
                ))
              )}
              <div ref={histEndRef} />
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500&family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;1,6..72,300&family=Inter:ital,wght@0,400;1,400&display=swap');

  *, *::before, *::after { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: rgb(12,12,14); height: 100%; }
  #root { height: 100%; }

  @keyframes phraseIn {
    from { opacity: 0; transform: translateY(3px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes revisePulse {
    0%   { background: rgba(251,191,36,0.15); border-radius: 3px; }
    100% { background: transparent; }
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.85); }
  }

  .phrase-in  { animation: phraseIn 0.22s ease-out forwards; opacity: 0; }
  .revise-pulse { animation: revisePulse 0.4s ease-out; }
  .pulse      { animation: pulse 1.4s ease-in-out infinite; }

  .btn-hover:hover:not(:disabled) { filter: brightness(1.12); transform: translateY(-1px); }
  .btn-hover:active:not(:disabled) { transform: translateY(0); filter: brightness(0.95); }
  .btn-hover { transition: filter 0.15s, transform 0.1s; }

  ::-webkit-scrollbar       { width: 3px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgb(45,45,51); border-radius: 2px; }
`;

const m = {
  shell: {
    minHeight: "100vh",
    height: "100vh",
    background: "rgb(12,12,14)",
    display: "flex",
    flexDirection: "column",
    fontFamily: "'Space Grotesk', sans-serif",
    color: "rgb(229,225,228)",
    overflow: "hidden",
  },

  topbar: {
    height: 56,
    flexShrink: 0,
    background: "rgb(12,12,14)",
    borderBottom: "1px solid rgb(45,45,51)",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0 16px",
  },
  brand: {
    fontFamily: "'Space Grotesk', monospace",
    fontWeight: 700,
    fontSize: 18,
    letterSpacing: "-0.05em",
    color: "rgb(255,255,255)",
  },
  topCenter: {
    position: "absolute",
    left: "50%",
    transform: "translateX(-50%)",
  },
  langBadge: {
    fontFamily: "'Space Grotesk', monospace",
    fontSize: 10,
    letterSpacing: "0.1em",
    color: "rgb(37,99,235)",
    border: "1px solid rgb(45,45,51)",
    background: "rgb(22,22,26)",
    borderRadius: 2,
    padding: "4px 8px",
  },
  topRight: {
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  statusPill: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    border: "1px solid rgb(45,45,51)",
    borderRadius: 2,
    padding: "3px 8px",
  },
  statusDot: {
    width: 7, height: 7,
    borderRadius: "50%",
    flexShrink: 0,
  },
  statusText: {
    fontSize: 9,
    letterSpacing: "0.1em",
    fontWeight: 500,
  },

  main: {
    flex: 1,
    display: "grid",
    gridTemplateColumns: "1fr 380px",
    gap: 0,
    overflow: "hidden",
    padding: "16px",
    gap: "12px",
  },

  leftCol: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
    overflow: "hidden",
    minWidth: 0,
  },
  controlBar: {
    flexShrink: 0,
    background: "rgb(22,22,26)",
    border: "1px solid rgb(45,45,51)",
    borderRadius: 4,
    padding: "12px 16px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  controlLeft: {
    display: "flex",
    alignItems: "center",
    gap: 16,
  },
  btnPrimary: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "7px 20px",
    border: "none",
    borderRadius: 2,
    color: "rgb(238,239,255)",
    fontSize: 11,
    fontFamily: "'Space Grotesk', sans-serif",
    fontWeight: 500,
    letterSpacing: "0.1em",
    cursor: "pointer",
  },
  btnMic: {
    fontSize: 8,
    lineHeight: 1,
  },
  btnSecondary: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "6px 16px",
    border: "1px solid rgb(45,45,51)",
    borderRadius: 2,
    background: "transparent",
    color: "rgb(229,225,228)",
    fontSize: 11,
    fontFamily: "'Space Grotesk', sans-serif",
    fontWeight: 500,
    letterSpacing: "0.1em",
    cursor: "pointer",
  },
  exportIcon: {
    fontSize: 13,
    lineHeight: 1,
  },
  liveChip: {
    display: "flex",
    alignItems: "center",
    gap: 6,
  },
  liveDot: {
    width: 7, height: 7,
    borderRadius: "50%",
    background: "rgb(37,164,117)",
    flexShrink: 0,
  },
  liveText: {
    fontSize: 11,
    letterSpacing: "0.1em",
    color: "rgb(37,164,117)",
    fontWeight: 500,
  },

  outputBox: {
    flex: 1,
    background: "rgb(22,22,26)",
    border: "1px solid rgb(45,45,51)",
    borderRadius: 4,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    minHeight: 0,
  },
  outputScroll: {
    flex: 1,
    overflowY: "auto",
    padding: "24px 24px 32px",
  },
  outputText: {
    margin: 0,
    fontFamily: "'Newsreader', Georgia, serif",
    fontSize: 48,
    lineHeight: 1.1,
    letterSpacing: "-0.02em",
    fontWeight: 300,
    color: "rgb(255,255,255)",
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  },
  placeholder: {
    fontFamily: "'Newsreader', Georgia, serif",
    fontStyle: "italic",
    color: "rgb(45,45,51)",
    fontSize: 32,
  },

  svStrip: {
    flexShrink: 0,
    background: "rgb(27,27,29)",
    borderTop: "1px solid rgb(45,45,51)",
    padding: "14px 24px",
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  svLabel: {
    fontSize: 11,
    letterSpacing: "0.1em",
    color: "rgb(100,116,139)",
    fontWeight: 500,
  },
  svText: {
    fontFamily: "'Inter', sans-serif",
    fontStyle: "italic",
    fontSize: 16,
    lineHeight: 1.6,
    color: "rgb(195,198,215)",
    fontWeight: 400,
  },

  rightCol: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
    overflow: "hidden",
    minWidth: 0,
  },

  panelHeader: {
    flexShrink: 0,
    padding: "12px 16px 10px",
    borderBottom: "1px solid rgb(45,45,51)",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  panelTitle: {
    fontSize: 11,
    letterSpacing: "0.1em",
    color: "rgb(100,116,139)",
    fontWeight: 500,
  },

  metricsPanel: {
    flexShrink: 0,
    background: "rgb(22,22,26)",
    border: "1px solid rgb(45,45,51)",
    borderRadius: 4,
    display: "flex",
    flexDirection: "column",
  },
  expertBadge: {
    display: "flex",
    alignItems: "center",
    gap: 5,
    border: "1px solid rgb(45,45,51)",
    borderRadius: 2,
    padding: "2px 8px",
    fontSize: 9,
    letterSpacing: "0.05em",
    color: "rgb(100,116,139)",
  },
  metricsGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr 1fr",
    padding: "12px 16px 14px",
    gap: "12px 0",
  },
  tile: {
    display: "flex",
    flexDirection: "column",
    gap: 3,
  },
  tileLabel: {
    fontSize: 11,
    letterSpacing: "0.1em",
    color: "rgb(100,116,139)",
    fontWeight: 500,
  },
  tileValue: {
    fontSize: 13,
    letterSpacing: "0.05em",
    fontWeight: 400,
    color: "rgb(37,164,117)",
  },

  historyPanel: {
    flex: 1,
    background: "rgb(22,22,26)",
    border: "1px solid rgb(45,45,51)",
    borderRadius: 4,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    minHeight: 0,
  },
  histCount: {
    fontSize: 10,
    color: "rgb(100,116,139)",
    letterSpacing: "0.05em",
    border: "1px solid rgb(45,45,51)",
    borderRadius: 2,
    padding: "1px 6px",
  },
  histScroll: {
    flex: 1,
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
  },
  histEmpty: {
    padding: "24px 16px",
    fontSize: 13,
    color: "rgb(45,45,51)",
    fontStyle: "italic",
    fontFamily: "'Inter', sans-serif",
  },

  histItem: {
    padding: "14px 16px",
    borderBottom: "1px solid rgb(45,45,51)",
    display: "flex",
    flexDirection: "column",
    gap: 4,
  },
  histHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "baseline",
  },
  histTime: {
    fontSize: 13,
    letterSpacing: "0.05em",
    color: "rgb(100,116,139)",
  },
  histConf: {
    fontSize: 13,
    letterSpacing: "0.05em",
    fontWeight: 500,
  },
  histEn: {
    fontSize: 14,
    lineHeight: 1.5,
    color: "rgb(229,225,228)",
    fontFamily: "'Inter', sans-serif",
  },
  histSv: {
    fontSize: 13,
    lineHeight: 1.45,
    color: "rgb(100,116,139)",
    fontFamily: "'Inter', sans-serif",
    fontStyle: "italic",
  },
  histMeta: {
    fontSize: 10,
    letterSpacing: "0.05em",
    color: "rgb(45,45,51)",
    fontWeight: 500,
    marginTop: 2,
  },
};
