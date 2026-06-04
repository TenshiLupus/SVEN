import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GOMP_CPU_AFFINITY"] = "0-7"

import re
import struct
import time
import asyncio
import logging
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps
from huggingface_hub import snapshot_download
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


SAMPLE_RATE = 16000
MIN_SPEECH_SEC = 0.8
MAX_SPEECH_SEC = 20.0
TRANSCRIBE_COOLDOWN = 1.5

END_SILENCE_SEC = 0.8
SOFT_SPEECH_SEC = 8.0
SOFT_END_SILENCE_SEC = 0.6

DEAD_MIC_THRESH = 0.0002

PRE_SPEECH_SEC = 0.8

VAD_THRESHOLD = 0.25


MIN_NEW_WORDS = 2
MIN_CHANGED_CHARS = 5

MAX_REVISIONS_PER_UTTERANCE = 2
STABILITY_WINDOW = 5

PROMPT_MAX_CHARS = 500

COMMON_WORDS = {
    "jag", "du", "vi", "de", "det", "den", "är", "var", "har", "hade",
    "och", "att", "i", "på", "med", "för", "av", "till", "en", "ett",
    "som", "inte", "om", "men", "så", "kan", "ska", "vill", "från",
    "när", "hur", "vad", "där", "här", "nu", "också", "bara", "sen",
    "alla", "lite", "mer", "mycket", "nog", "the", "is", "it", "and",
    "to", "a", "of", "in", "that", "this", "was", "for", "on", "with",
}

ASR_REPO_LIVE = "KBLab/kb-whisper-base"
ASR_REPO_FINAL = "KBLab/kb-whisper-base"
MT_REPO = "quickmt/quickmt-sv-en"

MIN_AVG_LOGPROB = -0.7
MAX_COMPRESSION_RATIO = 2.4

ADAPTIVE_MODEL_SEC = 1.5

MT_CONTEXT_SENTENCES = 3
CPU_THREADS = 8

pool = ThreadPoolExecutor(max_workers=2)

print(f"Loading live ASR model: {ASR_REPO_LIVE}...")
asr_live_path = snapshot_download(
    ASR_REPO_LIVE,
    local_dir="cache_models/asr_live",
    allow_patterns=["model.bin", "*.json", "*.txt"],
    max_workers=1,
)
model_live = WhisperModel(
    asr_live_path,
    device="cpu",
    compute_type="int8",
    cpu_threads=CPU_THREADS,
    num_workers=1,
)

if ASR_REPO_FINAL == ASR_REPO_LIVE:
  
    print(f"Final ASR matches live ({ASR_REPO_FINAL}) — sharing instance.")
    model_final = model_live
else:
    print(f"Loading final ASR model: {ASR_REPO_FINAL}...")
    asr_final_path = snapshot_download(
        ASR_REPO_FINAL,
        local_dir="cache_models/asr_final",
        allow_patterns=["model.bin", "*.json", "*.txt"],
        max_workers=1,
    )
    model_final = WhisperModel(
        asr_final_path,
        device="cpu",
        compute_type="int8",
        cpu_threads=CPU_THREADS,
        num_workers=1,
    )

print("Loading VAD...")
vad_model = load_silero_vad()

print("Loading MT model (quickmt-sv-en)...")
mt_model = None
try:
    from quickmt import Translator
    mt_path = snapshot_download(MT_REPO, ignore_patterns=["eole-model/*"])
    try:
        mt_model = Translator(mt_path, device="cpu", compute_type="int8")
        print("MT model loaded (int8).")
    except TypeError:
        mt_model = Translator(mt_path, device="cpu")
        print("MT model loaded (default precision).")
except ImportError:
    print("WARNING: quickmt not installed — pip install quickmt")
except Exception as e:
    print(f"WARNING: Failed to load MT model: {e}")

print("Warming up...")
_w = np.zeros(SAMPLE_RATE, dtype=np.float32)
list(model_live.transcribe(_w, language="sv")[0])
if model_final is not model_live:
    list(model_final.transcribe(_w, language="sv")[0])
get_speech_timestamps(_w, vad_model, sampling_rate=SAMPLE_RATE,
                      threshold=VAD_THRESHOLD)
if mt_model is not None:
    mt_model("Hej", beam_size=1)
del _w
print("Ready.")
print(f"Live model compute type: {model_live.model.compute_type}")
print(f"Final model compute type: {model_final.model.compute_type}")
print(
    f"Splitting: soft={SOFT_SPEECH_SEC}s "
    f"(end-silence drops {END_SILENCE_SEC}s -> {SOFT_END_SILENCE_SEC}s past soft) "
    f"| hard={MAX_SPEECH_SEC}s"
)



def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def is_junk(t: str) -> bool:
    t = clean(t)
    return (
        not t
        or "<|" in t
        or "nospeech" in t.lower()
        or bool(re.fullmatch(r"[\d\W_]+", t))
        or t.lower() in {"mm", "mmm", "eh", "öh", "ah", "uh", "um", "m"}
    )


def word_diff(old_text: str, new_text: str) -> str:
    old_words = old_text.split()
    new_words = new_text.split()
    if len(new_words) <= len(old_words):
        return ""
    return " ".join(new_words[len(old_words):])


def count_revisions(old_text: str, new_text: str) -> int:
    old_words = old_text.lower().split()
    new_words = new_text.lower().split()
    check_len = min(len(old_words), len(new_words))
    return sum(1 for i in range(check_len) if old_words[i] != new_words[i])

def build_smart_prompt(accumulated_sv: str, max_chars: int = PROMPT_MAX_CHARS) -> str:
    if not accumulated_sv:
        return ""
    words = accumulated_sv.split()
    if not words:
        return ""
    uncommon = []
    seen = set()
    for w in words:
        w_lower = w.lower().strip(".,!?;:")
        if w_lower not in COMMON_WORDS and w_lower not in seen and len(w_lower) > 2:
            uncommon.append(w)
            seen.add(w_lower)
    uncommon_str = " ".join(uncommon[-30:])
    recent_str = accumulated_sv[-(max_chars // 2):]
    prompt = (uncommon_str + " " + recent_str).strip()
    if len(prompt) > max_chars:
        prompt = prompt[-max_chars:]
    return prompt

def has_speech(audio_f32: np.ndarray) -> bool:
    ts = get_speech_timestamps(
        audio_f32, vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
    )
    return len(ts) > 0

def trim_trailing_silence(audio: np.ndarray, threshold: float = 0.01,
                          frame_ms: int = 30) -> np.ndarray:
    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    min_keep = int(SAMPLE_RATE * 0.1)
    end = len(audio)
    while end - frame_samples > min_keep:
        frame = audio[end - frame_samples:end]
        if np.max(np.abs(frame)) > threshold:
            break
        end -= frame_samples
    return audio[:end] if end < len(audio) else audio


def trim_leading_silence(audio: np.ndarray, threshold: float = 0.01,
                         frame_ms: int = 30,
                         keep_pad_ms: int = 150) -> np.ndarray:
   
    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    pad_samples = int(SAMPLE_RATE * keep_pad_ms / 1000)
    start = 0
    # Scan forward until we find a frame with energy above threshold
    while start + frame_samples < len(audio):
        frame = audio[start:start + frame_samples]
        if np.max(np.abs(frame)) > threshold:
            break
        start += frame_samples
    # Back off by keep_pad_ms so onset isn't clipped
    start = max(0, start - pad_samples)
    return audio[start:] if start > 0 else audio


def _extract_text(seg_list: list) -> tuple[str, float]:
    """Returns (text, avg_logprob) from accepted segments."""
    if not seg_list:
        return "", 0.0
    good_segments = []
    for s in seg_list:
        logprob = getattr(s, "avg_logprob", 0.0)
        compression = getattr(s, "compression_ratio", 1.0)
        no_speech = getattr(s, "no_speech_prob", 0.0)
        if no_speech > 0.6:
            continue
        if logprob < MIN_AVG_LOGPROB:
            continue
        if compression > MAX_COMPRESSION_RATIO:
            continue
        good_segments.append(s)
    if not good_segments:
        return "", 0.0
    text = clean(" ".join(s.text.strip() for s in good_segments))
    if is_junk(text):
        return "", 0.0
    avg_conf = sum(getattr(s, "avg_logprob", 0.0) for s in good_segments) / len(good_segments)
    return text, round(avg_conf, 3)


def transcribe(audio: np.ndarray) -> tuple[str, float]:
   
    segments, _ = model_live.transcribe(
        audio,
        language="sv",
        beam_size=1,
        condition_on_previous_text=False,
        temperature=0.0,
        no_speech_threshold=0.6,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        without_timestamps=True,
    )
    return _extract_text(list(segments))


def transcribe_final(audio: np.ndarray, previous_text: str = "") -> tuple[str, float]:
   
    audio = trim_leading_silence(audio)
    audio = trim_trailing_silence(audio)

    duration = len(audio) / SAMPLE_RATE
    active_model = model_live if duration < ADAPTIVE_MODEL_SEC else model_final

    segments, _ = active_model.transcribe(
        audio,
        language="sv",
        beam_size=5,
        condition_on_previous_text=False,
        temperature=0.0,
        no_speech_threshold=0.6,
        vad_filter=False,
        initial_prompt=previous_text if previous_text else None,
        without_timestamps=True,
    )
    return _extract_text(list(segments))

def translate(text: str) -> str:
    if mt_model is None or not text.strip():
        return ""

    word_count = len(text.split())

    try:
       
        result = clean(mt_model(
            text,
            beam_size=5,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            max_decoding_length=max(20 if word_count > 3 else 6,
                                    word_count * 3),
        ))

        return trim_translation(result, text)

    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return ""


def trim_translation(en: str, sv: str) -> str:
    if not en or not sv:
        return en

    sv_sents = [s.strip() for s in re.split(r'[.!?]+', sv) if s.strip()]
    en_sents = re.split(r'(?<=[.!?])\s+', en.strip())
    if len(en_sents) > max(1, len(sv_sents)):
        en = " ".join(en_sents[:max(1, len(sv_sents))])

    sv_words = len(sv.split())
    en_words = en.split()
    max_words = max(2, sv_words * 2)
    if len(en_words) > max_words:
        en = " ".join(en_words[:max_words])
        for i in range(len(en) - 1, -1, -1):
            if en[i] in '.!?':
                en = en[:i + 1]
                break

    TRAILING_JUNK = {
        'and', 'or', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to',
        'for', 'with', 'by', 'is', 'it', 'but', 'so', 'if', 'as', 'not',
        "i'm", "i", "we", "they", "he", "she", "you", "that", "this",
        "was", "were", "are", "been", "be", "have", "has", "had",
    }
    words = en.split()
    while words and words[-1].lower().rstrip('.,!?;:') in TRAILING_JUNK:
        words.pop()
    en = " ".join(words)

    en = re.sub(r'\s+([.,!?;:])', r'\1', en)
    en = en.rstrip(' ,;:-')

    return en.strip()


def transcribe_and_translate(
    audio: np.ndarray,
    final: bool = False,
    context_sentences: list[str] = None,
    previous_text: str = "",
):
    asr_start = time.monotonic()
    if final:
        sv, confidence = transcribe_final(audio, previous_text=previous_text)
    else:
        sv, confidence = transcribe(audio)
    asr_ms = round((time.monotonic() - asr_start) * 1000)

    if not sv:
        return ("", "", asr_ms, 0, confidence)

    mt_ms = 0
    if final:
        mt_start = time.monotonic()
        en = translate(sv)
        mt_ms = round((time.monotonic() - mt_start) * 1000)
    else:
        en = ""

    return (sv, en, asr_ms, mt_ms, confidence)


def parse_frame(data: bytes):
    t_sent = struct.unpack_from("<d", data, 0)[0]
    audio = np.frombuffer(data[8:], dtype=np.float32).copy()
    return t_sent, audio


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    # ── per-utterance state ──
    speech_buffer = np.zeros(int(MAX_SPEECH_SEC * SAMPLE_RATE + SAMPLE_RATE), dtype=np.float32)
    speech_len = 0
    is_speaking = False
    t_sent_speech_start = None
    last_speech_time = 0.0

    pre_buffer: deque[np.ndarray] = deque()
    pre_buffer_len = 0
    pre_buffer_max = int(PRE_SPEECH_SEC * SAMPLE_RATE)

    utterance_ready = asyncio.Event()
    utterance_finalized = asyncio.Event()
  
    finalize_reason: str = ""

    last_sent_sv = ""
    last_sent_en = ""
    last_transcribe_time = 0.0

    revision_count = 0
    stability_scores: deque[float] = deque(maxlen=STABILITY_WINDOW)
    live_suppressed = False

    accumulated_sv = ""
    sv_context: deque[str] = deque(maxlen=MT_CONTEXT_SENTENCES)

    closed = False

    def append_to_buffer(chunk: np.ndarray):
        nonlocal speech_buffer, speech_len
        end = speech_len + len(chunk)
        if end > len(speech_buffer):
            new_buf = np.zeros(end * 2, dtype=np.float32)
            new_buf[:speech_len] = speech_buffer[:speech_len]
            speech_buffer = new_buf
        speech_buffer[speech_len:end] = chunk
        speech_len = end

    def get_speech_buffer() -> np.ndarray:
        if speech_len == 0:
            return np.zeros(0, dtype=np.float32)
        return speech_buffer[:speech_len].copy()

    def reset_utterance():
        nonlocal is_speaking, speech_len, last_speech_time
        nonlocal t_sent_speech_start, last_sent_sv, last_sent_en
        nonlocal revision_count, live_suppressed, pre_buffer_len
        nonlocal finalize_reason
        speech_len = 0
        is_speaking = False
        t_sent_speech_start = None
        last_sent_sv = ""
        last_sent_en = ""
        last_speech_time = 0.0
        revision_count = 0
        live_suppressed = False
        finalize_reason = ""
        utterance_ready.clear()
        utterance_finalized.clear()
        pre_buffer.clear()
        pre_buffer_len = 0

    def add_pre_buffer_to_speech():
        nonlocal pre_buffer_len
        while pre_buffer:
            chunk = pre_buffer.popleft()
            append_to_buffer(chunk)
        pre_buffer_len = 0

    def push_pre_buffer(chunk: np.ndarray):
        nonlocal pre_buffer_len
        pre_buffer.append(chunk)
        pre_buffer_len += len(chunk)
        while pre_buffer_len > pre_buffer_max:
            removed = pre_buffer.popleft()
            pre_buffer_len -= len(removed)

    def current_effective_silence() -> float:
      
        buf_seconds = speech_len / SAMPLE_RATE
        if buf_seconds >= SOFT_SPEECH_SEC:
            return SOFT_END_SILENCE_SEC
        return END_SILENCE_SEC

    # ── Receiver ──
    async def _receiver():
        nonlocal is_speaking, speech_len
        nonlocal t_sent_speech_start, last_speech_time, closed
        nonlocal finalize_reason

        try:
            while True:
                data = await ws.receive_bytes()
                t_sent, audio_chunk = parse_frame(data)
                now = time.monotonic()

                peak = np.max(np.abs(audio_chunk))
                if peak < DEAD_MIC_THRESH:
                   
                    if is_speaking and (now - last_speech_time) >= current_effective_silence():
                        finalize_reason = "end_silence"
                        utterance_finalized.set()
                    continue

                if is_speaking:
                    append_to_buffer(audio_chunk)

                    if peak > 0.02:
                        last_speech_time = now

                    buf_seconds = speech_len / SAMPLE_RATE

                    if buf_seconds >= MIN_SPEECH_SEC:
                        utterance_ready.set()

                    if (now - last_speech_time) >= current_effective_silence():
                        finalize_reason = "end_silence"
                        utterance_finalized.set()
                        continue

                    if buf_seconds >= MAX_SPEECH_SEC:
                        finalize_reason = "max_speech"
                        logger.info(
                            f"Hard cap hit at {buf_seconds:.1f}s "
                            f"(MAX_SPEECH_SEC={MAX_SPEECH_SEC}) — forcing split"
                        )
                        utterance_finalized.set()

                else:
                    chunk_has_speech = await loop.run_in_executor(
                        pool, has_speech, audio_chunk
                    )

                    if chunk_has_speech:
                        last_speech_time = now
                        is_speaking = True
                        t_sent_speech_start = t_sent
                        add_pre_buffer_to_speech()
                        append_to_buffer(audio_chunk)

                        buf_seconds = speech_len / SAMPLE_RATE
                        if buf_seconds >= MIN_SPEECH_SEC:
                            utterance_ready.set()
                    else:
                        push_pre_buffer(audio_chunk)

        except WebSocketDisconnect:
            logger.info("Client disconnected.")
            closed = True
        except Exception as e:
            logger.info(f"Receiver closed: {e}")
            closed = True

    async def _transcriber():
        nonlocal last_sent_sv, last_sent_en, last_transcribe_time
        nonlocal revision_count, live_suppressed

        try:
            while not closed:
                try:
                    await asyncio.wait_for(utterance_ready.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if utterance_finalized.is_set():
                    await _do_final()
                    continue

                if live_suppressed:
                    await asyncio.sleep(0.05)
                    continue

                now = time.monotonic()
                if now - last_transcribe_time >= TRANSCRIBE_COOLDOWN:
                    full_audio = get_speech_buffer()
                    if len(full_audio) < int(MIN_SPEECH_SEC * SAMPLE_RATE):
                        await asyncio.sleep(0.05)
                        continue

               
                    sv, en, asr_ms, mt_ms, _ = await loop.run_in_executor(
                        pool, transcribe_and_translate, full_audio, False
                    )
                    last_transcribe_time = time.monotonic()

                    if sv and sv != last_sent_sv:
                        revisions = count_revisions(last_sent_sv, sv)
                        if revisions > 0:
                            revision_count += 1

                        if revision_count > MAX_REVISIONS_PER_UTTERANCE:
                            live_suppressed = True
                            logger.debug(
                                f"Live suppressed after {revision_count} revisions"
                            )
                            continue

                        # Decision Policy
                        new_words = word_diff(last_sent_sv, sv)
                        new_word_count = len(new_words.split()) if new_words else 0
                        char_diff = abs(len(sv) - len(last_sent_sv))

                        if (new_word_count < MIN_NEW_WORDS
                                and char_diff < MIN_CHANGED_CHARS
                                and revisions == 0):
                            continue

                        last_sent_sv = sv
                        last_sent_en = en
                        await ws.send_text(json.dumps({
                            "type": "live",
                            "text": sv,
                            "translation": en,
                            "new_words": new_words,
                            "revised": revisions > 0,
                            "stability": round(
                                1.0 - (revision_count / max(1, STABILITY_WINDOW)),
                                2
                            ),
                            "is_final": False,
                            "t_sent": t_sent_speech_start,
                            "asr_ms": asr_ms,
                        }))

                if utterance_finalized.is_set():
                    await _do_final()
                    continue

                await asyncio.sleep(0.05)

        except Exception as e:
            if not closed:
                logger.error(f"Transcriber error: {e}")

    async def _do_final():
        nonlocal last_sent_sv, last_sent_en, last_transcribe_time
        nonlocal accumulated_sv, revision_count

        asr_ms = 0
        mt_ms = 0
        confidence = 0.0
        reason_for_this_final = finalize_reason or "unknown"

        if speech_len >= int(MIN_SPEECH_SEC * SAMPLE_RATE):
            buf = get_speech_buffer()
            prompt = build_smart_prompt(accumulated_sv)
            sv, en, asr_ms, mt_ms, confidence = await loop.run_in_executor(
                pool, transcribe_and_translate, buf, True, None, prompt
            )
            last_transcribe_time = time.monotonic()
            if sv:
                last_sent_sv = sv
                last_sent_en = en

        if last_sent_sv:
            accumulated_sv = clean(accumulated_sv + " " + last_sent_sv)
            if len(accumulated_sv) > 2000:
                accumulated_sv = accumulated_sv[-1500:]
            sv_context.append(last_sent_sv)

            utt_stability = max(0.0, 1.0 - (revision_count / max(1, STABILITY_WINDOW)))
            stability_scores.append(utt_stability)
            avg_stability = round(
                sum(stability_scores) / len(stability_scores), 2
            ) if stability_scores else 1.0

            try:
                await ws.send_text(json.dumps({
                    "type": "final",
                    "text": last_sent_sv,
                    "translation": last_sent_en,
                    "is_final": True,
                    "t_sent": None,
                    "asr_ms": asr_ms,
                    "mt_ms": mt_ms,
                    "total_ms": asr_ms + mt_ms,
                    "confidence": confidence,
                    "utterance_revisions": revision_count,
                    "utterance_stability": utt_stability,
                    "session_stability": avg_stability,
                    "finalize_reason": reason_for_this_final,
                }))
            except Exception:
                pass

        reset_utterance()

    receiver_task = asyncio.create_task(_receiver())
    transcriber_task = asyncio.create_task(_transcriber())

    try:
        await receiver_task
    finally:
        transcriber_task.cancel()
        try:
            await transcriber_task
        except asyncio.CancelledError:
            pass