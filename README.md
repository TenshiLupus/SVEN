Real-Time Swedish-to-English Speech Translation and Transcription
This repository contains the files corresponding to the artefact used in the thesis:
"Real-Time Swedish-to-English Speech Translation and Transcription for Inclusive Communication"
---
How to Run the Project
1. Activate the Python virtual environment
In a terminal, run:
```bash
source .venv/Scripts/activate
```
---
2. After the environment is activated run the backend with: 
```bash
uvicorn server_interview:app --host 0.0.0.0 --port 8000
```
---
2a. if some dependecies are missing install them with:
```bash
pip install -r requirements.txt
```

---
3. Start the frontend
In another terminal(Make sure the backend is started as well), navigate to the frontend project:
```bash
cd captions-app
```
Then start the Vite development server:
```bash
npm run dev
```
---
Notes
Ensure Node.js and npm are installed before running the frontend.
Ensure the Python virtual environment is properly set up before starting the backend.

---
**Recorded demonstration**
A recorded short demonstration of the artefact in action is available at: 
https://youtube.com/shorts/6Vz0KpY56D4?feature=share
