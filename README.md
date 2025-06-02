
# 📽️ Video QA and Summary Tool

Video QA and Summary Tool is a web app that lets you upload or link a video, auto-transcribes it, summarizes the content, and allows you to ask questions about it in a chatbot style. It uses Whisper for transcription and Mistral-7B for answering questions.

A Gradio-based web app that lets you:

- 🎞️ Upload or link a YouTube video
- 🧠 Transcribe speech using Whisper
- 📚 Summarize content
- ❓ Ask questions about the video using Mistral-7B
- 🔍 Search contextually via LangChain & FAISS

---

## ⚙️ Installation & Setup

### Step 1: Accept Model License on HuggingFace
- Go to the Mistral-7B-Instruct-v0.3 model page (https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- ➡️ Click “Agree and Access” to accept the terms

### Step 2: Set Your HuggingFace Token
- Create a file named `.env` in your project root and Paste your HuggingFace API token:
```env
HF_TOKEN="your_huggingface_token_here"
```
### Step 3: Install Requirements
Install Python dependencies:
✅ For Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
✅ For Windows (CMD/PowerShell):
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
### Step 4: Run the App
```cmd
python app.py
```

### Step 5: Open the Web UI
Once the app is running, Gradio will give you a link like:
```nginx
Running on local URL:  http://127.0.0.1:7860
```
Open this link in your browser to use the Video QA and Summary Tool 🎉

## 🚀 Demo Screenshot
![Image](https://github.com/user-attachments/assets/79808fe0-a73c-4ec6-9e9c-328054952e8d)
![Image](https://github.com/user-attachments/assets/c2538315-6969-4e2f-b453-8f6723bca867)
![Image](https://github.com/user-attachments/assets/5d97856e-a834-4cbc-a573-361fd4642bdc)

---

## 🧑‍💻 Tech Stack

- Python
- Gradio – for the web UI
- LangChain – for document processing & retrieval
- Whisper (OpenAI) – for audio transcription
- Mistral 7B (via HuggingFace) – for LLM-based Q&A
- FAISS – for semantic vector search
- LlamaIndex – HuggingFace LLM connector
- yt-dlp – for downloading YouTube audio
- MoviePy – for audio extraction from video

---



