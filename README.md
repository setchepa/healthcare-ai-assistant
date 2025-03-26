# 🏥 AI Assistant for Healthcare Provider Search and Verification

This project is a conversational AI agent that helps patients find and confirm availability of healthcare providers in real time. It uses:

- 🧠 **OpenAI Realtime API** for voice and chat interaction
- 📞 **Twilio** for phone call handling
- 🌐 **FastAPI** as the backend server
- 🕵️ **LangChain + Tavily** for intelligent provider search
- 🎤 **Whisper API** for transcribing call recordings

## 🌟 Features

- Real-time phone calls with clinics using AI voice
- Confirms new patient acceptance and insurance compatibility
- Personalized search for healthcare providers
- Automated transcription and analysis of calls
- Modular and extensible with LangChain tooling

## 📦 Installation

```bash
git clone https://github.com/your-username/healthcare-ai-assistant.git
cd healthcare-ai-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ⚙️ Configuration

Copy `.env.example` to `.env` and fill in:

```env
OPENAI_API_KEY=sk-...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
PHONE_NUMBER_FROM=+1...
TAVILY_API_KEY=...
LANGSMITH_API_KEY=...
DOMAIN=https://your-ngrok-url.io
PORT=5050
```

## 🌐 Enable ngrok Tunnel

To allow Twilio to reach your local FastAPI server, start a tunnel:

```bash
ngrok http 5050
```

Copy the HTTPS URL from the ngrok output (e.g., `https://abcd1234.ngrok.io`) and paste it into your `.env` as the `DOMAIN`.

## 🚀 Running the Project

Open two terminals:

### Terminal 1: Start the FastAPI server

```bash
uvicorn app.fastapi_server:app --host 0.0.0.0 --port 5050
```

### Terminal 2: Start the AI assistant

```bash
python run.py
```

## 📂 Project Structure

- `app/fastapi_server.py`: All FastAPI endpoints and Twilio media stream handling
- `app/healthcare_agent.py`: The LangChain-based conversational agent and call orchestration
- `run.py`: Entrypoint to run the assistant
- `docs/`: Contains design documentation

## 🛡️ License

[MIT](LICENSE)

## 🙌 Credits

Project by Karla Acosta, Santiago Etchepare, Archita Mishra, Maria Cristina Rojas (UChicago Booth)
