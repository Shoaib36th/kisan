import os
import requests
import streamlit as st
from streamlit_mic_recorder import mic_recorder

# ============================================================
# CONFIG — fill these in before running
# ============================================================

# Your Hugging Face token is read from an environment variable.
# Set it in your terminal BEFORE running this app:
#   Windows (PowerShell):  $env:HF_TOKEN="your_token_here"
#   Mac/Linux:              export HF_TOKEN=your_token_here
HF_TOKEN = os.environ.get("HF_TOKEN")

# Replace this with your actual n8n webhook URL from Day 1
N8N_WEBHOOK_URL = "https://your-instance.app.n8n.cloud/webhook/kisan"

WHISPER_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
TTS_API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-urd"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def transcribe_audio(audio_bytes):
    """Send recorded audio to Whisper and return the transcribed text."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        response = requests.post(WHISPER_API_URL, headers=headers, data=audio_bytes, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("text", "").strip()
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return ""


def get_agent_reply(query):
    """Send the transcribed text to the n8n multi-agent webhook and return its reply."""
    try:
        response = requests.post(N8N_WEBHOOK_URL, json={"query": query}, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("reply", "Sorry, I couldn't get a response from the advisor.")
    except Exception as e:
        st.error(f"Could not reach the agent backend: {e}")
        return ""


def text_to_speech(text):
    """Convert text into spoken audio using a Hugging Face TTS model."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        response = requests.post(TTS_API_URL, headers=headers, json={"inputs": text}, timeout=60)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Text-to-speech failed: {e}")
        return None


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Kisan Voice Advisor", page_icon="🌾")
st.title("🌾 Kisan Voice Advisor")
st.write("Tap the mic, ask your farming question, and get a spoken answer back.")

if not HF_TOKEN:
    st.warning("HF_TOKEN environment variable is not set. Transcription and speech will not work until you set it.")

audio = mic_recorder(
    start_prompt="🎤 Speak your question",
    stop_prompt="⏹ Stop recording",
    key="recorder"
)

if audio:
    st.audio(audio["bytes"])

    with st.spinner("Transcribing..."):
        query_text = transcribe_audio(audio["bytes"])

    if query_text:
        st.write("**You said:**", query_text)

        with st.spinner("Thinking..."):
            reply_text = get_agent_reply(query_text)

        if reply_text:
            st.write("**Advisor says:**", reply_text)

            with st.spinner("Speaking..."):
                reply_audio = text_to_speech(reply_text)

            if reply_audio:
                st.audio(reply_audio, autoplay=True)
    else:
        st.warning("Couldn't understand the audio. Please try recording again, speaking clearly.")
