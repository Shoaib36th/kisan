import os
import io
import tempfile
import asyncio
import requests
import streamlit as st
from groq import Groq
import edge_tts
from streamlit_mic_recorder import mic_recorder
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# ============================================================
# CONFIGURATION
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
N8N_WEBHOOK_URL = "https://shoaib15.app.n8n.cloud/webhook-test/kisan-query"

client = Groq(api_key=GROQ_API_KEY)

DISEASE_MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"


# ============================================================
# MODEL LOADING (cached so it only downloads/loads once)
# ============================================================

@st.cache_resource
def load_disease_model():
    """Download and load the crop disease classification model locally.
    The fine-tuned repo doesn't ship its own preprocessor_config.json, so we load
    the image processor from its base model (same preprocessing) and the classification
    weights from the fine-tuned disease model.
    Cached by Streamlit so this only happens once per app session, not on every upload."""
    processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
    model = AutoModelForImageClassification.from_pretrained(DISEASE_MODEL_NAME)
    model.eval()
    return processor, model


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def transcribe_audio_groq(audio_bytes):
    """Transcribe recorded Urdu audio into Urdu text using Groq's Whisper model."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(tmp_path, file.read()),
                model="whisper-large-v3-turbo",
                language="ur",
                response_format="json"
            )

        os.remove(tmp_path)
        return transcription.text.strip()
    except Exception as e:
        st.error(f"Groq Transcription failed: {e}")
        return ""


def classify_crop_disease(image_bytes):
    """Run the uploaded image through the locally-loaded ViT model and return the top prediction."""
    try:
        processor, model = load_disease_model()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_idx = probs.argmax(-1).item()
            confidence = probs[0][top_idx].item()

        label = model.config.id2label[top_idx]
        return label, confidence
    except Exception as e:
        st.error(f"Disease classification failed: {e}")
        return None, None


def get_agent_reply(query):
    """Send Urdu text (or a disease-derived question) to the n8n backend."""
    try:
        response = requests.post(N8N_WEBHOOK_URL, json={"query": query}, timeout=60)
        response.raise_for_status()
        data = response.json()

        reply = data.get("reply") or data.get("output")
        if not reply:
            reply = "معذرت، جواب حاصل نہیں ہو سکا۔"
        return reply
    except Exception as e:
        st.error(f"Could not reach the agent backend: {e}")
        return ""


async def _generate_edge_tts_bytes(text, voice="ur-PK-AsadNeural"):
    """Async generator to fetch high-quality Neural Urdu TTS bytes."""
    communicate = edge_tts.Communicate(text, voice)
    audio_stream = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])
    audio_stream.seek(0)
    return audio_stream.read()


def text_to_speech_neural(text, voice="ur-PK-AsadNeural"):
    """Convert Urdu response text to natural spoken Urdu voice using Edge Neural TTS."""
    try:
        return asyncio.run(_generate_edge_tts_bytes(text, voice=voice))
    except Exception as e:
        st.error(f"Neural Text-to-speech failed: {e}")
        return None


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Kisan Voice Advisor", page_icon="🌾")
st.title("🌾 Kisan Voice Advisor")
st.write("Click the mic button, ask your farming question in Urdu, and listen to the reply.")

# ---------------- Voice Q&A ----------------
audio = mic_recorder(
    start_prompt="🎤 Speak your question",
    stop_prompt="⏹ Stop recording",
    key="recorder"
)

if audio and audio.get("id") != st.session_state.get("last_audio_id"):
    st.session_state["last_audio_id"] = audio["id"]

    st.audio(audio["bytes"])

    with st.spinner("Listening..."):
        query_text = transcribe_audio_groq(audio["bytes"])

    if query_text:
        st.write("**You said (Urdu):**", query_text)

        with st.spinner("Processing reply..."):
            reply_text = get_agent_reply(query_text)

        if reply_text:
            st.write("**Advisor reply (Urdu):**", reply_text)

            with st.spinner("Generating voice response..."):
                reply_audio_bytes = text_to_speech_neural(reply_text, voice="ur-PK-AsadNeural")

            if reply_audio_bytes:
                st.audio(reply_audio_bytes, format="audio/mp3", autoplay=True)
    else:
        st.warning("Audio was unclear. Please try recording again, speaking clearly.")

st.divider()

# ---------------- Crop Disease Photo Upload ----------------
st.subheader("📷 Or upload a photo of a sick crop")
uploaded_image = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])

if uploaded_image and uploaded_image.file_id != st.session_state.get("last_image_id"):
    st.session_state["last_image_id"] = uploaded_image.file_id

    st.image(uploaded_image, caption="Uploaded photo", width=300)
    image_bytes = uploaded_image.read()

    with st.spinner("Analyzing photo (first run may take a minute while the model loads)..."):
        disease_label, confidence = classify_crop_disease(image_bytes)

    if disease_label:
        clean_label = disease_label.replace("___", " ").replace("_", " ").strip()

        if confidence < 0.6:
            st.warning(f"⚠️ Low confidence ({confidence:.0%}) — this doesn't look like a clear match for a known crop disease. Please upload a clear, close-up photo of a single affected leaf.")
        elif "healthy" in clean_label.lower():
            st.success(f"**Detected:** {clean_label} ({confidence:.0%} confidence)")
            st.write("**Advisor says:** ✅ Your plant looks healthy! No treatment needed — keep up your current care routine.")
        else:
            st.write(f"**Detected:** {clean_label} ({confidence:.0%} confidence)")

            with st.spinner("Getting advice..."):
                disease_query = f"Meri fasal ko {clean_label} bimari hai, mujhe kya karna chahiye?"
                reply_text = get_agent_reply(disease_query)

            if reply_text:
                st.write("**Advisor reply (Urdu):**", reply_text)

                with st.spinner("Generating voice response..."):
                    reply_audio_bytes = text_to_speech_neural(reply_text, voice="ur-PK-AsadNeural")

                if reply_audio_bytes:
                    st.audio(reply_audio_bytes, format="audio/mp3", autoplay=True)

    st.caption("⚠️ This is an AI estimate (not a substitute for expert agricultural diagnosis). For best results, use a clear, close-up, well-lit photo of a single affected leaf against a plain background.")
