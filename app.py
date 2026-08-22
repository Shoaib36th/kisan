import os
import io
import tempfile
import asyncio
import requests
import streamlit as st
import streamlit.components.v1 as components
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

st.set_page_config(page_title="Kisan Voice Advisor", page_icon="🌾", layout="centered")

# ---------------- Global theme / CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@600;700&family=Poppins:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
    background:
        linear-gradient(rgba(5,19,11,0.96), rgba(5,19,11,0.98)),
        repeating-linear-gradient(0deg, rgba(34,197,94,0.06) 0px, rgba(34,197,94,0.06) 1px, transparent 1px, transparent 40px),
        repeating-linear-gradient(90deg, rgba(34,197,94,0.06) 0px, rgba(34,197,94,0.06) 1px, transparent 1px, transparent 40px),
        #05130b;
    color: #eafff1;
}

#MainMenu, footer, header {visibility: hidden;}

.kv-navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 4px 22px 4px; border-bottom: 1px solid rgba(34,197,94,0.15);
    margin-bottom: 28px;
}
.kv-brand { display: flex; align-items: center; gap: 12px; }
.kv-logo {
    font-size: 26px;
    filter: drop-shadow(0 0 10px rgba(34,197,94,0.8));
}
.kv-brand-text .kv-title-en { font-weight: 800; font-size: 19px; color: #f4fff7; line-height: 1.1; }
.kv-brand-text .kv-title-ur { font-family: 'Noto Nastaliq Urdu', serif; color: #9be8ac; font-size: 15px; }
.kv-badge {
    display: flex; align-items: center; gap: 8px;
    background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.45);
    color: #6ee7a4; padding: 8px 16px; border-radius: 999px; font-size: 13px; font-weight: 600;
}

.kv-hero { text-align: center; padding: 10px 0 34px 0; }
.kv-hero-icon { font-size: 44px; filter: drop-shadow(0 0 18px rgba(34,197,94,0.9)); margin-bottom: 6px; }
.kv-hero-ur {
    font-family: 'Noto Nastaliq Urdu', serif; font-size: 26px; color: #ffd75e;
    text-shadow: 0 0 14px rgba(255,215,94,0.35); margin: 6px 0 14px 0; direction: rtl;
}
.kv-hero h1 {
    font-size: 40px; font-weight: 800; line-height: 1.15; margin: 0;
    color: #f4fff7;
}
.kv-hero h1 .grad {
    background: linear-gradient(90deg, #22c55e, #bff56a);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    text-shadow: 0 0 26px rgba(34,197,94,0.35);
}
.kv-hero p { color: #a9c9b6; font-size: 15.5px; margin-top: 14px; max-width: 480px; margin-left:auto; margin-right:auto; }

.kv-card {
    background: linear-gradient(180deg, rgba(15,38,24,0.75), rgba(8,24,15,0.75));
    border: 1px solid rgba(34,197,94,0.22);
    border-radius: 22px; padding: 28px 24px; margin-bottom: 22px;
    box-shadow: 0 0 0 1px rgba(34,197,94,0.03), 0 20px 50px -20px rgba(0,0,0,0.6);
}
.kv-section-label {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: 'Noto Nastaliq Urdu', serif; font-size: 20px; color: #eafff1;
    margin-bottom: 2px; direction: rtl;
}
.kv-section-label-en {
    font-size: 11px; letter-spacing: 2px; color: #6ee7a4; text-transform: uppercase;
    margin-bottom: 14px; margin-top: 2px; font-weight: 600;
}
.kv-mic-wrap { text-align: center; padding: 6px 0 4px 0; }
.kv-mic-caption-ur { font-family: 'Noto Nastaliq Urdu', serif; color: #6ee7a4; font-size: 18px; margin-top: 14px; direction: rtl; }
.kv-mic-caption-en { color: #8fb8a0; font-size: 13px; margin-top: 2px; }

.kv-chips { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; margin-top: 20px; }
.kv-chip {
    font-family: 'Noto Nastaliq Urdu', serif; direction: rtl;
    border: 1px solid rgba(34,197,94,0.3); background: rgba(34,197,94,0.06);
    color: #dff7e6; padding: 8px 16px; border-radius: 999px; font-size: 14px;
}

/* mic_recorder component container glow */
iframe {
    filter: drop-shadow(0 0 24px rgba(34,197,94,0.25));
}

/* File uploader restyle to look like the dropzone mock */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(34,197,94,0.4) !important;
    border-radius: 18px !important;
    background: rgba(34,197,94,0.04) !important;
    padding: 10px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
}

/* "Browse files" button */
[data-testid="stFileUploader"] button,
[data-testid="stBaseButton-secondary"] {
    background: linear-gradient(180deg, #16321f, #0c2015) !important;
    color: #eafff1 !important;
    border: 1px solid rgba(34,197,94,0.55) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 16px rgba(34,197,94,0.2) !important;
}
[data-testid="stFileUploader"] button:hover {
    border-color: #22c55e !important;
    box-shadow: 0 0 24px rgba(34,197,94,0.4) !important;
}
[data-testid="stFileUploader"] small { color: #8fb8a0 !important; }

/* Uploaded-file preview chip */
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] * ,
[class*="fileUploaderFile"],
[class*="fileUploaderFile"] * {
    background: transparent !important;
    color: #eafff1 !important;
}
[data-testid="stFileUploaderFile"] {
    background: rgba(15,38,24,0.9) !important;
    border: 1px solid rgba(34,197,94,0.3) !important;
    border-radius: 12px !important;
    padding: 6px 10px !important;
}
[data-testid="stFileUploaderFile"] svg,
[class*="fileUploaderFile"] svg {
    fill: #6ee7a4 !important;
    stroke: #6ee7a4 !important;
}
[data-testid="stFileUploaderFileName"] { color: #eafff1 !important; }
[data-testid="stFileUploaderFileSize"], small[class*="fileUploaderFileData"] { color: #8fb8a0 !important; }

.kv-divider { text-align:center; color:#3f6650; margin: 8px 0 30px 0; font-size: 12px; letter-spacing:3px; text-transform:uppercase; }
.kv-divider::before, .kv-divider::after { content:""; }

.kv-footnote { color: #7fa48c; font-size: 12.5px; text-align:center; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------- Navbar ----------------
st.markdown("""
<div class="kv-navbar">
    <div class="kv-brand">
        <div class="kv-logo">🌾</div>
        <div class="kv-brand-text">
            <div class="kv-title-en">Kisan Voice Advisor</div>
            <div class="kv-title-ur">کسان وائس ایڈوائزر</div>
        </div>
    </div>
    <div class="kv-badge">🔊 Urdu Voice AI</div>
</div>
""", unsafe_allow_html=True)

# ---------------- Hero ----------------
st.markdown("""
<div class="kv-hero">
    <div class="kv-hero-icon">🌾</div>
    <div class="kv-hero-ur">اپنی زبان میں سوال پوچھیں</div>
    <h1>Your farm, <span class="grad">heard &amp;<br/>answered</span></h1>
    <p>Speak your farming question in Urdu and listen to instant expert advice —
    or scan a sick leaf and get a treatment plan in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Voice Q&A ----------------
st.markdown('<div class="kv-card">', unsafe_allow_html=True)
st.markdown('<div class="kv-section-label">آواز سے پوچھیں</div>', unsafe_allow_html=True)
st.markdown('<div class="kv-section-label-en">Ask by voice</div>', unsafe_allow_html=True)

st.markdown('<div class="kv-mic-wrap">', unsafe_allow_html=True)
audio = mic_recorder(
    start_prompt="🎤 Speak your question",
    stop_prompt="⏹ Stop recording",
    key="recorder"
)
st.markdown("""
    <div class="kv-mic-caption-ur">مائیک دبائیں اور سوال پوچھیں</div>
    <div class="kv-mic-caption-en">Tap the mic and ask your farming question in Urdu</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# The mic button is rendered by streamlit_mic_recorder inside its own sandboxed
# iframe, so page-level CSS can't reach it. This invisible helper iframe reaches
# up to the parent page, finds that iframe (same-origin), and injects a
# stylesheet into it so the button matches the rest of the theme. Re-applied on
# an interval since Streamlit can redraw the component on rerun.
components.html("""
<script>
function kvStyleMicRecorder() {
    try {
        const doc = window.parent.document;
        const frames = doc.querySelectorAll('iframe');
        frames.forEach(f => {
            const title = (f.title || "").toLowerCase();
            if (title.includes("mic_recorder")) {
                const idoc = f.contentDocument || (f.contentWindow && f.contentWindow.document);
                if (idoc && idoc.head && !idoc.getElementById('kv-mic-style')) {
                    const style = idoc.createElement('style');
                    style.id = 'kv-mic-style';
                    style.innerHTML = `
                        body { background: transparent !important; }
                        .myButton {
                            background: linear-gradient(180deg, #16321f, #0c2015) !important;
                            border: 1px solid rgba(34,197,94,0.55) !important;
                            color: #eafff1 !important;
                            border-radius: 999px !important;
                            padding: 12px 28px !important;
                            font-weight: 600 !important;
                            font-size: 15px !important;
                            font-family: 'Poppins', sans-serif !important;
                            box-shadow: 0 0 20px rgba(34,197,94,0.25) !important;
                            transition: box-shadow 0.2s ease, border-color 0.2s ease;
                        }
                        .myButton:hover {
                            border-color: #22c55e !important;
                            box-shadow: 0 0 28px rgba(34,197,94,0.5) !important;
                        }
                    `;
                    idoc.head.appendChild(style);
                }
            }
        });
    } catch (e) { /* cross-origin fallback: silently skip */ }
}
kvStyleMicRecorder();
setInterval(kvStyleMicRecorder, 700);
</script>
""", height=0)

st.markdown("""
<div class="kv-chips">
    <div class="kv-chip">میرے گندم کے پتوں پر پیلے دھبے ہیں، کیا کروں؟</div>
    <div class="kv-chip">کپاس کو کتنا پانی دوں؟</div>
    <div class="kv-chip">گندم کے لیے کون سی کھاد بہتر ہے؟</div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

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

st.markdown('<div class="kv-divider">— or —</div>', unsafe_allow_html=True)

# ---------------- Crop Disease Photo Upload ----------------
st.markdown('<div class="kv-card">', unsafe_allow_html=True)
st.markdown('<div class="kv-section-label">📷 فصل کی تصویر سکین کریں</div>', unsafe_allow_html=True)
st.markdown('<div class="kv-section-label-en">Scan a sick crop</div>', unsafe_allow_html=True)
uploaded_image = st.file_uploader(
    "پتے کی تصویر یہاں ڈالیں یا منتخب کریں  ·  JPG or PNG · tap to browse or drop a photo",
    type=["jpg", "jpeg", "png"]
)
st.markdown('</div>', unsafe_allow_html=True)

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
