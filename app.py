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
# FARM SIMULATOR (isolated game logic — does not touch the advisor above)
# ============================================================

CROPS = {
    "Wheat (گندم)": {"base_yield": 100},
    "Cotton (کپاس)": {"base_yield": 90},
    "Rice (چاول)": {"base_yield": 110},
}

GROWTH_STAGES = ["🌰 Seed", "🌱 Sprout", "🌿 Growing", "🌾 Maturing", "🏆 Harvest"]

ROUNDS = [
    {
        "title": "🌰 Round 1: Planting",
        "question": "Time to plant. How much water do you use to prep the soil?",
        "choices": {
            "Water generously": {"yield_change": 8, "water_cost": 20, "money_cost": 5},
            "Use minimal water": {"yield_change": -3, "water_cost": 5, "money_cost": 0},
            "Ask the Advisor first": "ask_weather",
        },
    },
    {
        "title": "🌱 Round 2: Irrigation Decision",
        "question": "It hasn't rained in a week. What do you do?",
        "choices": {
            "Irrigate now": {"yield_change": 10, "water_cost": 15, "money_cost": 5},
            "Wait 2 more days": {"yield_change": -5, "water_cost": 0, "money_cost": 0},
            "Ask the Advisor first": "ask_weather",
        },
    },
    {
        "title": "🌿 Round 3: Fertilizer Decision",
        "question": "Your crop looks a bit pale. What do you do?",
        "choices": {
            "Apply fertilizer now": {"yield_change": 15, "water_cost": 5, "money_cost": 10},
            "Skip it this season": {"yield_change": -10, "water_cost": 0, "money_cost": 0},
            "Ask the Advisor first": "ask_fertilizer",
        },
    },
    {
        "title": "🐛 Round 4: Pest Attack!",
        "question": "You spot pests on a few leaves. What do you do?",
        "choices": {
            "Spray pesticide immediately": {"yield_change": 12, "water_cost": 0, "money_cost": 15},
            "Ignore it, hope it's minor": {"yield_change": -15, "water_cost": 0, "money_cost": 0},
            "Ask the Advisor first": "ask_disease",
        },
    },
    {
        "title": "🌾 Round 5: Weather Surprise",
        "question": "An unexpected heatwave is coming. What do you do?",
        "choices": {
            "Irrigate extra to protect crop": {"yield_change": 10, "water_cost": 20, "money_cost": 5},
            "Do nothing, save resources": {"yield_change": -8, "water_cost": 0, "money_cost": 0},
            "Ask the Advisor first": "ask_weather",
        },
    },
    {
        "title": "💰 Round 6: Harvest & Selling",
        "question": "Harvest is ready. Market prices are fluctuating. What do you do?",
        "choices": {
            "Sell immediately": {"yield_change": 0, "water_cost": 0, "money_cost": 0},
            "Wait for a better price": {"yield_change": 5, "water_cost": 0, "money_cost": 0},
            "Ask the Advisor first": "ask_price",
        },
    },
]

ADVISOR_QUERIES = {
    "ask_weather": "Kya mujhe abhi apni fasal ko pani dena chahiye?",
    "ask_fertilizer": "Meri fasal thori si peeli lag rahi hai, kya mujhe khaad dalni chahiye?",
    "ask_disease": "Meri fasal par keere lag gaye hain, mujhe kya karna chahiye?",
    "ask_price": "Kya mujhe abhi apni fasal bechni chahiye ya rukna chahiye?",
}


def init_game_state():
    if "game_round" not in st.session_state:
        reset_game()


def reset_game():
    st.session_state.game_round = 0
    st.session_state.game_yield = 100
    st.session_state.game_water = 100
    st.session_state.game_money = 100
    st.session_state.game_crop = None
    st.session_state.game_log = []
    st.session_state.game_over = False


def run_farm_simulator():
    """Rule-based decision simulator with resource meters and growth stages.
    No extra AI calls unless the farmer taps 'Ask the Advisor first', which
    reuses the existing get_agent_reply() function already used above."""
    import time

    init_game_state()

    st.markdown('<div class="sim-card">', unsafe_allow_html=True)
    st.subheader("🎮 Farm Simulator")
    st.write("Grow your crop through 6 rounds of real farming decisions. Manage your water and money wisely — and ask your AI Advisor for help anytime!")

    if st.session_state.game_crop is None:
        crop_choice = st.selectbox("Choose your crop to begin:", list(CROPS.keys()))
        if st.button("Start Farming 🌾", use_container_width=True):
            st.session_state.game_crop = crop_choice
            st.session_state.game_yield = CROPS[crop_choice]["base_yield"]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    stage_idx = min(st.session_state.game_round, len(GROWTH_STAGES) - 1)
    st.markdown(f"### {GROWTH_STAGES[stage_idx]}  —  {st.session_state.game_crop}")

    c1, c2, c3 = st.columns(3)
    c1.metric("🌾 Yield", st.session_state.game_yield)
    c2.metric("💧 Water", f"{max(st.session_state.game_water, 0)}%")
    c3.metric("💰 Money", f"Rs {max(st.session_state.game_money, 0)}")

    st.progress(min(max(st.session_state.game_yield, 0), 150) / 150)

    if st.session_state.game_over:
        final_yield = st.session_state.game_yield
        if final_yield >= 140:
            st.balloons()
            st.success(f"🏆 Excellent farming! Final yield score: {final_yield}")
        elif final_yield >= 100:
            st.info(f"👍 Decent harvest. Final yield score: {final_yield}")
        else:
            st.warning(f"⚠️ Tough season. Final yield score: {final_yield}")

        st.write("**Your farming journey:**")
        for entry in st.session_state.game_log:
            st.write(f"- {entry}")

        if st.button("🔄 Play Again", use_container_width=True):
            reset_game()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    round_data = ROUNDS[st.session_state.game_round]
    st.markdown(f"#### {round_data['title']}")
    st.write(round_data["question"])

    cols = st.columns(len(round_data["choices"]))
    for idx, (choice_label, effect) in enumerate(round_data["choices"].items()):
        with cols[idx]:
            if st.button(choice_label, key=f"round{st.session_state.game_round}_{idx}", use_container_width=True):
                if isinstance(effect, str):
                    with st.spinner("🌾 Asking your advisor..."):
                        advice = get_agent_reply(ADVISOR_QUERIES[effect])
                    st.info(f"**Advisor says:** {advice}")
                    st.caption("Now pick a real decision based on this advice ⬆️")
                else:
                    with st.spinner("🌱 Growing..."):
                        time.sleep(0.6)
                    st.session_state.game_yield += effect["yield_change"]
                    st.session_state.game_water -= effect["water_cost"]
                    st.session_state.game_money -= effect["money_cost"]
                    st.session_state.game_log.append(
                        f"{round_data['title']}: chose '{choice_label}' ({'+' if effect['yield_change'] >= 0 else ''}{effect['yield_change']} yield)"
                    )
                    st.session_state.game_round += 1
                    if st.session_state.game_round >= len(ROUNDS):
                        st.session_state.game_over = True
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# STREAMLIT UI
# ============================================================

CUSTOM_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at 20% 20%, #0d2b1a 0%, #071a10 60%, #05130b 100%);
    color: #e8f5e9;
}
h1, h2, h3, h4 {
    color: #f2d675 !important;
}

/* ---- Buttons ---- */
.stButton>button {
    background: linear-gradient(135deg, #1f6b3a, #143d22);
    color: #eaffea;
    border: 1px solid #3fae5c;
    border-radius: 10px;
    padding: 0.5em 1em;
    transition: all 0.25s ease-in-out;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #2c8a4d, #1c5c30);
    border-color: #67d98a;
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 4px 14px rgba(76, 175, 80, 0.35);
}

/* ---- Simulator card ---- */
.sim-card {
    background: rgba(15, 40, 25, 0.55);
    border: 1px solid rgba(76, 175, 80, 0.35);
    border-radius: 16px;
    padding: 1.2em;
    margin-top: 0.5em;
    animation: fadeIn 0.5s ease-in-out;
}
[data-testid="stMetricValue"] {
    color: #7be495 !important;
}

/* ---- File uploader restyle ---- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(15, 40, 25, 0.55) !important;
    border: 1.5px dashed #3fae5c !important;
    border-radius: 14px !important;
    transition: all 0.25s ease-in-out;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #7be495 !important;
    background: rgba(20, 55, 33, 0.7) !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #1f6b3a, #143d22) !important;
    color: #eaffea !important;
    border: 1px solid #3fae5c !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div {
    color: #cfe9d6 !important;
}

/* ---- Mic recorder wrapper (partial styling — it's an embedded iframe) ---- */
.mic-wrapper {
    background: rgba(15, 40, 25, 0.55);
    border: 1.5px solid rgba(76, 175, 80, 0.35);
    border-radius: 14px;
    padding: 0.6em;
    margin-bottom: 0.5em;
    animation: fadeIn 0.5s ease-in-out;
}

/* ---- Floating background crop icons ---- */
.bg-float {
    position: fixed;
    font-size: 2rem;
    opacity: 0.12;
    z-index: 0;
    animation: floatUp 14s linear infinite;
    pointer-events: none;
}
.bg-float:nth-child(1) { left: 5%;  animation-delay: 0s;  }
.bg-float:nth-child(2) { left: 20%; animation-delay: 3s;  font-size: 2.5rem; }
.bg-float:nth-child(3) { left: 40%; animation-delay: 6s;  }
.bg-float:nth-child(4) { left: 65%; animation-delay: 2s;  font-size: 1.6rem; }
.bg-float:nth-child(5) { left: 80%; animation-delay: 8s;  }
.bg-float:nth-child(6) { left: 90%; animation-delay: 5s;  font-size: 2.2rem; }

@keyframes floatUp {
    0%   { transform: translateY(110vh) rotate(0deg);   opacity: 0; }
    10%  { opacity: 0.12; }
    90%  { opacity: 0.12; }
    100% { transform: translateY(-10vh) rotate(25deg);  opacity: 0; }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pageEntrance {
    from { opacity: 0; transform: scale(0.98); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes titleGlow {
    0%, 100% { text-shadow: 0 0 8px rgba(242, 214, 117, 0.4); }
    50%      { text-shadow: 0 0 20px rgba(242, 214, 117, 0.8); }
}
.stApp {
    animation: pageEntrance 0.8s ease-out;
}
h1 {
    animation: titleGlow 3s ease-in-out infinite;
}
</style>

<div class="bg-float">🌾</div>
<div class="bg-float">🌱</div>
<div class="bg-float">🍃</div>
<div class="bg-float">🌿</div>
<div class="bg-float">🌾</div>
<div class="bg-float">🍃</div>
"""

st.set_page_config(page_title="Kisan Voice Advisor", page_icon="🌾")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "welcomed" not in st.session_state:
    st.session_state.welcomed = True
    welcome_placeholder = st.empty()
    with welcome_placeholder.container():
        st.markdown(
            "<h2 style='text-align:center; animation: fadeIn 1s ease-in-out;'>🌾 Welcome to Kisan Voice Advisor 🌾</h2>"
            "<p style='text-align:center; color:#cfe9d6;'>Loading your farm assistant...</p>",
            unsafe_allow_html=True,
        )
    import time
    time.sleep(1.2)
    welcome_placeholder.empty()

st.title("🌾 Kisan Voice Advisor")

tab_advisor, tab_game = st.tabs(["🎙️ Voice Advisor", "🎮 Farm Simulator"])

with tab_advisor:
    st.write("Click the mic button, ask your farming question in Urdu, and listen to the reply.")

    # ---------------- Voice Q&A ----------------
    st.markdown('<div class="mic-wrapper">', unsafe_allow_html=True)
    audio = mic_recorder(
        start_prompt="🎤 Speak your question",
        stop_prompt="⏹ Stop recording",
        key="recorder"
    )
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

with tab_game:
    run_farm_simulator()
