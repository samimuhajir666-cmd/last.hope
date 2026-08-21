import html
import io
import os
import re
import numpy as np
import requests
import scipy.io.wavfile as wav
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder

# ============================
# 🖥️ STREAMLIT PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Human-Like Roman Urdu Voice Agent",
    page_icon="🧠",
    layout="centered",
)
load_dotenv()

# ============================
# 🔑 API KEYS CONFIGURATION
# ============================
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY") or st.secrets.get("DEEPGRAM_API_KEY", None)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)

if not DEEPGRAM_API_KEY or not GROQ_API_KEY:
    st.error("❌ DEEPGRAM_API_KEY ya GROQ_API_KEY missing hain. Apni GitHub Secrets ya .env file check karein.")
    st.stop()

# ============================
# ⚙️ CONFIGURATIONS
# ============================
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_LANGUAGE = "ur"
DEEPGRAM_TIMEOUT = 60

# ============================
# 🧠 HUMAN-LIKE ATTENTION & CONTEXT RECONSTRUCTION
# ============================
def human_like_context_correction(raw_transcript, confidence):
    """
    Acts like human attention and context inference:
    - High confidence: Transcribe normally.
    - Medium confidence: Use context to correct/reconstruct small unclear words.
    - Low confidence / Unclear: Do not guess, mark as [inaudible].
    - STRICT RULE: Never invent sentences, facts, or phrases that were not spoken.
    """
    if not raw_transcript or not raw_transcript.strip():
        return "[inaudible]", 0.0

    # If confidence is extremely low according to Deepgram
    if confidence < 0.40:
        return "[inaudible]", confidence

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the human-attention and cognitive speech-processing core of an AI voice agent. "
                        "Your job is to process raw transcriptions of spoken Roman Urdu + English. "
                        "STRICT RULES:\n"
                        "1. Preserve the speaker's actual words and meaning as accurately as possible.\n"
                        "2. Handle filler words ('um', 'uh'), pauses, and natural speech flow.\n"
                        "3. If a word is partially masked by noise or unclear, use the immediate sentence context to reconstruct ONLY that specific unclear word or short phrase (e.g., fixing a minor typo or masked word).\n"
                        "4. NEVER invent, fabricate, or add entire sentences, additional phrases, facts, or ideas that the speaker did not say.\n"
                        "5. If a section is genuinely unintelligible or unclear, output '[inaudible]' for that part.\n"
                        "6. Output ONLY the final processed Roman Urdu text. No conversational filler, quotes, or notes."
                    ),
                },
                {
                    "role": "user", 
                    "content": f"Raw Transcript: {raw_transcript}\nConfidence Score: {confidence}"
                },
            ],
            temperature=0.0,  # Zero temperature to prevent hallucinations/inventing text
            max_tokens=800,
        )
        corrected_text = response.choices[0].message.content.strip()
        # Clean any accidental Arabic script output
        corrected_text = re.sub(r"[\u0600-\u06FF\u0900-\u097F]", "", corrected_text)
        return corrected_text.strip(), confidence
    except Exception as e:
        return raw_transcript, confidence

# ============================
# 🎙️ DEEPGRAM STT ENGINE
# ============================
def transcribe_audio_stream(processed_bytes, debug=False):
    params = {
        "model": DEEPGRAM_MODEL,
        "language": DEEPGRAM_LANGUAGE,
        "smart_format": "true",
        "punctuate": "true",
        "numerals": "true",
    }

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }

    try:
        response = requests.post(
            DEEPGRAM_API_URL,
            params=params,
            headers=headers,
            data=processed_bytes,
            timeout=DEEPGRAM_TIMEOUT,
        )
    except requests.RequestException as e:
        if debug:
            st.exception(e)
        raise RuntimeError(f"Deepgram connection failed: {e}") from e

    if response.status_code != 200:
        detail = response.text[:1200]
        raise RuntimeError(f"Deepgram API error ({response.status_code}): {detail}")

    try:
        data = response.json()
    except Exception as e:
        raise RuntimeError("Invalid JSON response from Deepgram.") from e

    results = data.get("results", {})
    channels = results.get("channels", [])

    if not channels:
        return {"text": "[inaudible]", "confidence": 0.0}

    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return {"text": "[inaudible]", "confidence": 0.0}

    alternative = alternatives[0]
    transcript = (alternative.get("transcript") or "").strip()
    confidence = float(alternative.get("confidence", 0.0) or 0.0)

    if not transcript:
        return {"text": "[inaudible]", "confidence": 0.0}

    # Pass through human-like context & attention correction filter
    final_text, final_confidence = human_like_context_correction(transcript, confidence)

    return {
        "text": final_text,
        "confidence": final_confidence,
    }

# ============================
# 🎚️ NOISE SUPPRESSION & AUDIO PRE-PROCESSING
# ============================
def process_audio_buffer(audio_bytes):
    try:
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)

        if sample_rate <= 0 or len(audio_data) == 0:
            return None

        # Convert to Mono if Stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        audio_data = audio_data.astype(np.float64)
        duration = len(audio_data) / float(sample_rate)

        # Ignore accidental ultra-short clicks (< 0.25 seconds)
        if duration < 0.25:
            return None

        # Basic Noise Suppression & Dynamic Range Normalization
        # Ignores low-level background humming/rumbling floors
        noise_floor = np.percentile(np.abs(audio_data), 10)
        audio_data[np.abs(audio_data) < noise_floor * 1.2] = 0

        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = (audio_data / max_val) * 32767.0

        processed_audio = np.clip(audio_data, -32768, 32767).astype(np.int16)

        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, processed_audio)
        output_buffer.seek(0)

        return output_buffer.read()

    except Exception as e:
        return None

# ============================
# 🧠 SESSION STATE MANAGEMENT
# ============================
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = ""
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

# ============================
# 🖥️ STREAMLIT UI DESIGN
# ============================
st.title("🧠 Human-Like Voice Agent (STT)")
st.caption("Listens with human attention, suppresses background noise, and applies strict context correction.")

debug_mode = st.checkbox("🐞 Show Technical Debug Errors", value=False)

st.subheader("🎙️ Live Speech Input")
st.write("Mic button click karein, shor-sharabe mein bhi normal baat karein, phir stop karein:")

audio_output = mic_recorder(
    start_prompt="🎙️ Start Listening",
    stop_prompt="🛑 Stop & Process",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="human_like_agent_mic",
)

# ============================
# ⚡ EXECUTION PIPELINE
# ============================
if audio_output and "bytes" in audio_output:
    audio_bytes = audio_output["bytes"]

    if len(audio_bytes) > 0:
        with st.spinner("🎧 Filtering noise & focusing on primary speaker..."):
            processed_bytes = process_audio_buffer(audio_bytes)

        if processed_bytes is None:
            st.warning("⚠️ Recording bohat choti thi ya sirf background shor detect hua. Dobara koshish karein.")
        else:
            with st.spinner("🧠 Applying human-like attention & context recognition..."):
                try:
                    result = transcribe_audio_stream(processed_bytes, debug=debug_mode)
                    transcription = result["text"]
                    confidence = result["confidence"]

                    if transcription:
                        st.session_state.agent_memory = transcription
                        st.session_state.last_confidence = confidence
                        st.success("✅ Processed successfully!")
                    else:
                        st.warning("⚠️ Koi saaf speech detect nahi hui.")

                except Exception as e:
                    st.error(f"❌ Pipeline Error: {e}")
                    if debug_mode:
                        st.exception(e)

# ============================
# 📝 DISPLAY AGENT OUTPUT
# ============================
st.divider()
st.subheader("📝 Agent Transcription Output")

if st.session_state.agent_memory:
    safe_text = html.escape(st.session_state.agent_memory)
    st.markdown(
        f"""
        <div style="padding: 18px; border-radius: 10px; background-color: #1e1e2e; border: 1px solid #45475a; margin-top: 10px;">
            <div style="font-weight: bold; color: #89b4fa; margin-bottom: 8px; font-size: 1.1em;">Focused Speaker Input:</div>
            <div style="font-size: 1.25em; color: #cdd6f4; font-weight: 500; line-height: 1.5;">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_confidence is not None:
        st.caption(f"Attention Confidence Score: {st.session_state.last_confidence:.2f}")
else:
    st.info("Aapki awaz ka processed text yahan show hoga.")

# ============================
# 🛠️ SESSION CONTROLS
# ============================
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("🔒 Lock Input", use_container_width=True):
        if st.session_state.agent_memory:
            st.success("Input locked and ready for your agent workflow!")
        else:
            st.warning("Koi text maujood nahi hai.")
with col2:
    if st.button("🗑️ Reset Memory", use_container_width=True):
        st.session_state.agent_memory = ""
        st.session_state.last_confidence = None
        st.rerun()
