import html
import io
import os
import re
import numpy as np
import requests
import scipy.io.wavfile as wav
import scipy.signal as signal
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder

# ============================
# 🖥️ STREAMLIT PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Speech to Text (Roman Urdu)",
    page_icon="🎤",
    layout="centered",
)
load_dotenv()

# ============================
# 🔑 API KEYS CONFIGURATION
# ============================
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY") or st.secrets.get("DEEPGRAM_API_KEY", None)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)

if not DEEPGRAM_API_KEY:
    st.error("❌ DEEPGRAM_API_KEY nahi mili. .env file ya Streamlit Secrets check karein.")
    st.stop()

# ============================
# 🎙️ DEEPGRAM & GROQ CONFIG
# ============================
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_LANGUAGE = "ur"  # Deepgram listens in native Urdu script
DEEPGRAM_TIMEOUT = 60

# ============================
# 🧹 CLEAN & TRANSLITERATE
# ============================
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"[’'‘`\^\~]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def convert_to_roman_script(text):
    """Converts Urdu Perso-Arabic script into natural Roman Urdu + English."""
    if not text or not text.strip():
        return ""

    if not GROQ_API_KEY:
        st.warning("⚠️ GROQ_API_KEY missing. Transliteration skipped.")
        return text

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert audio transcript converter. "
                        "Convert the input Urdu text into clean, simple, accurate ROMAN URDU (Latin script). "
                        "Keep technical or English words (e.g., 'API', 'Python', 'project', 'code') in English text. "
                        "STRICT RULE: Output ONLY the converted Roman text. Absolutely NO Urdu script or Hindi script. "
                        "Do not add introductions, quotes, or notes."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        converted = response.choices[0].message.content.strip()
        # Clean remaining Urdu characters if any
        converted = re.sub(r"[\u0600-\u06FF\u0900-\u097F]", "", converted)
        return converted.strip()
    except Exception as e:
        return text


# ============================
# 🎙️ DEEPGRAM TRANSCRIBE ENGINE
# ============================
def transcribe_with_deepgram(processed_bytes, debug=False):
    params = [
        ("model", DEEPGRAM_MODEL),
        ("language", DEEPGRAM_LANGUAGE),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("utterances", "true"),
        ("numerals", "true"),
    ]

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
        return {"text": "", "confidence": 0.0}

    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return {"text": "", "confidence": 0.0}

    alternative = alternatives[0]
    transcript = (alternative.get("transcript") or "").strip()
    confidence = float(alternative.get("confidence", 0.0) or 0.0)

    if not transcript:
        utterances = results.get("utterances") or []
        transcript = " ".join(
            (u.get("transcript") or "").strip() for u in utterances if u.get("transcript")
        ).strip()

    if not transcript:
        return {"text": "", "confidence": 0.0}

    # Transliterate Urdu text to Roman script using Groq
    roman_text = convert_to_roman_script(transcript)

    return {
        "text": clean_text(roman_text),
        "confidence": confidence,
    }


# ============================
# 🎚️ RELIABLE AUDIO PRE-PROCESSING
# ============================
def process_audio_buffer(audio_bytes, enhance_audio=False):
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

        # Basic duration guard (ignore under 0.1s)
        if duration < 0.10:
            return None

        # Light Optional Normalization
        if enhance_audio:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = (audio_data / max_val) * 32767.0

        processed_audio = np.clip(audio_data, -32768, 32767).astype(np.int16)

        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, processed_audio)
        output_buffer.seek(0)

        return {
            "processed_bytes": output_buffer.read(),
            "sample_rate": int(sample_rate),
            "duration": float(duration),
        }

    except Exception as e:
        return None


# ============================
# 🧠 SESSION STATE
# ============================
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

# ============================
# 🖥️ UI INTERFACE
# ============================
st.title("🎤 SPEECH TO TEXT (Roman Urdu)")
st.caption("Deepgram Nova-3 + Groq Llama-3.3 Engine")

enhance_audio = st.checkbox("✨ Light audio normalization", value=False)
debug_mode = st.checkbox("🐞 Show technical debug errors", value=False)

st.subheader("🎤 Voice Input")
st.write("Press Start, speak your lesson or audio, then press Stop.")

audio_output = mic_recorder(
    start_prompt="🎤 Click to Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="listener_mic",
)

# ============================
# 🧠 TRANSCRIPTION EXECUTION
# ============================
if audio_output and "bytes" in audio_output:
    audio_bytes = audio_output["bytes"]

    if len(audio_bytes) > 0:
        with st.spinner("⏳ Processing audio buffer..."):
            result = process_audio_buffer(audio_bytes, enhance_audio=enhance_audio)

        if result is None:
            st.warning("⚠️ Audio format invalid or recording too short.")
        else:
            processed_bytes = result["processed_bytes"]

            with st.spinner("⚡ Transcribing & Transliterating..."):
                try:
                    transcription_result = transcribe_with_deepgram(
                        processed_bytes, debug=debug_mode
                    )
                    text_from_voice = transcription_result["text"]
                    confidence = transcription_result["confidence"]

                    if text_from_voice:
                        st.session_state.last_transcription = text_from_voice
                        st.session_state.last_confidence = confidence
                        st.success("✅ Complete!")
                    else:
                        st.warning("⚠️ No clear speech detected. Speak clearly into the mic.")

                except Exception as e:
                    st.error(f"❌ Transcription error: {e}")
                    if debug_mode:
                        st.exception(e)

# ============================
# 📝 DISPLAY OUTPUT
# ============================
st.divider()
st.subheader("📝 Transcribed Text (Roman Script)")

if st.session_state.last_transcription:
    safe_text = html.escape(st.session_state.last_transcription)
    st.markdown(
        f"""
        <div style="padding: 18px; border-radius: 10px; background-color: #1e1e2e; border: 1px solid #45475a; margin-top: 10px;">
            <div style="font-weight: bold; color: #89b4fa; margin-bottom: 8px; font-size: 1.1em;">Result:</div>
            <div style="font-size: 1.25em; color: #cdd6f4; font-weight: 500; line-height: 1.5;">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_confidence is not None:
        st.caption(f"Deepgram confidence score: {st.session_state.last_confidence:.2f}")
else:
    st.info("Your transcription will appear here.")

# ============================
# 🛠️ SESSION CONTROLS
# ============================
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("🛑 Lock Text", use_container_width=True):
        if st.session_state.last_transcription:
            st.success("Saved in session.")
        else:
            st.warning("No text available.")
with col2:
    if st.button("🗑️ Clear Text", use_container_width=True):
        st.session_state.last_transcription = ""
        st.session_state.last_confidence = None
        st.rerun()
