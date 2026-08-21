import html
import io
import os

import numpy as np
import requests
import scipy.io.wavfile as wav
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Deepgram Human-Like Speech Agent",
    page_icon="🎙️",
    layout="centered",
)

load_dotenv()


# ============================================================
# DEEPGRAM API KEY
# ============================================================

DEEPGRAM_API_KEY = (
    os.getenv("DEEPGRAM_API_KEY")
    or st.secrets.get("DEEPGRAM_API_KEY", None)
)

if not DEEPGRAM_API_KEY:
    st.error(
        "❌ DEEPGRAM_API_KEY missing hai. "
        "Apni .env file ya Streamlit Secrets check karein."
    )
    st.stop()


# ============================================================
# DEEPGRAM CONFIG
# ============================================================

DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"

DEEPGRAM_MODEL = "nova-3"

# Urdu speech recognition
DEEPGRAM_LANGUAGE = "ur"

DEEPGRAM_TIMEOUT = 60


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def process_audio(audio_bytes):
    """
    Basic safe audio preparation.

    IMPORTANT:
    We do NOT aggressively remove quiet sounds because doing so
    can destroy quiet speech.
    """

    try:
        audio_file = io.BytesIO(audio_bytes)

        sample_rate, audio_data = wav.read(
            audio_file
        )

        if len(audio_data) == 0:
            return None

        # Convert stereo -> mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(
                audio_data,
                axis=1
            )

        audio_data = audio_data.astype(
            np.float64
        )

        # Remove DC offset
        audio_data -= np.mean(audio_data)

        # Duration check
        duration = (
            len(audio_data)
            / float(sample_rate)
        )

        if duration < 0.25:
            return None

        # Normalize safely
        peak = np.max(
            np.abs(audio_data)
        )

        if peak <= 0:
            return None

        target_peak = 0.90

        audio_data = (
            audio_data / peak
        ) * (
            32767 * target_peak
        )

        audio_data = np.clip(
            audio_data,
            -32768,
            32767
        ).astype(np.int16)

        output = io.BytesIO()

        wav.write(
            output,
            sample_rate,
            audio_data
        )

        output.seek(0)

        return output.read()

    except Exception:
        return None


# ============================================================
# DEEPGRAM SPEECH AGENT
# ============================================================

def transcribe_with_deepgram(audio_bytes):

    params = {
        "model": DEEPGRAM_MODEL,
        "language": DEEPGRAM_LANGUAGE,

        # Formatting
        "smart_format": "true",
        "punctuate": "true",
        "numerals": "true",

        # Speech segmentation
        "utterances": "true",
        "paragraphs": "false",

        # Word information
        "words": "true",

        # Detect speech better
        "vad_events": "true",
    }

    headers = {
        "Authorization": (
            f"Token {DEEPGRAM_API_KEY}"
        ),
        "Content-Type": "audio/wav",
    }

    response = requests.post(
        DEEPGRAM_API_URL,
        params=params,
        headers=headers,
        data=audio_bytes,
        timeout=DEEPGRAM_TIMEOUT,
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"Deepgram error "
            f"{response.status_code}: "
            f"{response.text[:1000]}"
        )

    data = response.json()

    results = data.get(
        "results",
        {}
    )

    channels = results.get(
        "channels",
        []
    )

    if not channels:
        return {
            "text": "",
            "confidence": 0.0,
            "words": [],
        }

    alternatives = channels[0].get(
        "alternatives",
        []
    )

    if not alternatives:
        return {
            "text": "",
            "confidence": 0.0,
            "words": [],
        }

    alternative = alternatives[0]

    transcript = (
        alternative.get(
            "transcript",
            ""
        )
        or ""
    ).strip()

    confidence = float(
        alternative.get(
            "confidence",
            0.0
        )
        or 0.0
    )

    words = alternative.get(
        "words",
        []
    )

    return {
        "text": transcript,
        "confidence": confidence,
        "words": words,
    }


# ============================================================
# SESSION STATE
# ============================================================

if "transcription" not in st.session_state:
    st.session_state.transcription = ""

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0


# ============================================================
# UI
# ============================================================

st.title(
    "🎙️ Deepgram Speech Agent"
)

st.caption(
    "Deepgram-only speech recognition with "
    "foreground speech detection and safe audio processing."
)

st.subheader(
    "🎤 Speak naturally"
)

st.write(
    "Background noise ho sakta hai. "
    "Normal voice mein baat karein aur recording stop karein."
)


# ============================================================
# MICROPHONE
# ============================================================

audio_output = mic_recorder(
    start_prompt="🎙️ Start Listening",
    stop_prompt="🛑 Stop & Process",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="deepgram_speech_agent",
)


# ============================================================
# PROCESS AUDIO
# ============================================================

if audio_output and "bytes" in audio_output:

    audio_bytes = audio_output["bytes"]

    if audio_bytes:

        with st.spinner(
            "🎧 Processing speech..."
        ):

            processed_audio = process_audio(
                audio_bytes
            )

        if processed_audio is None:

            st.warning(
                "⚠️ Audio bohat short ya empty hai. "
                "Dobara try karein."
            )

        else:

            with st.spinner(
                "🧠 Deepgram listening..."
            ):

                try:

                    result = (
                        transcribe_with_deepgram(
                            processed_audio
                        )
                    )

                    text = result["text"]

                    confidence = result[
                        "confidence"
                    ]

                    if text:

                        st.session_state.transcription = text

                        st.session_state.confidence = (
                            confidence
                        )

                        st.success(
                            "✅ Speech recognized!"
                        )

                    else:

                        st.warning(
                            "⚠️ Speech detect nahi hui."
                        )

                except Exception as e:

                    st.error(
                        f"❌ Error: {e}"
                    )


# ============================================================
# OUTPUT
# ============================================================

st.divider()

st.subheader(
    "📝 Transcription"
)

if st.session_state.transcription:

    safe_text = html.escape(
        st.session_state.transcription
    )

    st.markdown(
        f"""
        <div style="
            padding: 20px;
            border-radius: 12px;
            background: #1e1e2e;
            border: 1px solid #45475a;
            margin-top: 10px;
        ">

            <div style="
                color: #89b4fa;
                font-weight: bold;
                margin-bottom: 10px;
            ">
                🎧 Deepgram Output
            </div>

            <div style="
                color: #cdd6f4;
                font-size: 1.3rem;
                line-height: 1.6;
            ">
                {safe_text}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Confidence: "
        f"{st.session_state.confidence:.2f}"
    )

else:

    st.info(
        "Aapki speech ka transcript yahan show hoga."
    )


# ============================================================
# RESET
# ============================================================

st.divider()

if st.button(
    "🗑️ Clear",
    use_container_width=True
):

    st.session_state.transcription = ""
    st.session_state.confidence = 0.0

    st.rerun()
