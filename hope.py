import html
import io
import os
import re

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
    page_title="Human-Like Roman Urdu Speech Agent",
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

# ============================================================
# IMPORTANT
#
# "ur" = Urdu-focused recognition
# "multi" = Urdu + English mixed speech / code-switching
#
# Agar aap Roman Urdu + English naturally bolte hain,
# "multi" usually better choice hai.
# ============================================================

DEEPGRAM_LANGUAGE = "multi"

DEEPGRAM_TIMEOUT = 60


# ============================================================
# OPTIONAL KEYTERMS
#
# Yahan woh words/phrases likhein jo aap frequently bolte hain.
#
# Example:
# "ChatGPT"
# "Deepgram"
# "Streamlit"
# "Python"
# "API"
#
# Nova-3 keyterm prompting important terminology ki recognition
# improve karne ke liye use hoti hai.
# ============================================================

KEYTERMS = [
    "ChatGPT",
    "Deepgram",
    "Streamlit",
    "Python",
    "API",
]


# ============================================================
# ROMAN URDU / URDU SCRIPT DETECTION
# ============================================================

def contains_urdu_script(text):
    """
    Check karta hai ke output mein Urdu/Arabic script hai ya nahi.

    Ye transliteration nahi karta.

    Deepgram-only mode mein hum LLM se Urdu ko Roman Urdu mein
    convert nahi karwa rahe.
    """

    if not text:
        return False

    for char in text:
        code = ord(char)

        # Arabic + Urdu Unicode range
        if 0x0600 <= code <= 0x06FF:
            return True

        # Arabic Supplement
        if 0x0750 <= code <= 0x077F:
            return True

        # Arabic Extended
        if 0x08A0 <= code <= 0x08FF:
            return True

    return False


# ============================================================
# CLEAN TRANSCRIPT
# ============================================================

def clean_transcript(text):
    """
    Basic cleanup only.

    IMPORTANT:
    Ye transcript ko rewrite nahi karta.
    Ye sentence generate nahi karta.
    """

    if not text:
        return ""

    text = text.strip()

    # Normalize excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# SAFE ROMAN OUTPUT CHECK
# ============================================================

def validate_output(text):
    """
    Final safety layer.

    Agar Deepgram Urdu/Arabic script return kare to us output ko
    Roman Urdu keh kar display nahi karega.

    IMPORTANT:
    Hum yahan Urdu ko automatically transliterate nahi kar rahe,
    kyunki user ne Deepgram-only requirement di hai.
    """

    text = clean_transcript(text)

    if not text:
        return ""

    if contains_urdu_script(text):
        return ""

    return text


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def process_audio(audio_bytes):
    """
    Human-like safe audio preprocessing.

    Main objective:
    - Quiet speech ko destroy na karna
    - Background noise ko unnecessarily amplify na karna
    - Audio ko clip/distort na karna
    """

    try:

        audio_file = io.BytesIO(
            audio_bytes
        )

        sample_rate, audio_data = wav.read(
            audio_file
        )

        if (
            sample_rate <= 0
            or len(audio_data) == 0
        ):
            return None

        # ====================================================
        # STEREO -> MONO
        # ====================================================

        if len(audio_data.shape) > 1:

            audio_data = np.mean(
                audio_data,
                axis=1
            )

        audio_data = audio_data.astype(
            np.float64
        )

        # ====================================================
        # REMOVE DC OFFSET
        # ====================================================

        audio_data -= np.mean(
            audio_data
        )

        # ====================================================
        # DURATION CHECK
        # ====================================================

        duration = (
            len(audio_data)
            / float(sample_rate)
        )

        if duration < 0.25:
            return None

        # ====================================================
        # PEAK
        # ====================================================

        peak = np.max(
            np.abs(audio_data)
        )

        if peak <= 0:
            return None

        # ====================================================
        # IMPORTANT CHANGE
        #
        # OLD:
        #
        # audio_data / peak * 32767 * 0.90
        #
        # Ye har recording ko aggressively normalize karta tha.
        #
        # NEW:
        # Gentle gain only.
        # Maximum gain = 3x
        # ====================================================

        target_peak = 0.75 * 32767

        desired_gain = (
            target_peak / peak
        )

        gain = min(
            desired_gain,
            3.0
        )

        # Sirf jab audio genuinely quiet ho
        if desired_gain > 1.0:

            audio_data = (
                audio_data * gain
            )

        # Agar audio already loud hai,
        # usko unnecessary amplify nahi karna.

        # ====================================================
        # SOFT CLIP PROTECTION
        # ====================================================

        audio_data = np.clip(
            audio_data,
            -32768,
            32767
        )

        audio_data = audio_data.astype(
            np.int16
        )

        # ====================================================
        # WRITE WAV
        # ====================================================

        output = io.BytesIO()

        wav.write(
            output,
            sample_rate,
            audio_data
        )

        output.seek(0)

        return output.read()

    except Exception as e:

        if st.session_state.get(
            "debug_mode",
            False
        ):
            st.exception(e)

        return None


# ============================================================
# DEEPGRAM TRANSCRIPTION
# ============================================================

def transcribe_with_deepgram(
    audio_bytes
):

    # ========================================================
    # DEEPGRAM PARAMETERS
    # ========================================================

    params = {
        "model": DEEPGRAM_MODEL,

        # Urdu + English code switching
        "language": DEEPGRAM_LANGUAGE,

        # Formatting
        "smart_format": "true",
        "punctuate": "true",
        "numerals": "true",

        # Speech segmentation
        "utterances": "true",
        "paragraphs": "false",

        # Word-level information
        "words": "true",

        # VAD
        "vad_events": "true",

        # Keep natural speech
        "filler_words": "true",
    }

    # ========================================================
    # KEYTERMS
    #
    # Deepgram Nova-3 keyterm prompting.
    # ========================================================

    for term in KEYTERMS:

        # requests automatically URL-encodes params
        # correctly when list tuples are used.
        pass

    # Build params as list of tuples so the same query
    # parameter can appear multiple times.
    query_params = [
        (
            "model",
            DEEPGRAM_MODEL
        ),
        (
            "language",
            DEEPGRAM_LANGUAGE
        ),
        (
            "smart_format",
            "true"
        ),
        (
            "punctuate",
            "true"
        ),
        (
            "numerals",
            "true"
        ),
        (
            "utterances",
            "true"
        ),
        (
            "paragraphs",
            "false"
        ),
        (
            "words",
            "true"
        ),
        (
            "vad_events",
            "true"
        ),
        (
            "filler_words",
            "true"
        ),
    ]

    # Add keyterms
    for term in KEYTERMS:

        query_params.append(
            (
                "keyterm",
                term
            )
        )

    # ========================================================
    # HEADERS
    # ========================================================

    headers = {
        "Authorization": (
            f"Token {DEEPGRAM_API_KEY}"
        ),
        "Content-Type": "audio/wav",
    }

    # ========================================================
    # REQUEST
    # ========================================================

    response = requests.post(
        DEEPGRAM_API_URL,
        params=query_params,
        headers=headers,
        data=audio_bytes,
        timeout=DEEPGRAM_TIMEOUT,
    )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    if response.status_code != 200:

        raise RuntimeError(
            f"Deepgram error "
            f"{response.status_code}: "
            f"{response.text[:1500]}"
        )

    # ========================================================
    # JSON
    # ========================================================

    try:

        data = response.json()

    except Exception:

        raise RuntimeError(
            "Deepgram ne valid JSON return nahi kiya."
        )

    # ========================================================
    # RESULTS
    # ========================================================

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
            "roman_valid": True,
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
            "roman_valid": True,
        }

    alternative = alternatives[0]

    # ========================================================
    # RAW TRANSCRIPT
    # ========================================================

    raw_transcript = (
        alternative.get(
            "transcript",
            ""
        )
        or ""
    ).strip()

    # ========================================================
    # OVERALL CONFIDENCE
    # ========================================================

    confidence = float(
        alternative.get(
            "confidence",
            0.0
        )
        or 0.0
    )

    # ========================================================
    # WORDS
    # ========================================================

    words = alternative.get(
        "words",
        []
    )

    # ========================================================
    # WORD CONFIDENCE INFORMATION
    # ========================================================

    word_details = []

    for word in words:

        word_text = (
            word.get(
                "punctuated_word"
            )
            or word.get(
                "word"
            )
            or ""
        )

        word_confidence = float(
            word.get(
                "confidence",
                0.0
            )
            or 0.0
        )

        word_details.append(
            {
                "word": word_text,
                "confidence": word_confidence,
            }
        )

    # ========================================================
    # VALIDATE FINAL OUTPUT
    # ========================================================

    final_transcript = validate_output(
        raw_transcript
    )

    roman_valid = bool(
        final_transcript
    ) if raw_transcript else True

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "text": final_transcript,
        "raw_text": raw_transcript,
        "confidence": confidence,
        "words": word_details,
        "roman_valid": roman_valid,
    }


# ============================================================
# SESSION STATE
# ============================================================

if "transcription" not in st.session_state:

    st.session_state.transcription = ""


if "raw_transcription" not in st.session_state:

    st.session_state.raw_transcription = ""


if "confidence" not in st.session_state:

    st.session_state.confidence = 0.0


if "word_details" not in st.session_state:

    st.session_state.word_details = []


if "roman_valid" not in st.session_state:

    st.session_state.roman_valid = True


if "debug_mode" not in st.session_state:

    st.session_state.debug_mode = False


# ============================================================
# UI
# ============================================================

st.title(
    "🎙️ Human-Like Roman Urdu Speech Agent"
)

st.caption(
    "Deepgram Nova-3 • Urdu/English speech • "
    "Background-noise tolerant • No LLM"
)

# ============================================================
# DEBUG
# ============================================================

debug_mode = st.checkbox(
    "🐞 Show technical debug information",
    value=False
)

st.session_state.debug_mode = debug_mode


# ============================================================
# INSTRUCTIONS
# ============================================================

st.subheader(
    "🎤 Speak naturally"
)

st.write(
    "Normal awaaz mein baat karein. "
    "Background noise ho to bhi agent primary speech "
    "ko recognize karne ki koshish karega."
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
    key="deepgram_human_speech_agent",
)


# ============================================================
# AUDIO PROCESSING
# ============================================================

if (
    audio_output
    and "bytes" in audio_output
):

    audio_bytes = audio_output["bytes"]

    if audio_bytes:

        # ====================================================
        # STEP 1
        # ====================================================

        with st.spinner(
            "🎧 Preparing audio..."
        ):

            processed_audio = process_audio(
                audio_bytes
            )

        if processed_audio is None:

            st.warning(
                "⚠️ Audio bohat short, empty, "
                "ya invalid hai. Dobara try karein."
            )

        else:

            # =================================================
            # STEP 2
            # =================================================

            with st.spinner(
                "🧠 Deepgram listening..."
            ):

                try:

                    result = (
                        transcribe_with_deepgram(
                            processed_audio
                        )
                    )

                    text = result[
                        "text"
                    ]

                    raw_text = result.get(
                        "raw_text",
                        ""
                    )

                    confidence = result[
                        "confidence"
                    ]

                    word_details = result[
                        "words"
                    ]

                    roman_valid = result[
                        "roman_valid"
                    ]

                    # =========================================
                    # SAVE
                    # =========================================

                    st.session_state.transcription = (
                        text
                    )

                    st.session_state.raw_transcription = (
                        raw_text
                    )

                    st.session_state.confidence = (
                        confidence
                    )

                    st.session_state.word_details = (
                        word_details
                    )

                    st.session_state.roman_valid = (
                        roman_valid
                    )

                    # =========================================
                    # RESULT
                    # =========================================

                    if not raw_text:

                        st.warning(
                            "⚠️ Deepgram ko clear speech nahi mili."
                        )

                    elif not roman_valid:

                        st.warning(
                            "⚠️ Deepgram ne Urdu/Arabic "
                            "script return ki. "
                            "Deepgram-only mode mein "
                            "automatic transliteration nahi ki ja rahi."
                        )

                    elif text:

                        st.success(
                            "✅ Speech recognized!"
                        )

                    else:

                        st.warning(
                            "⚠️ Valid Roman Urdu output "
                            "nahi mila."
                        )

                except Exception as e:

                    st.error(
                        f"❌ Deepgram Error: {e}"
                    )

                    if debug_mode:

                        st.exception(e)


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
                🎧 Primary Speaker
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
        f"Overall confidence: "
        f"{st.session_state.confidence:.2f}"
    )

else:

    st.info(
        "Aapki speech ka Roman Urdu output yahan show hoga."
    )


# ============================================================
# DEBUG INFORMATION
# ============================================================

if debug_mode:

    st.divider()

    st.subheader(
        "🔍 Debug Information"
    )

    if st.session_state.raw_transcription:

        st.write(
            "**Raw Deepgram transcript:**"
        )

        st.code(
            st.session_state.raw_transcription
        )

    if st.session_state.word_details:

        st.write(
            "**Word-level confidence:**"
        )

        for item in st.session_state.word_details:

            word = item["word"]

            conf = item["confidence"]

            if conf < 0.50:

                st.warning(
                    f"{word} → {conf:.2f} ⚠️"
                )

            else:

                st.write(
                    f"{word} → {conf:.2f}"
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

    st.session_state.raw_transcription = ""

    st.session_state.confidence = 0.0

    st.session_state.word_details = []

    st.session_state.roman_valid = True

    st.rerun()
