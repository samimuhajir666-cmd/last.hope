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


# ============================================================
# 🖥️ STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Human-Like Roman Urdu Voice Agent",
    page_icon="🧠",
    layout="centered",
)

load_dotenv()


# ============================================================
# 🔑 API KEYS
# ============================================================

DEEPGRAM_API_KEY = (
    os.getenv("DEEPGRAM_API_KEY")
    or st.secrets.get("DEEPGRAM_API_KEY", None)
)

GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY")
    or st.secrets.get("GROQ_API_KEY", None)
)

if not DEEPGRAM_API_KEY or not GROQ_API_KEY:
    st.error(
        "❌ DEEPGRAM_API_KEY ya GROQ_API_KEY missing hain. "
        "Apni .env file ya Streamlit Secrets check karein."
    )
    st.stop()


# ============================================================
# ⚙️ CONFIGURATION
# ============================================================

DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"

# Deepgram language.
# Keep Urdu recognition at the STT layer.
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_LANGUAGE = "ur"

DEEPGRAM_TIMEOUT = 60

GROQ_MODEL = "llama-3.3-70b-versatile"

# Minimum overall confidence before we consider the result
# completely unreliable.
MIN_CONFIDENCE = 0.30


# ============================================================
# 🧠 GROQ CLIENT
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# 🔤 ROMAN URDU CLEANING
# ============================================================

def clean_roman_urdu(text):
    """
    Cleans the model output while preserving:
    - Roman Urdu
    - English
    - numbers
    - punctuation
    """

    if not text:
        return ""

    text = text.strip()

    # Remove accidental Markdown/code formatting.
    text = text.replace("```", "")

    # Remove common model prefixes.
    prefixes = [
        "Final transcription:",
        "Final Transcript:",
        "Transcription:",
        "Roman Urdu:",
        "Output:",
    ]

    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Remove accidental surrounding quotes.
    text = text.strip("\"'")

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 🧠 HUMAN-LIKE CONTEXT CORRECTION
# ============================================================

def human_like_context_correction(raw_transcript, confidence):
    """
    Converts/corrects Deepgram output into natural Roman Urdu.

    VERY IMPORTANT:
    This function is NOT allowed to generate a response.

    It can:
        - convert Urdu script to Roman Urdu
        - fix obvious individual misrecognized words
        - fix very short unclear phrases
        - preserve English words
        - mark genuinely unintelligible audio

    It CANNOT:
        - invent a sentence
        - complete the speaker's thought
        - summarize
        - paraphrase
        - answer the speaker
        - add information
    """

    if not raw_transcript or not raw_transcript.strip():
        return "[inaudible]", 0.0

    # If the entire recognition is extremely unreliable,
    # do not ask the LLM to hallucinate a reconstruction.
    if confidence < MIN_CONFIDENCE:
        return "[inaudible]", confidence

    system_prompt = """
You are the final human-like transcription correction layer.

You receive a RAW speech-to-text transcript.

The speaker may be speaking:
- Roman Urdu
- Urdu
- English
- Urdu + English mixed together

Your ONLY job is to produce an accurate Roman Urdu transcription.

You are NOT a chatbot.
You are NOT an assistant answering the speaker.
You are NOT a summarizer.
You are NOT allowed to create new content.

========================
STRICT RULES
========================

RULE 1 — PRESERVE SPOKEN CONTENT

Keep the speaker's actual words and meaning.

Do not add information.

Do not remove meaningful spoken information.

Do not rewrite the speaker's statement into your own sentence.

--------------------------------

RULE 2 — ROMAN URDU OUTPUT

If the raw transcript is in Urdu/Arabic script, convert it into
natural readable Roman Urdu.

Example:

مجھے کل بازار جانا ہے

becomes:

mujhe kal bazaar jana hai

Do NOT simply delete Urdu characters.

--------------------------------

RULE 3 — MIXED LANGUAGE

The speaker may naturally mix English and Urdu.

Preserve English words when appropriate.

Example:

"mujhe meeting ke liye ready hona hai"

should remain:

"mujhe meeting ke liye ready hona hai"

Do not unnecessarily translate English words.

--------------------------------

RULE 4 — SMALL WORD CORRECTIONS ARE ALLOWED

If ONE word is clearly misrecognized and the surrounding context
strongly identifies the intended word, correct that word.

Example:

"mujhe hospitl jana hai"

can become:

"mujhe hospital jana hai"

This is allowed.

--------------------------------

RULE 5 — SHORT PHRASE CORRECTION

A very short phrase may be corrected if the intended phrase is
strongly supported.

But do NOT reconstruct an entire sentence.

--------------------------------

RULE 6 — NEVER INVENT A SENTENCE

This is the most important rule.

If the speaker says:

"main kal market ja raha hoon aur phir..."

and the rest is unclear,

DO NOT write:

"main kal market ja raha hoon aur phir ghar wapas aa jaunga."

That is hallucination.

Instead write:

"main kal market ja raha hoon aur phir [inaudible]"

--------------------------------

RULE 7 — NEVER COMPLETE THOUGHTS

Never guess what the speaker was going to say.

Context may be used to correct a WORD.

Context may NOT be used to create an entire missing sentence.

--------------------------------

RULE 8 — UNCLEAR AUDIO

If a word cannot reasonably be determined:

[inaudible]

If only a small part is unclear, keep the rest of the sentence.

Example:

"main kal [inaudible] jaunga"

--------------------------------

RULE 9 — NO SUMMARY

Do not summarize.

Input:
"main kal office gaya tha aur wahan boss ke sath meeting hui"

Output must preserve the statement.

Never output:
"speaker went to the office."

--------------------------------

RULE 10 — NO ANSWER

If the speaker asks:

"kal mausam kaisa hoga?"

DO NOT answer the question.

Output only:

"kal mausam kaisa hoga?"

--------------------------------

RULE 11 — NO EXTRA WORDS

Do not say:

"Here is the transcription."

Do not say:

"Corrected text:"

Do not explain anything.

Return ONLY the transcription.

--------------------------------

RULE 12 — LENGTH PROTECTION

Do not make the output substantially longer than the raw transcript.

Small corrections are allowed.

Large additions are forbidden.

--------------------------------

FINAL REQUIREMENT:

Output ONLY natural Roman Urdu transcription.
"""


    user_prompt = f"""
RAW TRANSCRIPT:
{raw_transcript}

RECOGNITION CONFIDENCE:
{confidence:.2f}

Convert/correct the transcript according to all rules above.

Remember:

CORRECT A WORD IF CLEAR.
DO NOT INVENT A SENTENCE.
DO NOT COMPLETE MISSING SPEECH.
DO NOT ANSWER THE SPEAKER.
DO NOT SUMMARIZE.

Return ONLY the final Roman Urdu transcription.
"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.0,
            max_tokens=800,
        )

        corrected_text = response.choices[0].message.content or ""

        corrected_text = clean_roman_urdu(corrected_text)

        if not corrected_text:
            return raw_transcript, confidence

        # ====================================================
        # 🛡️ HALLUCINATION/LENGTH SAFETY CHECK
        # ====================================================

        raw_words = raw_transcript.split()
        corrected_words = corrected_text.split()

        raw_count = len(raw_words)
        corrected_count = len(corrected_words)

        # If Groq suddenly produces a much longer sentence,
        # reject it and use the original transcript.
        #
        # Example:
        #
        # Raw:  "main kal market gaya"
        #
        # Bad:
        # "main kal market gaya tha aur wahan se grocery li
        #  aur phir ghar wapas aa gaya"
        #
        # We reject this kind of expansion.

        if raw_count >= 5:
            maximum_allowed = max(
                raw_count + 8,
                int(raw_count * 1.60)
            )

            if corrected_count > maximum_allowed:
                return raw_transcript, confidence

        return corrected_text, confidence

    except Exception:
        # If Groq fails, NEVER destroy the original transcript.
        return raw_transcript, confidence


# ============================================================
# 🎙️ DEEPGRAM STT
# ============================================================

def transcribe_audio_stream(processed_bytes, debug=False):

    params = {
        "model": DEEPGRAM_MODEL,
        "language": DEEPGRAM_LANGUAGE,

        # Formatting
        "smart_format": "true",
        "punctuate": "true",
        "numerals": "true",

        # Improve recognition behavior.
        "paragraphs": "false",
        "utterances": "true",

        # Return word information.
        "words": "true",
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

        raise RuntimeError(
            f"Deepgram connection failed: {e}"
        ) from e

    if response.status_code != 200:

        detail = response.text[:1500]

        raise RuntimeError(
            f"Deepgram API error "
            f"({response.status_code}): {detail}"
        )

    try:
        data = response.json()

    except Exception as e:

        raise RuntimeError(
            "Invalid JSON response from Deepgram."
        ) from e

    results = data.get("results", {})

    channels = results.get("channels", [])

    if not channels:
        return {
            "text": "[inaudible]",
            "confidence": 0.0,
        }

    alternatives = channels[0].get(
        "alternatives",
        []
    )

    if not alternatives:
        return {
            "text": "[inaudible]",
            "confidence": 0.0,
        }

    alternative = alternatives[0]

    transcript = (
        alternative.get("transcript") or ""
    ).strip()

    confidence = float(
        alternative.get("confidence", 0.0) or 0.0
    )

    if not transcript:

        return {
            "text": "[inaudible]",
            "confidence": 0.0,
        }

    # ========================================================
    # 🧠 HUMAN-LIKE CORRECTION
    # ========================================================

    final_text, final_confidence = (
        human_like_context_correction(
            transcript,
            confidence
        )
    )

    return {
        "text": final_text,
        "confidence": final_confidence,
        "raw_text": transcript,
    }


# ============================================================
# 🎚️ AUDIO PRE-PROCESSING
# ============================================================

def process_audio_buffer(audio_bytes):

    try:

        audio_file = io.BytesIO(audio_bytes)

        sample_rate, audio_data = wav.read(
            audio_file
        )

        if (
            sample_rate <= 0
            or len(audio_data) == 0
        ):
            return None

        # ====================================================
        # MONO
        # ====================================================

        if len(audio_data.shape) > 1:

            audio_data = np.mean(
                audio_data,
                axis=1
            )

        audio_data = audio_data.astype(
            np.float64
        )

        duration = (
            len(audio_data)
            / float(sample_rate)
        )

        # Ignore accidental clicks.
        if duration < 0.25:
            return None

        # ====================================================
        # IMPORTANT:
        #
        # DO NOT aggressively remove quiet audio.
        #
        # The old code:
        #
        # noise_floor = np.percentile(...)
        # audio_data[...] = 0
        #
        # could destroy quiet speech.
        # ====================================================

        # Remove DC offset.
        audio_data = (
            audio_data
            - np.mean(audio_data)
        )

        # ====================================================
        # SAFE NORMALIZATION
        # ====================================================

        max_val = np.max(
            np.abs(audio_data)
        )

        if max_val <= 0:
            return None

        # Normalize without clipping speech dynamically.
        target_peak = 0.90

        audio_data = (
            audio_data / max_val
        ) * (
            32767.0 * target_peak
        )

        processed_audio = np.clip(
            audio_data,
            -32768,
            32767
        ).astype(np.int16)

        # ====================================================
        # WRITE WAV
        # ====================================================

        output_buffer = io.BytesIO()

        wav.write(
            output_buffer,
            sample_rate,
            processed_audio
        )

        output_buffer.seek(0)

        return output_buffer.read()

    except Exception:
        return None


# ============================================================
# 🧠 SESSION STATE
# ============================================================

if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = ""

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

if "last_raw_transcript" not in st.session_state:
    st.session_state.last_raw_transcript = ""


# ============================================================
# 🖥️ UI
# ============================================================

st.title(
    "🧠 Human-Like Roman Urdu Voice Agent"
)

st.caption(
    "Focuses on speech, ignores irrelevant noise, "
    "corrects small recognition mistakes, and "
    "does not invent complete sentences."
)

debug_mode = st.checkbox(
    "🐞 Show Technical Debug Errors",
    value=False
)


# ============================================================
# 🎙️ MICROPHONE
# ============================================================

st.subheader("🎙️ Live Speech Input")

st.write(
    "Mic start karein, naturally baat karein, "
    "phir Stop & Process press karein."
)

audio_output = mic_recorder(
    start_prompt="🎙️ Start Listening",
    stop_prompt="🛑 Stop & Process",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="human_like_agent_mic",
)


# ============================================================
# ⚡ EXECUTION PIPELINE
# ============================================================

if audio_output and "bytes" in audio_output:

    audio_bytes = audio_output["bytes"]

    if len(audio_bytes) > 0:

        # ====================================================
        # STEP 1 — AUDIO PROCESSING
        # ====================================================

        with st.spinner(
            "🎧 Processing audio..."
        ):

            processed_bytes = (
                process_audio_buffer(
                    audio_bytes
                )
            )

        if processed_bytes is None:

            st.warning(
                "⚠️ Recording bohat choti thi "
                "ya audio empty thi. Dobara try karein."
            )

        else:

            # =================================================
            # STEP 2 — DEEPGRAM
            # =================================================

            with st.spinner(
                "🎙️ Listening to your speech..."
            ):

                try:

                    result = (
                        transcribe_audio_stream(
                            processed_bytes,
                            debug=debug_mode
                        )
                    )

                    transcription = result["text"]

                    confidence = result[
                        "confidence"
                    ]

                    raw_transcript = result.get(
                        "raw_text",
                        ""
                    )

                    # =========================================
                    # SAVE
                    # =========================================

                    st.session_state.agent_memory = (
                        transcription
                    )

                    st.session_state.last_confidence = (
                        confidence
                    )

                    st.session_state.last_raw_transcript = (
                        raw_transcript
                    )

                    st.success(
                        "✅ Speech processed successfully!"
                    )

                except Exception as e:

                    st.error(
                        f"❌ Pipeline Error: {e}"
                    )

                    if debug_mode:
                        st.exception(e)


# ============================================================
# 📝 FINAL OUTPUT
# ============================================================

st.divider()

st.subheader(
    "📝 Roman Urdu Transcription"
)

if st.session_state.agent_memory:

    safe_text = html.escape(
        st.session_state.agent_memory
    )

    st.markdown(
        f"""
        <div style="
            padding: 18px;
            border-radius: 12px;
            background-color: #1e1e2e;
            border: 1px solid #45475a;
            margin-top: 10px;
        ">

            <div style="
                font-weight: bold;
                color: #89b4fa;
                margin-bottom: 8px;
                font-size: 1.05em;
            ">
                🎧 Focused Speaker Input
            </div>

            <div style="
                font-size: 1.25em;
                color: #cdd6f4;
                font-weight: 500;
                line-height: 1.6;
            ">
                {safe_text}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if (
        st.session_state.last_confidence
        is not None
    ):

        st.caption(
            "Recognition Confidence: "
            f"{st.session_state.last_confidence:.2f}"
        )

    # ========================================================
    # DEBUG RAW TRANSCRIPT
    # ========================================================

    if (
        debug_mode
        and st.session_state.last_raw_transcript
    ):

        with st.expander(
            "🔍 Raw Deepgram Transcript"
        ):

            st.code(
                st.session_state.last_raw_transcript
            )

else:

    st.info(
        "Aapki Roman Urdu speech ka "
        "processed text yahan show hoga."
    )


# ============================================================
# 🛠️ SESSION CONTROLS
# ============================================================

st.divider()

col1, col2 = st.columns(2)


with col1:

    if st.button(
        "🔒 Lock Input",
        use_container_width=True
    ):

        if st.session_state.agent_memory:

            st.success(
                "Input locked and ready "
                "for your agent workflow!"
            )

        else:

            st.warning(
                "Koi text maujood nahi hai."
            )


with col2:

    if st.button(
        "🗑️ Reset Memory",
        use_container_width=True
    ):

        st.session_state.agent_memory = ""
        st.session_state.last_confidence = None
        st.session_state.last_raw_transcript = ""

        st.rerun()
