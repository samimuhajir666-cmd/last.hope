import io
import os
import re
import time
import numpy as np
import scipy.signal as signal
import noisereduce as nr
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

load_dotenv()

# ==============================================================================
# 🔑 API KEY
# ==============================================================================
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    try:
        if "DEEPGRAM_API_KEY" in st.secrets:
            DEEPGRAM_API_KEY = st.secrets["DEEPGRAM_API_KEY"]
    except Exception:
        pass

if not DEEPGRAM_API_KEY:
    st.error("❌ DEEPGRAM_API_KEY missing! Please check your .env or Streamlit Secrets.")
    st.stop()

try:
    deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
except Exception as init_error:
    st.error(f"❌ Failed to initialize Deepgram: {init_error}")
    st.stop()

SYSTEM_PROMPT = "Roman Urdu, Arabic, and English mixed linguistic pipeline."

def force_roman_script(text):
    """Convert Urdu/Arabic script to Romanized English."""
    if not text:
        return text
    if bool(re.search(r'[^\x00-\x7F]', text)):
        return unidecode(text)
    return text

# ==============================================================================
# 🎚️ AUDIO PROCESSING
# ==============================================================================
SPEECH_LOW_HZ = 85.0
SPEECH_HIGH_HZ = 3500.0

def apply_hardware_acoustic_filters(raw_bytes, sensitivity=0.7):
    """Apply noise reduction and anti-shouting compression."""
    import scipy.io.wavfile as wav
    
    sample_rate, data = wav.read(io.BytesIO(raw_bytes))
    
    if data.dtype == np.int16:
        audio_float = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio_float = data.astype(np.float32) / 2147483648.0
    else:
        audio_float = data.astype(np.float32)
        
    if len(audio_float.shape) > 1:
        audio_float = np.mean(audio_float, axis=1)
        
    # Anti-shouting limiter
    max_peak = np.max(np.abs(audio_float))
    if max_peak > 0.75:
        audio_float = np.tanh(audio_float / max_peak) * 0.75
        
    # Bandpass filter
    nyquist = 0.5 * sample_rate
    low_cut = SPEECH_LOW_HZ / nyquist
    high_cut = min(SPEECH_HIGH_HZ / nyquist, 0.99)
    b, a = signal.butter(4, [low_cut, high_cut], btype="band")
    filtered_signal = signal.filtfilt(b, a, audio_float)
    
    # Noise reduction
    reduced_noise = nr.reduce_noise(
        y=filtered_signal, 
        sr=sample_rate, 
        prop_decrease=0.85, 
        n_fft=1024
    )
    
    clean_pcm = np.clip(reduced_noise * 32768.0, -32768, 32767).astype(np.int16)
    
    output_io = io.BytesIO()
    wav.write(output_io, sample_rate, clean_pcm)
    return output_io.getvalue()

# ==============================================================================
# 🎙️ DEEPGRAM TRANSCRIBE (FIXED — v3.7.0 compatible)
# ==============================================================================
def execute_agent_transcription(processed_wav_bytes):
    """Transcribe using Deepgram Nova-3 with multi-language support."""
    try:
        # 🔥 FIX: Proper FileSource for v3.7.0
        payload = {
            "buffer": processed_wav_bytes,
            "mimetype": "audio/wav",
        }
        
        options = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
            punctuate=True,
            utterances=True,
            language="multi",
        )
        
        response = deepgram_client.listen.prerecorded.v("1").transcribe_file(
            FileSource(**payload), options
        )
        
        # 🔥 FIX: Safe response parsing for v3.7.0
        if response and hasattr(response, 'results'):
            channel = response.results.channels[0]
            alternative = channel.alternatives[0]
            raw_text = alternative.transcript
            confidence = alternative.confidence
            
            final_roman_text = force_roman_script(raw_text)
            return final_roman_text, confidence
        else:
            return "", 0.0
            
    except Exception as api_error:
        st.error(f"Deepgram API error: {api_error}")
        return "", 0.0

# ==============================================================================
# 🖥️ STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="Multi-Language STT Agent", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Language AI Speech-To-Text Agent")
st.caption("Production Build: Anti-Shouting Limiter + 85% Noise Filter")

st.sidebar.header("⚙️ Agent Controls")
noise_reduction_sensitivity = st.sidebar.slider("Noise Reduction Power", 0.1, 1.0, 0.7, step=0.05)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Directive:** `{SYSTEM_PROMPT}`")

uploaded_file = st.file_uploader("Upload Audio File (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])
st.write("✨ **-- OR SPEAK LIVE --** ✨")
recorded_audio = mic_recorder(
    start_prompt="🔴 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    key="live_agent_mic"
)

audio_payload_bytes = None

if uploaded_file is not None:
    audio_payload_bytes = uploaded_file.read()
elif recorded_audio is not None and 'bytes' in recorded_audio:
    audio_payload_bytes = recorded_audio['bytes']

if audio_payload_bytes is not None:
    st.info("📁 Audio received. Processing...")
    
    try:
        cleaned_bytes = apply_hardware_acoustic_filters(
            audio_payload_bytes, 
            sensitivity=noise_reduction_sensitivity
        )
        
        with st.spinner("🧠 Transcribing..."):
            transcript, confidence_score = execute_agent_transcription(cleaned_bytes)
            
        if transcript:
            st.success("✅ Complete!")
            
            col1, col2 = st.columns(2)
            col1.metric("🌐 Language Mode", "Multi (Urdu/English)")
            col2.metric("📊 Confidence", f"{confidence_score * 100:.2f}%")
            
            st.markdown("### 📝 Output:")
            st.code(transcript, language="text")
            
            st.download_button(
                label="📥 Download Text",
                data=transcript,
                file_name=f"transcript_{int(time.time())}.txt",
                mime="text/plain"
            )
        else:
            st.warning("⚠️ No speech detected. Try speaking clearly.")
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
