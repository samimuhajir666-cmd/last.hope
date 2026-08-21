import io
import re
import torch
import numpy as np
import scipy.io.wavfile as wav
from groq import Groq

# ============================
# 1. ADAPTIVE NOISE & VAD FILTER
# ============================
class HumanAttentionFilter:
    def __init__(self):
        # Load Silero VAD model for natural voice detection
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        (self.get_speech_timestamps, _, _, _, _) = self.utils

    def extract_dominant_speech(self, audio_bytes, sample_rate=16000):
        """
        Adapts dynamically to changing noise floor and isolates 
        the dominant speaker's voice activity.
        """
        # Convert bytes to float32 tensor normalized between -1.0 and 1.0
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        wav_tensor = torch.from_numpy(audio_float32)

        # Dynamic VAD timestamps detection
        speech_timestamps = self.get_speech_timestamps(
            wav_tensor, 
            self.model, 
            sampling_rate=sample_rate,
            threshold=0.4,            # Human-like sensitivity threshold
            min_speech_duration_ms=250, # Ignore random transient clicks/cough
            min_silence_duration_ms=300 # Handle natural speech pauses
        )

        if not speech_timestamps:
            return None

        # Stitch only relevant target speech chunks
        focused_speech = []
        for segment in speech_timestamps:
            start = segment['start']
            end = segment['end']
            focused_speech.append(audio_float32[start:end])

        if not focused_speech:
            return None

        concatenated_audio = np.concatenate(focused_speech)
        
        # Convert back to int16 PCM bytes for STT model
        processed_pcm = (concatenated_audio * 32767.0).astype(np.int16).tobytes()
        return processed_pcm


# ============================
# 2. CONTEXTUAL LLM CLEANER (HALLUCINATION & NOISE GUARD)
# ============================
def apply_human_listener_guardrails(raw_transcript, groq_client):
    """
    Acts as the cognitive human brain:
    1. Ignores background chatter hallucination.
    2. Reconstructs masked words using surround context.
    3. Replaces completely distorted sound with [inaudible].
    """
    if not raw_transcript or not raw_transcript.strip():
        return ""

    system_prompt = (
        "You are a Human Listener Attention Module for a Speech-to-Text system.\n"
        "Your task is to refine the raw audio transcription to mimic human focus.\n\n"
        "RULES:\n"
        "1. PRESERVE EXACT WORDS: Keep the primary speaker's actual words, filler words ('um', 'uh'), "
        "   pauses, and natural repetitions.\n"
        "2. IGNORE BACKGROUND NOISE/CHATTER: If the raw text contains stray background noise "
        "   or phantom words created by background sounds, strip them out.\n"
        "3. NO HALLUCINATION: Do NOT invent new facts or complete sentences that were not spoken.\n"
        "4. UNCLEAR AUDIO: If a word/phrase is completely garbled, masked by noise, or unintelligible, "
        "   mark it strictly as [inaudible].\n"
        "5. PUNCTUATION: Maintain natural punctuation and sentence boundaries.\n"
        "OUTPUT ONLY the final cleaned transcript without explanations or commentary."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Raw Transcript: {raw_transcript}"}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return raw_transcript

# ============================
# 3. PIPELINE EXECUTION
# ============================
def process_human_like_audio(audio_bytes, groq_client, stt_engine_callback):
    """
    End-to-End Pipeline
    """
    attention_filter = HumanAttentionFilter()
    
    # Step 1: Suppress noise & isolate dominant voice
    filtered_audio_bytes = attention_filter.extract_dominant_speech(audio_bytes)
    
    if filtered_audio_bytes is None:
        return "[No speech detected]"

    # Step 2: Speech-to-Text Transcription (Whisper / Deepgram / Groq Speech)
    raw_transcript = stt_engine_callback(filtered_audio_bytes)

    # Step 3: Cognitive Post-Processing (Contextual Inference & [inaudible] tagging)
    final_transcript = apply_human_listener_guardrails(raw_transcript, groq_client)

    return final_transcript
