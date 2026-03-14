import asyncio
import io
import edge_tts
import speech_recognition as sr


async def _synthesize(text: str, voice: str) -> bytes:
    """Generate audio bytes from text using edge-tts."""
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
        return audio_buffer.getvalue()
    except Exception as e:
        raise RuntimeError(f"TTS synthesis failed: {e}")


def text_to_speech(text: str, voice: str = "en-US-JennyNeural") -> bytes:
    """Convert text to audio bytes. Returns bytes to be played via st.audio()."""
    try:
        return asyncio.run(_synthesize(text, voice))
    except Exception as e:
        raise RuntimeError(f"TTS failed: {e}")


def speech_to_text(audio_bytes: bytes) -> str:
    """Convert recorded audio bytes to text using SpeechRecognition.
    Tries pydub (needs ffmpeg) first; falls back to raw WAV if unavailable.
    """
    try:
        recognizer = sr.Recognizer()

        # Try pydub conversion (works locally with ffmpeg)
        try:
            from pydub import AudioSegment
            webm_buffer = io.BytesIO(audio_bytes)
            audio_segment = AudioSegment.from_file(webm_buffer, format="webm")
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
        except Exception:
            # ffmpeg not available (Streamlit Cloud) — treat bytes as raw WAV
            wav_buffer = io.BytesIO(audio_bytes)

        with sr.AudioFile(wav_buffer) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)

    except sr.UnknownValueError:
        return ""
    except Exception as e:
        raise RuntimeError(f"Speech recognition failed: {e}")
