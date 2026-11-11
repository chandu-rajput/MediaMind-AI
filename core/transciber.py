
import whisper
import os
from dotenv import load_dotenv

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None


def load_model():
    """Load Whisper model only once."""

    global _model

    if _model is None:
        print(f"Loading Whisper {WHISPER_MODEL} model...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Model loaded successfully.")

    return _model




def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    """Transcribe or translate a single audio chunk."""

    model = load_model()

    task = "translate" if translate else "transcribe"

    result = model.transcribe(chunk_path, task=task)

    return result["text"].strip()




def transcribe_all(chunks: list, translate: bool = False) -> str:
    """Transcribe all audio chunks and combine the results."""

    if not chunks:
        raise ValueError("No audio chunks provided.")

    full_transcript = []

    for i, chunk in enumerate(chunks, start=1):

        if not os.path.exists(chunk):
            raise FileNotFoundError(f"Audio chunk not found: {chunk}")

        print(f"Transcribing chunk {i}/{len(chunks)}...")

        text = transcribe_chunk(chunk, translate=translate)

        full_transcript.append(text)

    print("Transcription completed.")

    return " ".join(full_transcript).strip()
