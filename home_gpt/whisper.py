from typing import BinaryIO

from faster_whisper import WhisperModel
from pydantic import BaseModel

MODEL_SIZE = "tiny.en"

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


class Segment(BaseModel):
    start: float
    end: float
    text: str


class Transcription(BaseModel):
    detected_language: str
    language_probability: float
    duration: float
    segments: list[Segment]


async def transcribe(file: BinaryIO) -> Transcription:
    segments, info = model.transcribe(file, beam_size=5)

    return Transcription(
        detected_language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        segments=[
            Segment(start=segment.start, end=segment.end, text=segment.text)
            for segment in segments
        ],
    )
