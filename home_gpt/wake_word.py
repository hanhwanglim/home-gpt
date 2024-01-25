import logging
import tempfile
from pathlib import Path
from time import sleep
from typing import NoReturn

import numpy as np
import pyaudio
import requests
from gtts import gTTS
from openwakeword import Model
from playsound import playsound

from home_gpt.record import record_audio
from home_gpt.whisper import Transcription

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512
WAKE_WORD_MODEL_PATH = Path(__file__).parent.parent / "models/hey_gpt.tflite"
THRESHOLD = 0.5


audio = pyaudio.PyAudio()
microphone = audio.open(
    rate=16000,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
)
wake_word_models = Model(
    wakeword_models=[str(WAKE_WORD_MODEL_PATH)],
    inference_framework=WAKE_WORD_MODEL_PATH.suffix.lstrip("."),
)


def listen() -> NoReturn:
    while True:
        logger.info("Listening for wake word")
        audio_stream = np.frombuffer(microphone.read(CHUNK_SIZE), dtype=np.int16)
        wake_word_models.predict(audio_stream)

        for prediction_buffer in wake_word_models.prediction_buffer.values():
            if prediction_buffer[-1] < THRESHOLD:
                continue

            logger.info("Wake word detected")
            audio_data = record_audio()
            logger.info("Recording finished")

            response = requests.post(
                "http://localhost:5000/upload",
                files={"audio_file": audio_data.get_wav_data()},
            )
            transcription = Transcription(**response.json())
            text = "".join([segment.text for segment in transcription.segments])
            logger.info(text)

            with tempfile.NamedTemporaryFile() as f:
                tts = gTTS(text=text, lang="en")
                tts.save(f.name)
                playsound(f.name)

            wake_word_models.reset()
            sleep(1)


if __name__ == "__main__":
    listen()
