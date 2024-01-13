import logging
from pathlib import Path

import numpy as np
import pyaudio
from openwakeword import Model

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

while True:
    audio_stream = np.frombuffer(microphone.read(CHUNK_SIZE), dtype=np.int16)
    wake_word_models.predict(audio_stream)

    for prediction_buffer in wake_word_models.prediction_buffer.values():
        if prediction_buffer[-1] > THRESHOLD:
            logger.info("Wake word detected")
