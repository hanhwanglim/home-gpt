import logging

import speech_recognition as sr
from speech_recognition import AudioData

logger = logging.getLogger(__name__)


def record_audio() -> AudioData:
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        logger.info("Adjusting for ambient noise.")
        recognizer.adjust_for_ambient_noise(source)

        logger.info("Microphone is active.")
        audio_data = recognizer.listen(source)

    return audio_data
