import logging

import speech_recognition as sr

logger = logging.getLogger(__name__)


def record_audio():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        logger.info("Adjusting for ambient noise.")
        recognizer.adjust_for_ambient_noise(source)

        logger.info("Microphone is active.")
        audio_data = recognizer.listen(source)

    with open("recording.wav", "wb") as f:
        f.write(audio_data.get_wav_data())
        logger.info("Saving recording clip.")
