# Home GPT

Home GPT is a personal home assistant similar to Siri. It listens to a wake word in the background and records your input from the microphone and sends to OpenAI to process.

## Motivation

Siri sucks. OpenAI is smart. Replace Siri with OpenAI :skull:.

## Roadmap

- [x] Capture audio from device microphone
- [x] Integration with OpenWakeWord
- [x] Transcribe audio using Whisper
- [x] OpenAI integration
- [x] Play response to speaker using google's TTS
- [ ] Containerize with docker

## Installation

1. Install poetry
2. Install the dependencies

  ```bash
  git clone https://github.com/hanhwanglim/home-gpt
  poetry shell
  poetry install
  ```

3. Start the application

```bash
python record.py  # Audio service
python server.py  # Server service
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
