import tempfile
from typing import Any

from fastapi import FastAPI, UploadFile
from openai import OpenAI

from home_gpt.whisper import transcribe

app = FastAPI()
client = OpenAI()


@app.post("/upload/")
async def upload(audio_file: UploadFile) -> dict[str, Any]:
    # By default, FastAPI loads files into memory until it exceeds a certain size.
    # This is not compatible with the av library and throws an error.
    # We need to save it into a tempfile and load it into whisper instead.
    # https://github.com/SYSTRAN/faster-whisper/issues/74
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(await audio_file.read())
        transcription = await transcribe(temp_file.name)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful home assistant like Siri. Please answer all your home owner's questions. ",
            },
            {
                "role": "user",
                "content": " ".join(
                    [segment.text for segment in transcription.segments]
                ),
            },
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response.choices[0].message


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", port=5000, reload=True, access_log=False)
