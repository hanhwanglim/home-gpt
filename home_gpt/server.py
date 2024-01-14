import tempfile
from typing import Any

from fastapi import FastAPI, UploadFile

from home_gpt.whisper import transcribe

app = FastAPI()


@app.post("/upload/")
async def upload(file: UploadFile) -> dict[str, Any]:
    # By default, FastAPI loads files into memory until it exceeds a certain size.
    # This is not compatible with the av library and throws an error.
    # We need to save it into a tempfile and load it into whisper instead.
    # https://github.com/SYSTRAN/faster-whisper/issues/74
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(await file.read())
        transcription = await transcribe(temp_file.name)
    return transcription.model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", port=5000, reload=True, access_log=False)
