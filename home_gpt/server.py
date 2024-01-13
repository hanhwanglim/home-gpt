import os

from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel
import tempfile

app = FastAPI()
model_size = "tiny.en"
model = WhisperModel(model_size, device="cpu", compute_type="int8")


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f"_{file.filename}"
    ) as temp_file:
        # Write content to the temporary file
        content = await file.read()  # async read
        temp_file.write(content)  # write to temp file
        file_location = temp_file.name  # Store the temp file name to use later

    try:
        response = await transcribe(file_location)
        return response
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup: delete the temporary file after processing
        os.remove(file_location)


async def transcribe(file_location):
    segments, info = model.transcribe(file_location, beam_size=5)
    response = {
        "Detected language": info.language,
        "Language probability": info.language_probability,
        "Transcription": [
            {"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments
        ],
    }
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", port=5000, reload=True, access_log=False)
