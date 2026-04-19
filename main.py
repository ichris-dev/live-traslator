from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

class Message(BaseModel):
    text: str

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/message")
async def receive_message(data: Message):
    print(f"You sent: {data.text}")
    return {"message": f"You sent: {data.text}"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename  # type: ignore

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    print(f"Received audio: {file.filename}")

    return {
        "message": "Audio received successfully",
        "filename": file.filename,
        "saved_to": str(file_path),
    }