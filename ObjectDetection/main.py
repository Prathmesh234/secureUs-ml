from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from securityNew import ObjectDetection

UPLOAD_DIR = Path() / 'uploads'

if not UPLOAD_DIR.exists():
    UPLOAD_DIR.mkdir()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
object_detection = ObjectDetection()

@app.post('/uploadfile/')
async def create_upload_file(file_upload: UploadFile):
    try:
        data = await file_upload.read()
        save_to = UPLOAD_DIR / file_upload.filename
        with open(save_to, 'wb') as f:
            f.write(data)
        print(f"File saved to: {save_to}")
        object_detection(file_upload.filename)
        return {"filename": file_upload.filename}
    except Exception as e:
        print(f"Error saving file: {e}")
        return {"error": "Failed to save file"}
