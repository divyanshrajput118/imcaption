from fastapi import FastAPI, UploadFile, File, HTTPException
from imgCaption.pipeline.prediction import PredictionPipeline
from pathlib import Path
import tempfile
import shutil
from fastapi.middleware.cors import CORSMiddleware
from imgCaption import logger
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "*"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger.info("Initializing PredictionPipeline...")
pipeline = PredictionPipeline()
logger.info("PredictionPipeline initialized.")

@app.get("/")
def serve_home():
    return FileResponse("frontend/index.html")

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = Path(temp_file.name)

    try:
        caption = pipeline.predict(temp_path)
        return {"caption": caption}
    except Exception as exc:
        logger.exception(exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")
    finally:
        if temp_path.exists():
            temp_path.unlink()
