import os
import sys
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory

# ── Allow importing from src/prediction even when running as `python app.py`
sys.path.insert(0, str(Path(__file__).parent / "src"))

from imgCaption.pipeline.prediction import PredictionPipeline
from imgCaption import logger

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Folder where uploaded images are saved temporarily
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Dataset images folder (used for sample gallery)
DATASET_IMAGES_FOLDER = Path("artifacts/data_ingestion/Images")

# Curated sample images shown in the UI gallery — ranked by per-image BLEU-4 score
# Scores: 0.699, 0.669, 0.540, 0.535, 0.502, 0.482
SAMPLE_IMAGES = [
    "1119015538_e8e796281e.jpg",   # BLEU 0.699 – dog running in grass
    "1472053993_bed67a3ba7.jpg",   # BLEU 0.669 – skier on snowy mountain
    "1425013325_bff69bc9da.jpg",   # BLEU 0.540 – two dogs playing in water
    "1213336750_2269b51397.jpg",   # BLEU 0.535 – man photographing on plaza
    "1413956047_c826f90c8b.jpg",   # BLEU 0.502 – three men on mountain peak
    "1248953128_24c9f8d924.jpg",   # BLEU 0.482 – girl kicking soccer ball
]

# Model's predicted caption for each sample (index-matched)
SAMPLE_LABELS = [
    "a brown dog is running through a grassy field",
    "a person skiing down a snowy hill",
    "two dogs are playing in the water",
    "a man in a red jacket is standing on a street",
    "a group of people are standing in front of a mountain",
    "a little boy in a red shirt is playing with a soccer ball",
]

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024          # 10 MB hard limit
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ---------------------------------------------------------------------------
# Load prediction pipeline once at startup (heavy — models are loaded here)
# ---------------------------------------------------------------------------

logger.info("Loading PredictionPipeline …")
pipeline = PredictionPipeline()
logger.info("PredictionPipeline ready.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed(filename: str) -> bool:
    """Return True if the filename has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Serve the main UI page."""
    return render_template("index.html")


@app.route("/sample-images/<path:filename>", methods=["GET"])
def sample_image(filename):
    """Serve a single dataset image by filename."""
    return send_from_directory(str(DATASET_IMAGES_FOLDER.resolve()), filename)


@app.route("/sample-images", methods=["GET"])
def list_sample_images():
    """Return the list of curated sample image filenames and labels for the gallery."""
    return jsonify({"images": SAMPLE_IMAGES, "labels": SAMPLE_LABELS})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a multipart/form-data POST with a field named ``file``.

    Returns
    -------
    JSON: ``{ "caption": "<generated caption>" }``
    or on error: ``{ "error": "<message>" }`` with an appropriate HTTP status.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _allowed(file.filename):
        return jsonify({"error": "File type not allowed. Use JPG, JPEG or PNG."}), 415

    # Save to a temporary path with a unique name to avoid conflicts
    temp_name = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = UPLOAD_FOLDER / temp_name

    try:
        file.save(str(temp_path))
        logger.info(f"Saved upload to: {temp_path}")

        caption = pipeline.predict(temp_path)
        logger.info(f"Caption generated: {caption}")

        return jsonify({"caption": caption})

    except Exception as exc:
        logger.exception(exc)
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500

    finally:
        # Always clean up the temporary upload
        if temp_path.exists():
            temp_path.unlink()
            logger.info(f"Removed temp file: {temp_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
