import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config
from dotenv import load_dotenv
from importlib import resources

from model import KeystrokeDataProcessor, load_model, predict_single

load_dotenv()

app = FastAPI(debug=True)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize S3 client
s3_client = boto3.client(
    's3',
    config=Config(signature_version='s3v4')
)

# Get bucket name from environment variable
S3_BUCKET = os.getenv('S3_BUCKET_NAME')

# Model configuration
MODEL_FILE = resources.files("model.saved").joinpath("keystroke_model.pth")
# MODEL_FILE = os.path.join(os.path.dirname(__file__), '..', 'model', 'keystroke_model.pth')

# Global variables for model artifacts (loaded on startup)
model = None
scaler = None
class_map = None
device = None


# Load model on startup
try:
    with MODEL_FILE.open("rb") as f:
        model, scaler, class_map, device = load_model(f)
        print(f"Model loaded successfully. Classes: {class_map}")
except Exception as e:
    print(f"Warning: Failed to load model on startup: {e}")
    print("Inference endpoint will not be available until model is loaded.")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload a file to S3 bucket.

    Args:
        file: The file to upload

    Returns:
        JSON response with upload status and file details
    """
    if not S3_BUCKET:
        raise HTTPException(
            status_code=500,
            detail="S3_BUCKET_NAME environment variable not set"
        )

    try:
        # Read file content
        file_content = await file.read()

        # Parse JSON to extract name field
        try:
            data = json.loads(file_content)
            name = data.get("name")
            if not name:
                raise HTTPException(
                    status_code=400,
                    detail="JSON must contain a 'name' field"
                )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON format"
            )

        # Generate random UUID filename with name as parent directory
        s3_key = f"new/{name}/{uuid.uuid4()}.json"

        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=file_content,
            ContentType='application/json'
        )

        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully",
                "key": s3_key,
                "name": name,
                "original_filename": file.filename,
                "bucket": S3_BUCKET,
                "size": len(file_content)
            }
        )

    except ClientError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload to S3: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    """
    Perform inference on keystroke data to identify the user.

    Args:
        file: JSON file containing keystroke logs

    Returns:
        JSON response with prediction results
    """
    # Check if model is loaded
    if model is None or scaler is None or class_map is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Read file content
        file_content = await file.read()

        # Parse JSON to extract keystroke logs
        try:
            data = json.loads(file_content)
            keystroke_logs = data.get("keystrokeLogs", [])
            if not keystroke_logs:
                raise HTTPException(
                    status_code=400,
                    detail="JSON must contain a 'keystrokeLogs' field with data"
                )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON format"
            )

        # Extract features using KeystrokeDataProcessor
        processor = KeystrokeDataProcessor()
        features = processor._extract_features(keystroke_logs)

        # Truncate to 41 features (same as training)
        features = features[:41]

        if len(features) == 0:
            raise HTTPException(
                status_code=400,
                detail="No valid keystroke features extracted"
            )

        # Perform prediction
        predicted_user, confidence = predict_single(
            model, features, scaler, device, class_map
        )

        return JSONResponse(
            status_code=200,
            content={
                "predicted_user": predicted_user,
                "confidence": float(confidence),
                "num_keystrokes": len(features),
                "original_filename": file.filename,
                "all_classes": class_map
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
