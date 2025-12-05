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
        s3_key = f"miketest2/{name}/{uuid.uuid4()}.json"

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


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
