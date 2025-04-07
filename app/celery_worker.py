# celery_worker.py

import asyncio
from celery import Celery
from app.services.video_service import VideoService # adjust this import to match your project structure

import boto3
from app.core.config import get_settings
from app.core.logger import get_logger

# Tailscale IPs for your EC2 orchestrator instance
RABBITMQ_HOST = "100.70.212.98"   # replace with actual Tailscale IP
REDIS_HOST = "100.70.212.98"

settings = get_settings()
logger = get_logger()

# Initialize S3 client with error handling
try:
    if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
        logger.warning("AWS credentials not set in environment variables or .env file")
        s3_client = None
    else:
        s3_client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        logger.info("S3 client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {str(e)}")
    s3_client = None

celery_app = Celery(
    "video_tasks",
    broker=f"pyamqp://guest@{RABBITMQ_HOST}:5672//",
    backend=f"redis://{REDIS_HOST}:6379/0"
)
celery_app.conf.task_acks_on_failure_or_timeout = True

@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 15}
)
def run_video_generation(self, prompt: str, video_key: str = None) -> str:
    """
    Celery task to generate a video via VideoService and return the video URL.
    """
    try:
        video_path = None
        if video_key:  # If a video key is provided, download from S3
            video_path = download_video_from_s3(video_key)
            logger.info(f"Using video/image input from S3: {video_path}")

        service = VideoService()

        # Call the service's async method from sync context
        loop = asyncio.get_event_loop()
        video_url = loop.run_until_complete(service._process_video_generation_job(prompt, video_path))

        s3_key = upload_video_to_s3(video_url)

        return s3_key  # return video ID or path
    except Exception as e:
        return f"ERROR: {str(e)}"

def upload_video_to_s3(video_url: str) -> str:
    """
    Uploads the video file to S3 under a folder named with the job_id.
    Returns the S3 key for the uploaded video.
    """
    bucket_name = "cosmos-storage"
    s3_key = video_url
    local_file_path = "/workspace/world-model-portal-backend"+video_url
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        return s3_key
    except Exception as e:
        raise RuntimeError(f"Failed to upload video: {str(e)}")

def download_video_from_s3(video_key: str) -> str:
    bucket_name = "cosmos-storage"
    local_file_path = video_key
    try:
        s3.download_file(bucket_name, video_key, local_file_path)
        print(f"✅ File downloaded: {local_file_path}")
        return local_file_path
    except Exception as e:
        print(f"❌ Failed to download file: {e}")
