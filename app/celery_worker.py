# celery_worker.py

import asyncio
from celery import Celery
from app.services.video_service import VideoService # adjust this import to match your project structure

# Tailscale IPs for your EC2 orchestrator instance
RABBITMQ_HOST = "100.70.212.98"   # replace with actual Tailscale IP
REDIS_HOST = "100.70.212.98"

celery_app = Celery(
    "video_tasks",
    broker=f"pyamqp://guest@{RABBITMQ_HOST}:5672//",
    backend=f"redis://{REDIS_HOST}:6379/0"
)

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True)
def run_video_generation(self, prompt: str) -> str:
    """
    Celery task to generate a video via VideoService and return the video URL.
    """
    try:
        service = VideoService()

        # Call the service's async method from sync context
        loop = asyncio.get_event_loop()
        job_id = loop.run_until_complete(service._process_video_generation_job(prompt))

        return job_id  # return video ID or path
    except Exception as e:
        return f"ERROR: {str(e)}"
