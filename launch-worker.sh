#!/bin/bash
# chmod +x /workspace/world-model-portal-backend/launch-worker.sh


echo "Starting celery worker"

celery -A app.celery_worker.celery_app worker --loglevel=info --concurrency=1