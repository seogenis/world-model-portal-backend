#!/bin/bash
# chmod +x /workspace/world-model-portal-backend/flush_celery.sh
# Flush Celery tasks and RabbitMQ queues

# 1. Check if we have the right tools installed
if ! command -v redis-cli &> /dev/null; then
    echo "redis-cli is not installed. Install it with: apt-get install redis-tools"
    exit 1
fi

# Flush Redis results backend
echo "Flushing Redis results backend..."
redis-cli -h 100.70.212.98 flushdb

# Purge all Celery tasks in queues
echo "Purging Celery task queues..."
celery -A app.celery_worker.celery_app purge -f

# Kill running Celery worker processes (optional, can be commented out if needed)
echo "Killing Celery worker processes..."
pkill -f "celery worker"

fi

echo "Celery tasks and queues have been flushed."