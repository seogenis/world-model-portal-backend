#!/bin/bash

echo "Starting uvicorn orchestrator server on port 80"

sudo .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 80 --reload