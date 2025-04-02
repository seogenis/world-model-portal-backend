import os
import uuid
import asyncio
import tempfile
import zipfile
import shutil
import aiohttp
import time
import json
from typing import Dict, Optional, Tuple, Any, List

from app.core.logger import get_logger
from app.core.config import get_settings, nvidia_api_semaphore

logger = get_logger()
settings = get_settings()

class VideoService:
    """Service for handling video generation requests using Nvidia's API."""
    
    def __init__(self):
        """Initialize the video service with API credentials and worker pool."""
        settings = get_settings()
        
        # Initialize API keys from settings
        self.primary_api_key = settings.NVIDIA_API_KEY
        self.backup_keys = [
            settings.NVIDIA_API_KEY_BACKUP1,
            settings.NVIDIA_API_KEY_BACKUP2
        ]
        # Filter out empty keys
        self.backup_keys = [key for key in self.backup_keys if key and len(key) >= 10]
        
        # Start with the primary key
        self.nvidia_api_key = self.primary_api_key
        self.current_key_index = 0
        self.enable_key_rotation = settings.ENABLE_API_KEY_ROTATION
        
        # Log API key status (without revealing it)
        if not self.nvidia_api_key or len(self.nvidia_api_key) < 10:
            logger.warning("NVIDIA API key is missing or appears invalid - check your .env file")
        else:
            logger.info(f"NVIDIA API key loaded (length: {len(self.nvidia_api_key)})")
            
        if self.backup_keys:
            logger.info(f"Found {len(self.backup_keys)} backup API keys available for rotation")
            
        self.invoke_url = "https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos-predict1-7b"
        self.fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
        self._update_headers()
        # In-memory store for job statuses
        self.video_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Worker pool configuration
        self.max_concurrent_jobs = settings.NVIDIA_MAX_CONCURRENT  # NVIDIA API allows only one concurrent call
        self.active_jobs = 0          # Currently active jobs
        self.job_queue = asyncio.Queue()  # Queue for pending jobs
        self.worker_task = None
        
        # API URLs - updated to match NVIDIA's official example
        self.invoke_url = "https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos-predict1-7b"
        self.fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{}"
        
        # Rate limiting config
        self.retry_delay = settings.NVIDIA_RETRY_DELAY
        self.max_retries = settings.NVIDIA_RETRY_ATTEMPTS
        
        # Sanity check for API limit
        if settings.NVIDIA_MAX_CONCURRENT != 1:
            logger.warning("NVIDIA API strictly enforces a single concurrent API call limit. Setting NVIDIA_MAX_CONCURRENT to a value other than 1 may cause unexpected errors.")
        
    async def startup(self):
        """Start the worker pool - must be called from an async context"""
        if self.worker_task is None:
            # Start worker pool
            self.worker_task = asyncio.create_task(self._job_worker())
            # Make sure worker doesn't die if there's an exception
            self.worker_task.add_done_callback(self._restart_worker)
            logger.info("Video service worker started")
        
    def _restart_worker(self, future):
        """Callback to restart the worker if it crashes"""
        if future.exception():
            logger.error(f"Worker crashed with exception: {future.exception()}")
            # Need to restart in an async context - we'll do this on the next API call
            self.worker_task = None
            logger.info("Worker task will be restarted on next API call")
            
    def _update_headers(self):
        """Update API headers with the current API key"""
        self.headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Accept": "application/json",
        }
        
    def rotate_api_key(self):
        """Rotate to the next available API key in the rotation"""
        if not self.enable_key_rotation or not self.backup_keys:
            logger.warning("API key rotation disabled or no backup keys available")
            return False
            
        # Start with first backup key if we're currently on primary
        if self.nvidia_api_key == self.primary_api_key:
            self.nvidia_api_key = self.backup_keys[0]
            self.current_key_index = 0
            logger.info(f"Rotating from primary API key to backup key 1")
        else:
            # Move to next backup key
            next_index = (self.current_key_index + 1) % len(self.backup_keys)
            self.current_key_index = next_index
            self.nvidia_api_key = self.backup_keys[next_index]
            logger.info(f"Rotating to backup API key {next_index + 1}")
            
        # Update headers with new key
        self._update_headers()
        return True
    
    async def _job_worker(self):
        """Worker that processes jobs from the queue with rate limiting"""
        logger.info("Starting video generation worker pool")
        
        # Maximum age for jobs in queue (5 minutes in seconds)
        MAX_JOB_AGE = 300  # 5 minutes
        
        while True:
            try:
                # Check if we can process more jobs
                if self.active_jobs < self.max_concurrent_jobs:
                    try:
                        # Check and clean expired jobs in queue
                        await self._clean_expired_jobs(MAX_JOB_AGE)
                        
                        # Get next job with timeout (so we can check active_jobs periodically)
                        job_id, prompt, websocket, timestamp = await asyncio.wait_for(
                            self.job_queue.get(), timeout=1.0
                        )
                        
                        # Check if job is too old
                        current_time = time.time()
                        job_age = current_time - timestamp
                        
                        if job_age > MAX_JOB_AGE:
                            # Skip this job and update its status
                            logger.warning(f"Skipping job {job_id} as it's too old ({job_age:.1f} seconds)")
                            self.video_jobs[job_id] = {
                                "status": "expired",
                                "message": f"Job expired after waiting in queue for {job_age:.1f} seconds",
                                "progress": 0,
                                "video_url": None
                            }
                            # Mark job as done in queue and continue
                            self.job_queue.task_done()
                            continue
                        
                        # Update job status
                        self.video_jobs[job_id]["status"] = "starting"
                        self.video_jobs[job_id]["message"] = "Job picked up by worker"
                        
                        # WebSocket updates removed
                        
                        # Start processing
                        self.active_jobs += 1
                        logger.debug(f"Starting job {job_id} (active jobs: {self.active_jobs}/{self.max_concurrent_jobs})")
                        
                        # Start the job processing
                        task = asyncio.create_task(
                            self._process_video_generation(job_id, prompt, websocket)
                        )
                        
                        # Add callback to decrease active job count when done
                        def on_complete(_):
                            self.active_jobs = max(0, self.active_jobs - 1)
                            logger.debug(f"Completed job {job_id} (active jobs: {self.active_jobs}/{self.max_concurrent_jobs})")
                            
                        task.add_done_callback(on_complete)
                        
                        # Mark task as done in queue
                        self.job_queue.task_done()
                        
                    except asyncio.TimeoutError:
                        # No job available, continue
                        pass
                else:
                    # At max capacity, wait before checking again
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error in job worker: {e}")
                await asyncio.sleep(1)  # Avoid tight loop in case of persistent errors
                
    async def _clean_expired_jobs(self, max_age_seconds):
        """Clean up expired jobs from the queue
        
        Args:
            max_age_seconds: Maximum job age in seconds before removing from queue
        """
        # We can't directly access queue items, need to rebuild the queue
        temp_queue = asyncio.Queue()
        expired_count = 0
        
        # Try to get all items from the queue without blocking
        while True:
            try:
                job_id, prompt, websocket, timestamp = await asyncio.wait_for(
                    self.job_queue.get(), timeout=0.01
                )
                
                # Check job age
                current_time = time.time()
                job_age = current_time - timestamp
                
                if job_age > max_age_seconds:
                    # Job expired, update status
                    logger.warning(f"Removing expired job {job_id} from queue (age: {job_age:.1f} seconds)")
                    self.video_jobs[job_id] = {
                        "status": "expired",
                        "message": f"Job expired after waiting in queue for {job_age:.1f} seconds",
                        "progress": 0,
                        "video_url": None
                    }
                    expired_count += 1
                    # Job is not added back to the queue
                else:
                    # Job is still valid, add it back to the temp queue
                    await temp_queue.put((job_id, prompt, websocket, timestamp))
                
                # Mark as done in original queue
                self.job_queue.task_done()
                
            except asyncio.TimeoutError:
                # No more items in the queue
                break
        
        # Now add all items from temp queue back to original queue
        while not temp_queue.empty():
            item = await temp_queue.get()
            await self.job_queue.put(item)
            temp_queue.task_done()
        
        if expired_count > 0:
            logger.info(f"Cleaned {expired_count} expired jobs from the queue")
    
    async def generate_video(self, prompt: str) -> str:
        """
        Generate a video from a prompt using Nvidia API.
        
        Args:
            prompt: The text-to-video prompt
            
        Returns:
            job_id: Unique identifier for tracking the video generation
        """
        # Ensure the worker is running
        await self.startup()
        
        job_id = str(uuid.uuid4())
        
        # Initialize job in memory store immediately
        self.video_jobs[job_id] = {
            "status": "pending",
            "message": "Job queued and will start soon",
            "progress": 0,
            "video_url": None
        }
        
        # Add job to the queue for processing by worker pool with current timestamp
        logger.debug(f"Queueing job {job_id}")
        current_time = time.time()
        await self.job_queue.put((job_id, prompt, None, current_time))  # Added timestamp
        
        # Return job_id immediately
        return job_id
        

    async def _process_video_generation(self, job_id: str, prompt: str, websocket=None) -> None:
        """
        Process the video generation job using NVIDIA's official API pattern.
        
        Args:
            job_id: The job ID
            prompt: The text-to-video prompt
            websocket: Parameter kept for backward compatibility
        """
        try:
            # Update status in memory store
            self.video_jobs[job_id] = {
                "status": "generating",
                "message": "Generating video from prompt",
                "progress": 10,
                "video_url": None
            }
            
            # WebSocket updates removed
            
            # Create the payload for Nvidia API - using the official format
            payload = {
                "inputs": [
                    {
                        "name": "command",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [
                            f"text2world --prompt=\"{prompt}\""
                        ]
                    }
                ],
                "outputs": [
                    {
                        "name": "status",
                        "datatype": "BYTES",
                        "shape": [1]
                    }
                ]
            }
            
            # Use the class's retry settings
            retry_delay = self.retry_delay
            max_retries = self.max_retries
            
            # Update job status to waiting for API availability (semaphore access)
            self.video_jobs[job_id] = {
                "status": "pending",
                "message": "Waiting for NVIDIA API availability (only one job can access API at a time)...",
                "progress": 5,
                "video_url": None
            }
            
            # WebSocket updates removed
            
            # NVIDIA API allows only one concurrent call - we must use a semaphore
            logger.debug(f"Waiting for NVIDIA API access for job {job_id}")
            
            # Try to acquire the semaphore (this will block if another job is using the API)
            # This is critical for NVIDIA API which strictly enforces a single concurrent request limit
            async with nvidia_api_semaphore:
                logger.debug(f"Acquired NVIDIA API access for job {job_id}")
                
                # Update job status to sending request
                self.video_jobs[job_id] = {
                    "status": "generating",
                    "message": "Sending request to NVIDIA API...",
                    "progress": 10,
                    "video_url": None
                }
                
                # Use aiohttp to make async HTTP requests following NVIDIA's official pattern
                async with aiohttp.ClientSession() as session:
                    request_id = None
                    content = None
                    content_type = None
                    
                    # Step 1: Submit request to NVIDIA API
                    try:
                        logger.info(f"Submitting request to NVIDIA API for job {job_id}")
                        async with session.post(self.invoke_url, headers=self.headers, json=payload) as initial_response:
                            # Log response status for debugging, only log headers at debug level
                            logger.info(f"Initial response status: {initial_response.status}")
                            if settings.DEBUG:
                                logger.debug(f"Initial response headers: {dict(initial_response.headers)}")
                            
                            # Handle different status codes
                            if initial_response.status == 429:
                                error_text = await initial_response.text()
                                logger.warning(f"NVIDIA API rate limited: {error_text}")
                                # This typically happens if another job circumvented our semaphore system
                                # or if another client is accessing the same API key
                                logger.error("Rate limit detected despite semaphore protection - possible API key sharing or semaphore bypass")
                                raise RuntimeError(f"NVIDIA API rate limited: {error_text} - possibly due to shared API usage")
                                
                            elif initial_response.status in [401, 403]:
                                error_text = await initial_response.text()
                                logger.error(f"NVIDIA API authentication error: {initial_response.status}, {error_text}")
                                
                                # Try to rotate API key if enabled
                                if self.rotate_api_key():
                                    logger.info("API key rotated due to authentication error, retrying request")
                                    # Retry the request with the new key
                                    async with session.post(self.invoke_url, headers=self.headers, json=payload) as retry_response:
                                        if retry_response.status in [200, 202]:
                                            logger.info("Request succeeded after API key rotation")
                                            initial_response = retry_response
                                            # Continue processing with the new response
                                        else:
                                            # Still failing, raise error
                                            retry_error = await retry_response.text()
                                            logger.error(f"Request still failing after API key rotation: {retry_response.status}, {retry_error}")
                                            raise ValueError(f"NVIDIA API authentication error persists after key rotation: {retry_response.status}")
                                else:
                                    # No key rotation available
                                    raise ValueError(f"NVIDIA API authentication error: {initial_response.status} - check your API key")
                                
                            elif initial_response.status == 429:
                                # Rate limit error - try key rotation
                                error_text = await initial_response.text()
                                logger.warning(f"NVIDIA API rate limited: {error_text}")
                                
                                # Try to rotate API key if enabled
                                if self.rotate_api_key():
                                    logger.info("API key rotated due to rate limit error, retrying request")
                                    # Retry the request with the new key
                                    async with session.post(self.invoke_url, headers=self.headers, json=payload) as retry_response:
                                        if retry_response.status in [200, 202]:
                                            logger.info("Request succeeded after API key rotation")
                                            initial_response = retry_response
                                            # Continue processing with the new response
                                        else:
                                            # Still failing, raise error
                                            retry_error = await retry_response.text()
                                            logger.error(f"Request still failing after API key rotation: {retry_response.status}, {retry_error}")
                                            raise RuntimeError(f"NVIDIA API rate limit persists after key rotation: {retry_response.status}")
                                else:
                                    # No key rotation available
                                    raise RuntimeError(f"NVIDIA API rate limited: {error_text} - possibly due to shared API usage")
                                
                            elif initial_response.status not in [200, 202]:
                                error_text = await initial_response.text()
                                logger.error(f"NVIDIA API error: {initial_response.status}, {error_text}")
                                
                                # Try to rotate API key for other errors too
                                if self.rotate_api_key():
                                    logger.info(f"API key rotated due to error {initial_response.status}, retrying request")
                                    # Retry the request with the new key
                                    async with session.post(self.invoke_url, headers=self.headers, json=payload) as retry_response:
                                        if retry_response.status in [200, 202]:
                                            logger.info("Request succeeded after API key rotation")
                                            initial_response = retry_response
                                            # Continue processing with the new response
                                        else:
                                            # Still failing, raise error
                                            retry_error = await retry_response.text()
                                            logger.error(f"Request still failing after API key rotation: {retry_response.status}, {retry_error}")
                                            raise RuntimeError(f"NVIDIA API error persists after key rotation: {initial_response.status}, {error_text}")
                                else:
                                    # No key rotation available
                                    raise RuntimeError(f"NVIDIA API error: {initial_response.status}, {error_text}")
                            
                            # For status 200: immediate response (rare)
                            if initial_response.status == 200:
                                content = await initial_response.read()
                                content_type = initial_response.headers.get('Content-Type', '')
                                logger.info(f"Received immediate response for job {job_id}")
                                
                            # For status 202: async processing (normal case)
                            elif initial_response.status == 202:
                                # Get request_id from headers (key is case-sensitive)
                                request_id = initial_response.headers.get("NVCF-REQID")
                                
                                # If not in headers, try response body (different API versions may vary)
                                if not request_id:
                                    try:
                                        resp_body = await initial_response.json()
                                        # Only log response body at debug level
                                        if settings.DEBUG:
                                            logger.debug(f"Response body: {resp_body}")
                                        
                                        # Try to find request ID in various possible locations
                                        if 'id' in resp_body:
                                            request_id = resp_body['id']
                                        elif 'request_id' in resp_body:
                                            request_id = resp_body['request_id']
                                    except Exception as e:
                                        logger.warning(f"Could not parse response body: {e}")
                                
                                # Still no request_id? That's an error
                                if not request_id:
                                    logger.error("No request ID returned from NVIDIA API")
                                    raise ValueError("No request ID returned from NVIDIA API - check your API key")
                                
                                logger.info(f"Request {job_id} accepted, request_id: {request_id}")
                    
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        logger.error(f"Network error submitting request: {str(e)}")
                        raise RuntimeError(f"Network error submitting request: {str(e)}")
                    
                    # If we have content already (status 200), skip polling
                    if content is not None:
                        logger.info(f"Already received content for job {job_id} (immediate response)")
                    
                    # Step 2: Poll for results (status 202 case)
                    elif request_id:
                        # Format the fetch URL with the request ID
                        fetch_url = self.fetch_url_format.format(request_id)
                        logger.debug(f"Will poll for results at: {fetch_url}")
                        
                        # Initial values for polling
                        progress = 10
                        poll_interval = 2  # Start with 2 seconds
                        max_poll_interval = 10  # Max 10 seconds between polls
                        max_poll_time = 600  # Max 10 minutes polling
                        start_time = time.time()
                        
                        # Step 3: Poll until completion
                        while True:
                            # Check if we've been polling too long
                            if time.time() - start_time > max_poll_time:
                                raise TimeoutError(f"Polling timed out after {max_poll_time} seconds")
                            
                            # Update progress for user
                            progress += 5
                            progress = min(progress, 90)
                            
                            # Update job status
                            self.video_jobs[job_id] = {
                                "status": "generating",
                                "message": f"Generating video... (request_id: {request_id})",
                                "progress": progress,
                                "video_url": None
                            }
                            
                            # WebSocket updates removed
                            
                            # Wait before polling
                            await asyncio.sleep(poll_interval)
                            
                            # Poll for status
                            try:
                                logger.debug(f"Polling status for job {job_id}, request_id {request_id}")
                                async with session.get(fetch_url, headers=self.headers) as status_response:
                                    status_code = status_response.status
                                    logger.debug(f"Poll response status: {status_code} for job {job_id}")
                                    
                                    # Status 202: Still processing
                                    if status_code == 202:
                                        # Increase poll interval with a cap
                                        poll_interval = min(poll_interval * 1.5, max_poll_interval)
                                        logger.debug(f"Job {job_id} still processing. Next poll in {poll_interval}s")
                                        continue
                                        
                                    # Status 200: Complete
                                    elif status_code == 200:
                                        logger.info("Processing complete, getting results")
                                        content = await status_response.read()
                                        content_type = status_response.headers.get('Content-Type', '')
                                        break
                                        
                                    # Rate limiting
                                    elif status_code == 429:
                                        error_text = await status_response.text()
                                        logger.warning(f"Rate limited during polling: {error_text}")
                                        
                                        # Try to rotate API key if enabled
                                        if self.rotate_api_key():
                                            logger.info("API key rotated due to rate limit during polling, retrying poll")
                                            # Update fetch URL with new headers
                                            async with session.get(fetch_url, headers=self.headers) as retry_response:
                                                if retry_response.status == 200:
                                                    logger.info("Polling succeeded after API key rotation")
                                                    content = await retry_response.read()
                                                    content_type = retry_response.headers.get('Content-Type', '')
                                                    break
                                                elif retry_response.status == 202:
                                                    logger.info("Polling still in progress after API key rotation")
                                                    # Continue with normal polling
                                                    poll_interval = min(poll_interval * 1.5, max_poll_interval)
                                                    continue
                                                else:
                                                    # Something else went wrong
                                                    logger.warning(f"Polling still failing after API key rotation: {retry_response.status}")
                                                    # Fall back to increasing poll interval
                                                    poll_interval = min(poll_interval * 2, max_poll_interval)
                                                    continue
                                        else:
                                            # No key rotation available, just increase poll interval
                                            poll_interval = min(poll_interval * 2, max_poll_interval)
                                            continue
                                        
                                    # Authentication errors (401/403)
                                    elif status_code in [401, 403]:
                                        error_text = await status_response.text()
                                        logger.error(f"Authentication error during polling: {status_code}, {error_text}")
                                        
                                        # Try to rotate API key if enabled
                                        if self.rotate_api_key():
                                            logger.info("API key rotated due to authentication error during polling, retrying poll")
                                            # Update fetch URL with new headers
                                            async with session.get(fetch_url, headers=self.headers) as retry_response:
                                                if retry_response.status in [200, 202]:
                                                    logger.info("Polling succeeded after API key rotation")
                                                    if retry_response.status == 200:
                                                        content = await retry_response.read()
                                                        content_type = retry_response.headers.get('Content-Type', '')
                                                        break
                                                    else:
                                                        # Still processing
                                                        continue
                                                else:
                                                    # Still failing 
                                                    retry_error = await retry_response.text()
                                                    logger.error(f"Polling still failing after API key rotation: {retry_response.status}, {retry_error}")
                                                    raise RuntimeError(f"Authentication error persists during polling after key rotation: {retry_response.status}")
                                        else:
                                            # No key rotation available
                                            raise RuntimeError(f"Authentication error during polling: {status_code}, {error_text}")
                                        
                                    # Other errors
                                    else:
                                        error_text = await status_response.text()
                                        logger.error(f"Error during polling: {status_code}, {error_text}")
                                        
                                        # Try to rotate API key for other errors too
                                        if self.rotate_api_key():
                                            logger.info(f"API key rotated due to error {status_code} during polling, retrying poll")
                                            # Update fetch URL with new headers
                                            async with session.get(fetch_url, headers=self.headers) as retry_response:
                                                if retry_response.status in [200, 202]:
                                                    logger.info("Polling succeeded after API key rotation")
                                                    if retry_response.status == 200:
                                                        content = await retry_response.read()
                                                        content_type = retry_response.headers.get('Content-Type', '')
                                                        break
                                                    else:
                                                        # Still processing
                                                        continue
                                                else:
                                                    # Still failing
                                                    retry_error = await retry_response.text()
                                                    logger.error(f"Polling still failing after API key rotation: {retry_response.status}, {retry_error}")
                                                    raise RuntimeError(f"Error persists during polling after key rotation: {status_code}, {error_text}")
                                        else:
                                            # No key rotation available
                                            raise RuntimeError(f"Error during polling: {status_code}, {error_text}")
                                        
                            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                                logger.warning(f"Network error during polling: {str(e)}")
                                # Don't fail immediately, try again after a delay
                                poll_interval = min(poll_interval * 2, max_poll_interval)
                                continue
            
            # Update status to processing the received content
            self.video_jobs[job_id] = {
                "status": "processing",
                "message": "Processing video files",
                "progress": 95,
                "video_url": None
            }
            
            # WebSocket updates removed
            
            # Create directory for storing the video
            output_dir = f"static/videos/{job_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Process the content - according to the NVIDIA official example,
            # the API should always return a ZIP file containing the video
            try:
                # Step 4: Save the result (ZIP file)
                zip_path = f"{output_dir}/result.zip"
                logger.debug(f"Saving content to {zip_path}")
                with open(zip_path, 'wb') as f:
                    f.write(content)
                
                # Step 5: Extract the ZIP file
                logger.debug(f"Extracting ZIP file for job {job_id}")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(output_dir)
                    
                    # Standard path for video file according to NVIDIA example
                    video_path = f"{output_dir}/video.mp4"
                    
                    # If the standard path doesn't exist, look for any MP4 file
                    if not os.path.exists(video_path):
                        logger.debug("Standard video.mp4 not found, searching for any MP4 file")
                        mp4_files = []
                        for root, _, files in os.walk(output_dir):
                            for file in files:
                                if file.endswith(".mp4"):
                                    mp4_files.append(os.path.join(root, file))
                        
                        if mp4_files:
                            # Use the first found MP4 file and copy it to the standard location
                            source_path = mp4_files[0]
                            logger.debug(f"Found MP4 file at {source_path}, copying to {video_path}")
                            shutil.copy(source_path, video_path)
                        else:
                            logger.error("No MP4 files found in extracted content")
                            # List files in directory for debugging
                            all_files = []
                            for root, _, files in os.walk(output_dir):
                                for file in files:
                                    all_files.append(os.path.join(root, file))
                            logger.info(f"Files in extracted content: {all_files}")
                            raise FileNotFoundError("No MP4 files found in extracted content")
                
                except zipfile.BadZipFile:
                    logger.warning("Content is not a valid ZIP file, checking if it's directly an MP4")
                    
                    # Check if content might be a direct MP4 file
                    content_start = content[:16]
                    is_mp4 = False
                    
                    # Check for MP4 file signature
                    if len(content_start) > 8 and content_start[4:8] == b'ftyp':
                        is_mp4 = True
                        logger.info("Content appears to be a direct MP4 file (has signature)")
                    
                    if is_mp4 or 'video/mp4' in content_type.lower():
                        # Save as MP4 directly
                        video_path = f"{output_dir}/video.mp4"
                        logger.info(f"Saving content as direct MP4 to {video_path}")
                        with open(video_path, 'wb') as f:
                            f.write(content)
                    else:
                        # Not ZIP or MP4, save for inspection
                        error_file = f"{output_dir}/response.bin"
                        logger.error(f"Content is neither ZIP nor MP4, saving to {error_file}")
                        with open(error_file, 'wb') as f:
                            f.write(content)
                        
                        # Save content type for debugging
                        with open(f"{output_dir}/content_type.txt", 'w') as f:
                            f.write(content_type)
                            
                        # Try to decode as text for debugging
                        try:
                            text_content = content.decode('utf-8')
                            with open(f"{output_dir}/response.txt", 'w') as f:
                                f.write(text_content)
                        except UnicodeDecodeError:
                            logger.warning("Content could not be decoded as text")
                            
                        raise ValueError(f"Response is not a ZIP or MP4 file. Content-Type: {content_type}")
                
                # Create a URL for the video file
                video_url = f"/static/videos/{job_id}/video.mp4"
                logger.debug(f"Video URL: {video_url}")
                
                # Update status to complete with clear indication of video path
                self.video_jobs[job_id] = {
                    "status": "complete",
                    "message": f"Video generation complete. Video available at: /static/videos/{job_id}/video.mp4",
                    "progress": 100,
                    "video_url": video_url
                }
                
                # WebSocket updates removed
                        
                logger.info(f"Video generation complete for job {job_id}")
                
            except Exception as e:
                logger.error(f"Error processing video content: {str(e)}")
                raise ValueError(f"Error processing video content: {str(e)}")
            
        except Exception as e:
            # Log detailed error information
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error generating video for job {job_id}: {str(e)}")
            logger.debug(f"Error details: {error_details}")
            
            # Create error message based on type of exception
            error_message = str(e)
            if "authentication error" in error_message.lower() or "check your api key" in error_message.lower():
                error_message = "NVIDIA API authentication error - please check your API key"
            elif "rate limited" in error_message.lower():
                error_message = "Rate limited by NVIDIA API - please try again later"
            elif "timeout" in error_message.lower():
                error_message = "Request timed out - the NVIDIA API may be experiencing high load"
            elif "no request id" in error_message.lower():
                error_message = "No request ID received from NVIDIA API - check your API key and configuration"
            
            # Update status to failed
            self.video_jobs[job_id] = {
                "status": "failed",
                "message": f"Error generating video: {error_message}",
                "progress": 0,
                "video_url": None
            }
            
            # WebSocket error updates removed
                    
            # Save error details to file for debugging
            try:
                output_dir = f"static/videos/{job_id}"
                os.makedirs(output_dir, exist_ok=True)
                with open(f"{output_dir}/{job_id}.error", "w") as error_file:
                    error_file.write(f"Error: {str(e)}\n\nStack trace:\n{error_details}")
            except Exception as file_error:
                logger.warning(f"Failed to save error details to file: {file_error}")