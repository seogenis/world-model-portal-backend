# Frontend Developer Guide

This guide provides instructions for frontend developers working with the World Model Portal backend API.

## Core Concepts

The World Model Portal backend provides:
1. Text-to-video generation using NVIDIA's Cosmos model
2. Prompt tuning and enhancement capabilities
3. Batch video generation across multiple GPUs

## Recommended Implementation Pattern

### For Single Video Generation:

```javascript
// 1. Generate a video
async function generateVideo(prompt) {
  const response = await fetch('/api/video/single_inference', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt })
  });
  
  const data = await response.json();
  return data.job_id;
}

// 2. Poll for status updates
async function pollVideoStatus(jobId) {
  const POLL_INTERVAL = 2000; // 2 seconds
  let isComplete = false;
  
  while (!isComplete) {
    try {
      const response = await fetch(`/api/video/status/${jobId}`);
      const statusData = await response.json();
      
      // Update UI with status
      updateStatusUI(statusData);
      
      if (statusData.status === 'complete') {
        // Video is ready - display it
        displayVideo(statusData.video_url);
        isComplete = true;
      } else if (statusData.status === 'failed') {
        // Handle failure
        showError(statusData.message);
        isComplete = true;
      } else {
        // Wait before polling again
        await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
      }
    } catch (error) {
      console.error('Error polling video status:', error);
      showError('Error checking video status');
      isComplete = true;
    }
  }
}

// Example usage
async function handleVideoGeneration() {
  const prompt = "A beautiful sunset over mountains";
  const jobId = await generateVideo(prompt);
  await pollVideoStatus(jobId);
}
```

### For Batch Video Generation:

```javascript
// 1. Generate multiple videos
async function generateBatchVideos(prompts) {
  const response = await fetch('/api/video/batch_inference', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompts })
  });
  
  const data = await response.json();
  return data.batch_id;
}

// 2. Poll for batch status updates
async function pollBatchStatus(batchId) {
  const POLL_INTERVAL = 3000; // 3 seconds
  let isComplete = false;
  
  while (!isComplete) {
    try {
      const response = await fetch(`/api/video/batch_status/${batchId}`);
      const batchData = await response.json();
      
      // Update UI with batch status
      updateBatchStatusUI(batchData);
      
      if (batchData.status === 'complete' || batchData.status === 'failed') {
        isComplete = true;
        
        if (batchData.completed > 0) {
          // Offer download button for completed videos
          showDownloadButton(`/api/video/batch_download/${batchId}`);
        }
      } else {
        // Wait before polling again
        await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
      }
    } catch (error) {
      console.error('Error polling batch status:', error);
      showError('Error checking batch status');
      isComplete = true;
    }
  }
}

// Example usage
async function handleBatchGeneration() {
  const prompts = [
    "A sunrise over mountains",
    "A sunset over ocean",
    "Clouds over a forest"
  ];
  
  const batchId = await generateBatchVideos(prompts);
  await pollBatchStatus(batchId);
}
```

## Important Implementation Notes

1. **Avoid WebSockets**: While WebSocket endpoints are still available, they are not recommended due to reliability issues. Use the polling approach instead.

2. **Error Handling**: Always implement proper error handling, including:
   - Network failures
   - Server errors (500 responses)
   - Timeouts (videos can take 1-2 minutes to generate)

3. **Proper Video Display**: When displaying videos:
   - Use the video URL from the status response
   - Ensure proper MIME types and headers
   - Implement fallback content for video loading failures

4. **Progress Indication**: Use the `progress` field from status responses to show meaningful progress indicators to users.

## Reference Examples

Study these example files for best practices:

1. **video_generator_complete.html**: Comprehensive example with all features
2. **polling_example.html**: Minimal implementation of the polling pattern
3. **batch_video_generator.html**: Legacy implementation retained for reference

## API Endpoints

For complete API documentation, see [API_DOCUMENTATION.md](./static/API_DOCUMENTATION.md).

## Troubleshooting

1. **Videos Not Showing**: Check that your application is correctly accessing the video URL. Video URLs are in the format `/api/videos/{job_id}`.

2. **Status Shows "pending" Too Long**: If a job stays in the "pending" state for more than 5 minutes, it may have failed silently. Consider implementing a timeout.

3. **CORS Issues**: Make sure your frontend is either served from the same domain as the API or that CORS is properly configured on the server.