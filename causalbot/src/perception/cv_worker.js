import { pipeline, env } from '@xenova/transformers';

// Disable local models fallback to ensure it fetches from HF Hub correctly
env.allowLocalModels = false;

let detector = null;

console.log('[CVWorker] Thread started, initializing pipeline...');

// Initialize the model in the background
pipeline('object-detection', 'Xenova/detr-resnet-50').then(pipe => {
  detector = pipe;
  console.log('[CVWorker] Pipeline ready!');
  postMessage({ type: 'ready' });
}).catch(err => {
  console.error('[CVWorker] Failed to load pipeline:', err);
  postMessage({ type: 'error', error: err.message });
});

// Listen for inference requests from the main thread
self.addEventListener('message', async (event) => {
  const { id, dataUrl } = event.data;
  
  if (!detector) {
    postMessage({ id, error: 'Detector not initialized yet' });
    return;
  }

  try {
    // Run the heavy inference on the worker thread
    // Threshold bumped to 0.45 to prevent wild hallucinations
    const output = await detector(dataUrl, { threshold: 0.45, percentage: false });
    postMessage({ id, output });
  } catch (error) {
    postMessage({ id, error: error.message });
  }
});
