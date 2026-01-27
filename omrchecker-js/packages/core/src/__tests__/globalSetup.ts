// Global setup runs ONCE before all test workers start
import { resolve } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

export default async function globalSetup() {
  console.log('[Global Setup] Running one-time setup...');

  // Verify opencv.js exists and is readable
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = dirname(__filename);
  const opencvPath = resolve(__dirname, '../../lib/opencv.js');

  try {
    const stats = await import('fs').then((fs) => fs.promises.stat(opencvPath));
    console.log(
      `[Global Setup] ✓ OpenCV.js verified at ${opencvPath} (${(stats.size / 1024 / 1024).toFixed(2)} MB)`
    );
  } catch (error) {
    throw new Error(`[Global Setup] ✗ OpenCV.js not found at ${opencvPath}: ${error}`);
  }

  // Store the path in an environment variable for workers to use
  process.env.OPENCV_JS_PATH = opencvPath;

  console.log('[Global Setup] ✓ Complete - workers now start loading OpenCV');

  // Return a teardown function (optional)
  return () => {
    console.log('[Global Teardown] Cleaning up...');
  };
}
