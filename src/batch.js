/**
 * Batch API helpers for the NanoBanana MCP Server.
 *
 * Handles: JSONL creation, file upload to Gemini Files API,
 * batch job submission, status polling, and result retrieval.
 *
 * Job tracking is persisted to data/batch_jobs.json.
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.join(__dirname, "..", "data");
const JOBS_FILE = path.join(DATA_DIR, "batch_jobs.json");

// Ensure data directory exists
fs.mkdirSync(DATA_DIR, { recursive: true });

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GEMINI_API = "https://generativelanguage.googleapis.com";
const FILES_UPLOAD_URL = `${GEMINI_API}/upload/v1beta/files`;
const BATCHES_URL = `${GEMINI_API}/v1beta/batches`;

// ---------------------------------------------------------------------------
// Job tracking persistence
// ---------------------------------------------------------------------------

/**
 * Load all tracked batch jobs from disk.
 * @returns {Array<object>}
 */
export function loadJobs() {
  if (!fs.existsSync(JOBS_FILE)) return [];
  try {
    return JSON.parse(fs.readFileSync(JOBS_FILE, "utf-8"));
  } catch {
    return [];
  }
}

/**
 * Save the full jobs array to disk.
 */
export function saveJobs(jobs) {
  fs.writeFileSync(JOBS_FILE, JSON.stringify(jobs, null, 2));
}

/**
 * Append a new job record and persist.
 */
export function trackJob(job) {
  const jobs = loadJobs();
  jobs.push(job);
  saveJobs(jobs);
}

/**
 * Update a job by its batch name and persist.
 */
export function updateJob(batchName, updates) {
  const jobs = loadJobs();
  const idx = jobs.findIndex(j => j.batchName === batchName);
  if (idx !== -1) {
    Object.assign(jobs[idx], updates);
    saveJobs(jobs);
  }
  return jobs[idx] ?? null;
}

// ---------------------------------------------------------------------------
// Gemini Files API helpers
// ---------------------------------------------------------------------------

/**
 * Upload a local file to the Gemini Files API using resumable upload.
 *
 * @param {string} filePath  - Local path to the file
 * @param {string} apiKey    - Gemini API key
 * @param {string} mimeType  - MIME type (default: text/plain for JSONL)
 * @returns {Promise<string>} - The file resource name (e.g. "files/abc123")
 */
export async function uploadFile(filePath, apiKey, mimeType = "text/plain") {
  const fileBytes = fs.readFileSync(filePath);
  const displayName = path.basename(filePath);

  // Step 1: Initiate resumable upload
  const initRes = await fetch(`${FILES_UPLOAD_URL}?key=${apiKey}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Goog-Upload-Protocol": "resumable",
      "X-Goog-Upload-Command": "start",
      "X-Goog-Upload-Header-Content-Length": String(fileBytes.length),
      "X-Goog-Upload-Header-Content-Type": mimeType,
    },
    body: JSON.stringify({ file: { displayName } }),
  });

  if (!initRes.ok) {
    const err = await initRes.text();
    throw new Error(`File upload init failed (${initRes.status}): ${err}`);
  }

  const uploadUrl = initRes.headers.get("x-goog-upload-url");
  if (!uploadUrl) {
    throw new Error("No upload URL returned from Files API");
  }

  // Step 2: Upload bytes
  const uploadRes = await fetch(uploadUrl, {
    method: "PUT",
    headers: {
      "Content-Length": String(fileBytes.length),
      "X-Goog-Upload-Offset": "0",
      "X-Goog-Upload-Command": "upload, finalize",
    },
    body: fileBytes,
  });

  if (!uploadRes.ok) {
    const err = await uploadRes.text();
    throw new Error(`File upload failed (${uploadRes.status}): ${err}`);
  }

  const result = await uploadRes.json();
  return result.file?.name ?? result.name;
}

/**
 * Download a file from the Gemini Files API.
 *
 * @param {string} fileName - File resource name (e.g. "files/abc123")
 * @param {string} apiKey   - Gemini API key
 * @returns {Promise<string>} - File contents as a string
 */
export async function downloadFile(fileName, apiKey) {
  const url = `${GEMINI_API}/v1beta/${fileName}:download?key=${apiKey}&alt=media`;
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`File download failed (${res.status}): ${err}`);
  }
  return res.text();
}

// ---------------------------------------------------------------------------
// Batch API helpers
// ---------------------------------------------------------------------------

/**
 * Create a batch job from an uploaded JSONL file.
 *
 * @param {string} model      - Model ID (e.g. "gemini-3-pro-image-preview")
 * @param {string} fileName   - Uploaded file name from Files API
 * @param {string} apiKey     - Gemini API key
 * @param {string} displayName - Human-readable batch name
 * @returns {Promise<object>}  - Batch job object from Gemini
 */
export async function createBatchJob(model, fileName, apiKey, displayName) {
  const url = `${GEMINI_API}/v1beta/models/${model}:batchGenerateContent?key=${apiKey}`;

  const body = {
    batch: {
      display_name: displayName,
      input_config: {
        file_name: fileName,
      },
    },
  };

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Batch creation failed (${res.status}): ${err}`);
  }

  return res.json();
}

/**
 * Get the current status of a batch job.
 *
 * @param {string} batchName - e.g. "batches/abc123"
 * @param {string} apiKey
 * @returns {Promise<object>}
 */
export async function getBatchStatus(batchName, apiKey) {
  const url = `${GEMINI_API}/v1beta/${batchName}?key=${apiKey}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Batch status check failed (${res.status}): ${err}`);
  }

  return res.json();
}

/**
 * List all batch jobs.
 *
 * @param {string} apiKey
 * @returns {Promise<object>}
 */
export async function listBatchJobs(apiKey) {
  const url = `${BATCHES_URL}?key=${apiKey}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Batch list failed (${res.status}): ${err}`);
  }

  return res.json();
}

// ---------------------------------------------------------------------------
// JSONL builder
// ---------------------------------------------------------------------------

/**
 * Build a JSONL file from an array of prompt objects and save it to disk.
 *
 * @param {Array<{key: string, prompt: string, aspect_ratio?: string, image_paths?: string}>} requests
 * @param {string} model
 * @returns {string} - Path to the saved JSONL file
 */
export function buildJsonlFile(requests, model) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const jsonlPath = path.join(DATA_DIR, `batch_input_${timestamp}.jsonl`);

  const lines = requests.map(req => {
    // Build the parts array
    const parts = [];

    // Add reference images if provided
    if (req.image_paths) {
      const paths = req.image_paths.split(",").map(p => p.trim()).filter(Boolean);
      for (const p of paths) {
        if (fs.existsSync(p)) {
          const base64 = fs.readFileSync(p).toString("base64");
          const ext = path.extname(p).toLowerCase();
          let mime_type = "image/png";
          if (ext === ".jpg" || ext === ".jpeg") mime_type = "image/jpeg";
          else if (ext === ".webp") mime_type = "image/webp";
          else if (ext === ".gif") mime_type = "image/gif";
          parts.push({ inline_data: { mime_type, data: base64 } });
        }
      }
    }

    // Add the text prompt
    parts.push({ text: req.prompt });

    // Build generation_config (snake_case for Gemini REST/batch API)
    const generation_config = {
      response_modalities: ["TEXT", "IMAGE"],
    };
    if (req.aspect_ratio) {
      generation_config.image_config = { aspect_ratio: req.aspect_ratio };
    }

    // Note: model is NOT included here — it's already specified in the batch URL
    const line = {
      key: req.key,
      request: {
        contents: [{ parts }],
        generation_config,
      },
    };

    return JSON.stringify(line);
  });

  fs.writeFileSync(jsonlPath, lines.join("\n") + "\n");
  return jsonlPath;
}

// ---------------------------------------------------------------------------
// Result processor
// ---------------------------------------------------------------------------

/**
 * Parse batch output JSONL and save images to the output directory.
 *
 * @param {string} outputJsonl - Raw JSONL output string from Gemini
 * @param {string} outputDir   - Directory to save images into
 * @returns {Array<{key: string, imagePath: string|null, text: string|null, error: string|null}>}
 */
export function processBatchResults(outputJsonl, outputDir) {
  fs.mkdirSync(outputDir, { recursive: true });
  const results = [];

  for (const line of outputJsonl.split("\n").filter(Boolean)) {
    let parsed;
    try {
      parsed = JSON.parse(line);
    } catch {
      continue;
    }

    const key = parsed.key ?? "unknown";
    const response = parsed.response;

    if (!response || !response.candidates) {
      results.push({ key, imagePath: null, text: null, error: "No response candidates" });
      continue;
    }

    let imagePath = null;
    let text = null;

    for (const candidate of response.candidates) {
      for (const part of candidate?.content?.parts ?? []) {
        // Handle both camelCase and snake_case from Google's response
        const imgData = part.inlineData || part.inline_data;
        if (imgData) {
          const mimeType = imgData.mimeType || imgData.mime_type || "image/jpeg";
          const ext = mimeType.includes("png") ? ".png"
            : mimeType.includes("webp") ? ".webp"
            : mimeType.includes("gif") ? ".gif"
            : ".jpg";

          const fileName = `${key}${ext}`;
          imagePath = path.join(outputDir, fileName);
          fs.writeFileSync(imagePath, Buffer.from(imgData.data, "base64"));
        }
        if (part.text) {
          text = text ? text + "\n" + part.text : part.text;
        }
      }
    }

    results.push({ key, imagePath, text, error: null });
  }

  return results;
}
