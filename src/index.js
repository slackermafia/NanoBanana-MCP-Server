#!/usr/bin/env node

/**
 * NanoBanana MCP Server
 *
 * An MCP server that connects to the Google Gemini API to generate
 * and edit images using the Nano Banana Pro image generation model.
 *
 * Supports:
 *   - Text-to-image generation
 *   - Image editing (single or multiple input images + text prompt)
 *   - Aspect ratio control
 *
 * Requires GEMINI_API_KEY environment variable.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import fs from "fs";
import path from "path";
import {
  loadJobs, saveJobs, trackJob, updateJob,
  uploadFile, downloadFile,
  createBatchJob, getBatchStatus, listBatchJobs,
  buildJsonlFile, processBatchResults,
} from "./batch.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const DEFAULT_MODEL = "gemini-3-pro-image-preview";
const SUPPORTED_MODELS = [
  "gemini-3-pro-image-preview",
];

const SUPPORTED_MIME_TYPES = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
];

const SUPPORTED_ASPECT_RATIOS = [
  "1:1", "3:2", "2:3", "3:4", "4:3",
  "4:5", "5:4", "9:16", "16:9", "21:9",
];

// ---------------------------------------------------------------------------
// Gemini API helpers
// ---------------------------------------------------------------------------

function getApiKey() {
  const key = process.env.GEMINI_API_KEY;
  if (!key) {
    throw new Error(
      "GEMINI_API_KEY environment variable is not set. " +
      "Get an API key at https://aistudio.google.com/apikey"
    );
  }
  return key;
}

/**
 * Detect MIME type from file extension.
 */
function detectMimeType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  if (ext === ".webp") return "image/webp";
  if (ext === ".gif") return "image/gif";
  return "image/png";
}

/**
 * Load a local image file and return an inlineData part for the Gemini API.
 */
function loadImagePart(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }
  const base64 = fs.readFileSync(filePath).toString("base64");
  const mimeType = detectMimeType(filePath);
  return { inlineData: { mimeType, data: base64 } };
}

/**
 * Call the Gemini generateContent endpoint.
 *
 * @param {object} options
 * @param {string} options.model              - Model ID to use
 * @param {Array}  options.parts              - Content parts (text / inlineData)
 * @param {Array}  options.responseModalities - e.g. ["TEXT","IMAGE"]
 * @param {string} [options.aspectRatio]      - e.g. "16:9"
 * @returns {Promise<object>}                 - Raw JSON response from Gemini
 */
async function callGemini({ model, parts, responseModalities, aspectRatio, imageSize }) {
  const apiKey = getApiKey();
  const url = `${GEMINI_API_BASE}/${model}:generateContent?key=${apiKey}`;

  const generationConfig = {
    responseModalities: responseModalities ?? ["TEXT", "IMAGE"],
  };

  const imageConfig = {};
  if (aspectRatio) imageConfig.aspectRatio = aspectRatio;
  if (imageSize) imageConfig.imageSize = imageSize;
  if (Object.keys(imageConfig).length > 0) {
    generationConfig.imageConfig = imageConfig;
  }

  const body = {
    contents: [{ parts }],
    generationConfig,
  };

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errBody = await res.text();
    throw new Error(`Gemini API error (${res.status}): ${errBody}`);
  }

  return res.json();
}

/**
 * Extract image and text parts from a Gemini generateContent response.
 *
 * @param {object} response - Raw Gemini API response
 * @returns {{ images: Array<{mimeType:string, base64:string}>, text: string|null }}
 */
function parseGeminiResponse(response) {
  const images = [];
  let text = null;

  const candidates = response?.candidates ?? [];
  for (const candidate of candidates) {
    const parts = candidate?.content?.parts ?? [];
    for (const part of parts) {
      if (part.inlineData) {
        images.push({
          mimeType: part.inlineData.mimeType,
          base64: part.inlineData.data,
        });
      }
      if (part.text) {
        text = text ? text + "\n" + part.text : part.text;
      }
    }
  }

  return { images, text };
}

/**
 * Build MCP content blocks from parsed Gemini response.
 */
function buildMcpContent(parsed, outputPath) {
  const content = [];

  if (parsed.text) {
    content.push({ type: "text", text: parsed.text });
  }

  if (parsed.images.length === 0) {
    if (content.length === 0) {
      content.push({ type: "text", text: "No image was generated. Try rephrasing your prompt." });
    }
    return content;
  }

  for (let i = 0; i < parsed.images.length; i++) {
    const img = parsed.images[i];

    // If an output path was provided, save the file and report path
    if (outputPath) {
      const ext = img.mimeType === "image/png" ? ".png"
        : img.mimeType === "image/webp" ? ".webp"
        : img.mimeType === "image/gif" ? ".gif"
        : ".jpg";

      const filePath = parsed.images.length === 1
        ? outputPath.replace(/\.[^.]+$/, "") + ext
        : outputPath.replace(/\.[^.]+$/, "") + `_${i + 1}` + ext;

      fs.mkdirSync(path.dirname(filePath), { recursive: true });
      fs.writeFileSync(filePath, Buffer.from(img.base64, "base64"));

      content.push({
        type: "text",
        text: `Image saved to: ${filePath}`,
      });
    }

    // Always return the image as an embedded image content block
    content.push({
      type: "image",
      data: img.base64,
      mimeType: img.mimeType,
    });
  }

  return content;
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

const server = new McpServer({
  name: "nanobanana-mcp-server",
  version: "1.2.0",
});

// ---- Tool: generate_image ----------------------------------------------------

server.registerTool(
  "gemini_generate_image",
  {
    title: "Generate Image",
    description:
      `Generate an image from a text prompt using the Google Gemini Nano Banana image generation model.

Args:
  - prompt (string, required): A detailed description of the image to generate.
  - aspect_ratio (string, optional): Aspect ratio for the output image.
    Supported: ${SUPPORTED_ASPECT_RATIOS.join(", ")}
  - model (string, optional): Gemini model ID to use. Defaults to "${DEFAULT_MODEL}".
  - output_path (string, optional): File path to save the generated image.

Returns:
  - The generated image as an embedded image block (and optionally saved to disk).
  - Any text the model returns alongside the image.

Examples:
  - "A photorealistic golden retriever surfing a wave at sunset"
  - "An isometric pixel-art castle on a floating island"`,

    inputSchema: {
      prompt: z.string()
        .min(1, "Prompt must not be empty")
        .describe("A detailed text description of the image to generate"),
      aspect_ratio: z.string()
        .optional()
        .describe(`Aspect ratio for the output image. Supported: ${SUPPORTED_ASPECT_RATIOS.join(", ")}`),
      model: z.string()
        .default(DEFAULT_MODEL)
        .describe(`Gemini model ID (default: ${DEFAULT_MODEL})`),
      image_size: z.string()
        .optional()
        .describe('Output image resolution: "1K" (1024px), "2K" (2048px), or "4K" (4096px)'),
      output_path: z.string()
        .optional()
        .describe("Optional file path to save the generated image"),
    },

    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },

  async ({ prompt, aspect_ratio, model, image_size, output_path }) => {
    try {
      if (aspect_ratio && !SUPPORTED_ASPECT_RATIOS.includes(aspect_ratio)) {
        return {
          isError: true,
          content: [{
            type: "text",
            text: `Error: Unsupported aspect ratio "${aspect_ratio}". Supported: ${SUPPORTED_ASPECT_RATIOS.join(", ")}`,
          }],
        };
      }

      const parts = [{ text: prompt }];
      const response = await callGemini({
        model: model || DEFAULT_MODEL,
        parts,
        responseModalities: ["TEXT", "IMAGE"],
        aspectRatio: aspect_ratio,
        imageSize: image_size,
      });

      const parsed = parseGeminiResponse(response);
      return { content: buildMcpContent(parsed, output_path) };
    } catch (error) {
      return {
        isError: true,
        content: [{ type: "text", text: `Error generating image: ${error.message}` }],
      };
    }
  }
);

// ---- Tool: edit_image --------------------------------------------------------

server.registerTool(
  "gemini_edit_image",
  {
    title: "Edit Image",
    description:
      `Edit or transform one or more images using a text instruction, powered by the Google Gemini Nano Banana image model.

Supports single or multiple input images. Provide file paths and/or base64-encoded image data along with a text instruction.

Args:
  - prompt (string, required): A text instruction describing the edit (e.g. "Remove the background", "Combine these two images").
  - image_paths (string, optional): Comma-separated list of file paths to input images on disk.
    Example: "/path/to/img1.png,/path/to/img2.jpg"
  - image_base64_list (string, optional): JSON array of objects with "data" and "mimeType" fields for base64-encoded images.
    Example: [{"data":"base64...","mimeType":"image/png"}]
  - aspect_ratio (string, optional): Aspect ratio for the output image.
    Supported: ${SUPPORTED_ASPECT_RATIOS.join(", ")}
  - model (string, optional): Gemini model ID. Defaults to "${DEFAULT_MODEL}".
  - output_path (string, optional): File path to save the edited image.

Returns:
  - The edited image as an embedded image block (and optionally saved to disk).
  - Any text the model returns alongside the image.

Examples:
  - prompt: "Change the car color to red" with one image
  - prompt: "Combine these two photos into a collage" with two images
  - prompt: "Apply the style of the first image to the second image" with two images`,

    inputSchema: {
      prompt: z.string()
        .min(1, "Prompt must not be empty")
        .describe("Text instruction describing the desired image edit"),
      image_paths: z.string()
        .optional()
        .describe("Comma-separated list of file paths to input images (e.g. '/path/img1.png,/path/img2.jpg')"),
      image_base64_list: z.string()
        .optional()
        .describe('JSON array of objects with "data" and "mimeType" fields for base64-encoded images'),
      aspect_ratio: z.string()
        .optional()
        .describe(`Aspect ratio for the output image. Supported: ${SUPPORTED_ASPECT_RATIOS.join(", ")}`),
      model: z.string()
        .default(DEFAULT_MODEL)
        .describe(`Gemini model ID (default: ${DEFAULT_MODEL})`),
      image_size: z.string()
        .optional()
        .describe('Output image resolution: "1K" (1024px), "2K" (2048px), or "4K" (4096px)'),
      output_path: z.string()
        .optional()
        .describe("Optional file path to save the edited image"),
    },

    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },

  async ({ prompt, image_paths, image_base64_list, aspect_ratio, model, image_size, output_path }) => {
    try {
      const imageParts = [];

      // Load images from file paths
      if (image_paths) {
        const paths = image_paths.split(",").map(p => p.trim()).filter(Boolean);
        for (const p of paths) {
          imageParts.push(loadImagePart(p));
        }
      }

      // Load images from base64 JSON array
      if (image_base64_list) {
        let parsed;
        try {
          parsed = JSON.parse(image_base64_list);
        } catch {
          return {
            isError: true,
            content: [{
              type: "text",
              text: "Error: image_base64_list must be a valid JSON array of {data, mimeType} objects.",
            }],
          };
        }

        if (!Array.isArray(parsed)) {
          return {
            isError: true,
            content: [{
              type: "text",
              text: "Error: image_base64_list must be a JSON array.",
            }],
          };
        }

        for (const item of parsed) {
          if (!item.data || !item.mimeType) {
            return {
              isError: true,
              content: [{
                type: "text",
                text: 'Error: Each item in image_base64_list must have "data" and "mimeType" fields.',
              }],
            };
          }
          if (!SUPPORTED_MIME_TYPES.includes(item.mimeType)) {
            return {
              isError: true,
              content: [{
                type: "text",
                text: `Error: Unsupported MIME type "${item.mimeType}". Supported: ${SUPPORTED_MIME_TYPES.join(", ")}`,
              }],
            };
          }
          imageParts.push({ inlineData: { mimeType: item.mimeType, data: item.data } });
        }
      }

      if (imageParts.length === 0) {
        return {
          isError: true,
          content: [{
            type: "text",
            text: "Error: You must provide at least one image via image_paths or image_base64_list.",
          }],
        };
      }

      if (aspect_ratio && !SUPPORTED_ASPECT_RATIOS.includes(aspect_ratio)) {
        return {
          isError: true,
          content: [{
            type: "text",
            text: `Error: Unsupported aspect ratio "${aspect_ratio}". Supported: ${SUPPORTED_ASPECT_RATIOS.join(", ")}`,
          }],
        };
      }

      // Build parts: all images first, then the text prompt
      const parts = [...imageParts, { text: prompt }];

      const response = await callGemini({
        model: model || DEFAULT_MODEL,
        parts,
        responseModalities: ["TEXT", "IMAGE"],
        aspectRatio: aspect_ratio,
        imageSize: image_size,
      });

      const parsedResponse = parseGeminiResponse(response);
      return { content: buildMcpContent(parsedResponse, output_path) };
    } catch (error) {
      return {
        isError: true,
        content: [{ type: "text", text: `Error editing image: ${error.message}` }],
      };
    }
  }
);

// ---- Tool: batch_submit -----------------------------------------------------

server.registerTool(
  "gemini_batch_submit",
  {
    title: "Batch Submit",
    description:
      `Submit a batch of image generation requests to the Gemini API at 50% reduced cost.

Creates a JSONL input file (saved locally for debugging), uploads it to the Gemini Files API,
and submits a batch job. The job processes asynchronously (usually completes within 24 hours).
Use gemini_batch_status to check progress and gemini_batch_results to retrieve images.

Args:
  - requests (string, required): JSON array of request objects. Each object must have:
      - key (string): Unique identifier for this request (used as output filename)
      - prompt (string): The image generation prompt
      - aspect_ratio (string, optional): Aspect ratio for this image
      - image_paths (string, optional): Comma-separated input image paths for editing
    Example: [{"key":"sunset-cat","prompt":"A cat watching a sunset","aspect_ratio":"16:9"}]
  - output_dir (string, required): Directory where completed images will be saved.
  - model (string, optional): Gemini model ID. Defaults to "${DEFAULT_MODEL}".
  - image_size (string, optional): Output image resolution. Values: "1K" (1024px), "2K" (2048px), "4K" (4096px). Defaults to "2K".
  - display_name (string, optional): Human-readable name for the batch job.

Returns:
  - The batch job name/ID for tracking, plus the JSONL file path for debugging.`,

    inputSchema: {
      requests: z.string()
        .describe('JSON array of {key, prompt, aspect_ratio?, image_paths?} objects'),
      output_dir: z.string()
        .describe("Directory where completed images will be saved"),
      model: z.string()
        .default(DEFAULT_MODEL)
        .describe(`Gemini model ID (default: ${DEFAULT_MODEL})`),
      image_size: z.string()
        .default("2K")
        .describe('Output image resolution: "1K" (1024px), "2K" (2048px), or "4K" (4096px). Default: "2K"'),
      display_name: z.string()
        .optional()
        .describe("Human-readable name for the batch job"),
    },

    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },

  async ({ requests, output_dir, model, image_size, display_name }) => {
    try {
      const apiKey = getApiKey();

      // Parse requests
      let parsedRequests;
      try {
        parsedRequests = JSON.parse(requests);
      } catch {
        return {
          isError: true,
          content: [{ type: "text", text: "Error: requests must be a valid JSON array." }],
        };
      }

      if (!Array.isArray(parsedRequests) || parsedRequests.length === 0) {
        return {
          isError: true,
          content: [{ type: "text", text: "Error: requests must be a non-empty JSON array." }],
        };
      }

      // Validate each request has key and prompt
      for (const req of parsedRequests) {
        if (!req.key || !req.prompt) {
          return {
            isError: true,
            content: [{ type: "text", text: `Error: Each request must have "key" and "prompt". Got: ${JSON.stringify(req)}` }],
          };
        }
      }

      const useModel = model || DEFAULT_MODEL;
      const jobDisplayName = display_name || `nanobanana-batch-${Date.now()}`;

      // 1. Build JSONL file
      const jsonlPath = buildJsonlFile(parsedRequests, useModel, image_size || "2K");

      // 2. Upload to Gemini Files API
      const fileName = await uploadFile(jsonlPath, apiKey);

      // 3. Create batch job
      const batchJob = await createBatchJob(useModel, fileName, apiKey, jobDisplayName);
      const batchName = batchJob.name;

      // 4. Track the job locally
      trackJob({
        batchName,
        displayName: jobDisplayName,
        model: useModel,
        state: batchJob.state || "JOB_STATE_PENDING",
        inputFile: fileName,
        jsonlPath,
        outputDir: output_dir,
        requestCount: parsedRequests.length,
        requestKeys: parsedRequests.map(r => r.key),
        createdAt: new Date().toISOString(),
        completedAt: null,
        outputFile: null,
        error: null,
      });

      return {
        content: [{
          type: "text",
          text: [
            `Batch job submitted successfully!`,
            ``,
            `  Batch ID:    ${batchName}`,
            `  Display Name: ${jobDisplayName}`,
            `  Requests:    ${parsedRequests.length}`,
            `  Model:       ${useModel}`,
            `  JSONL File:  ${jsonlPath}`,
            `  Output Dir:  ${output_dir}`,
            ``,
            `Use gemini_batch_status to check progress.`,
            `Use gemini_batch_results to download images when complete.`,
          ].join("\n"),
        }],
      };
    } catch (error) {
      return {
        isError: true,
        content: [{ type: "text", text: `Error submitting batch: ${error.message}` }],
      };
    }
  }
);

// ---- Tool: batch_status -----------------------------------------------------

server.registerTool(
  "gemini_batch_status",
  {
    title: "Batch Status",
    description:
      `Check the status of one or all pending batch image generation jobs.

Args:
  - batch_name (string, optional): Specific batch ID to check (e.g. "batches/abc123").
    If omitted, checks ALL tracked jobs and updates their status.

Returns:
  - Current state of each job (PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED).
  - For completed jobs, includes the output file reference.`,

    inputSchema: {
      batch_name: z.string()
        .optional()
        .describe('Specific batch ID to check (e.g. "batches/abc123"). Omit to check all.'),
    },

    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
  },

  async ({ batch_name }) => {
    try {
      const apiKey = getApiKey();
      const jobs = loadJobs();

      if (jobs.length === 0) {
        return {
          content: [{ type: "text", text: "No batch jobs are being tracked." }],
        };
      }

      // Filter to specific job or all pending jobs
      const toCheck = batch_name
        ? jobs.filter(j => j.batchName === batch_name)
        : jobs.filter(j => !["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"].includes(j.state));

      if (toCheck.length === 0 && batch_name) {
        return {
          content: [{ type: "text", text: `No tracked job found with name: ${batch_name}` }],
        };
      }

      const statusLines = [];

      for (const job of toCheck) {
        try {
          const status = await getBatchStatus(job.batchName, apiKey);
          const newState = status.state || job.state;

          const updates = { state: newState };

          const destFile = status.dest?.fileName || status.dest?.file_name
            || status.output_config?.destination?.file_name
            || status.outputConfig?.destination?.fileName;
          if (destFile) {
            updates.outputFile = destFile;
          }
          if (newState === "JOB_STATE_SUCCEEDED") {
            updates.completedAt = new Date().toISOString();
          }
          if (status.error) {
            updates.error = JSON.stringify(status.error);
          }

          updateJob(job.batchName, updates);

          statusLines.push(
            `${job.batchName}`,
            `  Name:      ${job.displayName}`,
            `  State:     ${newState}`,
            `  Requests:  ${job.requestCount}`,
            `  Created:   ${job.createdAt}`,
            destFile ? `  Output:    ${destFile}` : "",
            ""
          );
        } catch (err) {
          statusLines.push(`${job.batchName}: Error checking status - ${err.message}`, "");
        }
      }

      // Also show summary of all jobs
      const allJobs = loadJobs();
      const summary = [
        `--- All Tracked Jobs ---`,
        `Pending:   ${allJobs.filter(j => j.state === "JOB_STATE_PENDING").length}`,
        `Running:   ${allJobs.filter(j => j.state === "JOB_STATE_RUNNING").length}`,
        `Succeeded: ${allJobs.filter(j => j.state === "JOB_STATE_SUCCEEDED").length}`,
        `Failed:    ${allJobs.filter(j => j.state === "JOB_STATE_FAILED").length}`,
      ];

      return {
        content: [{
          type: "text",
          text: [...statusLines, ...summary].filter(Boolean).join("\n"),
        }],
      };
    } catch (error) {
      return {
        isError: true,
        content: [{ type: "text", text: `Error checking batch status: ${error.message}` }],
      };
    }
  }
);

// ---- Tool: batch_results ----------------------------------------------------

server.registerTool(
  "gemini_batch_results",
  {
    title: "Batch Results",
    description:
      `Download and save images from a completed batch job.

Retrieves the output JSONL from the Gemini Files API, decodes each image,
and saves them to the output directory that was specified when the batch was submitted
(or to a different directory if output_dir is provided here).

Args:
  - batch_name (string, optional): Specific batch ID to retrieve. If omitted,
    retrieves results for ALL completed jobs that haven't been downloaded yet.
  - output_dir (string, optional): Override the output directory for saving images.
    If omitted, uses the output_dir from when the batch was submitted.

Returns:
  - List of saved image paths and any errors encountered.`,

    inputSchema: {
      batch_name: z.string()
        .optional()
        .describe('Specific batch ID to retrieve. Omit to process all completed jobs.'),
      output_dir: z.string()
        .optional()
        .describe("Override output directory for saving images"),
    },

    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },

  async ({ batch_name, output_dir }) => {
    try {
      const apiKey = getApiKey();
      const jobs = loadJobs();

      // Find completed jobs that need results downloaded
      const toProcess = batch_name
        ? jobs.filter(j => j.batchName === batch_name)
        : jobs.filter(j => j.state === "JOB_STATE_SUCCEEDED" && !j.resultsDownloaded);

      if (toProcess.length === 0) {
        return {
          content: [{
            type: "text",
            text: batch_name
              ? `No job found with name: ${batch_name}`
              : "No completed jobs with pending results to download.",
          }],
        };
      }

      const allResults = [];

      for (const job of toProcess) {
        // If the job isn't done yet, check its status first
        if (job.state !== "JOB_STATE_SUCCEEDED") {
          const status = await getBatchStatus(job.batchName, apiKey);
          if (status.state !== "JOB_STATE_SUCCEEDED") {
            allResults.push(`${job.batchName}: Not yet complete (state: ${status.state})`);
            continue;
          }
          // Update to succeeded
          const destFile = status.dest?.fileName || status.dest?.file_name
            || status.output_config?.destination?.file_name
            || status.outputConfig?.destination?.fileName;
          updateJob(job.batchName, {
            state: "JOB_STATE_SUCCEEDED",
            completedAt: new Date().toISOString(),
            outputFile: destFile || job.outputFile,
          });
          job.outputFile = destFile || job.outputFile;
        }

        if (!job.outputFile) {
          allResults.push(`${job.batchName}: No output file reference found.`);
          continue;
        }

        const saveDir = output_dir || job.outputDir;
        if (!saveDir) {
          allResults.push(`${job.batchName}: No output directory specified.`);
          continue;
        }

        try {
          // Download output JSONL
          const outputContent = await downloadFile(job.outputFile, apiKey);

          // Save the raw output JSONL for debugging
          const outputJsonlPath = job.jsonlPath.replace("_input_", "_output_");
          fs.writeFileSync(outputJsonlPath, outputContent);

          // Process results and save images
          const results = processBatchResults(outputContent, saveDir);

          // Mark as downloaded
          updateJob(job.batchName, { resultsDownloaded: true });

          const succeeded = results.filter(r => r.imagePath);
          const failed = results.filter(r => r.error);

          allResults.push(
            `${job.batchName} (${job.displayName}):`,
            `  Images saved: ${succeeded.length}/${results.length}`,
            `  Output dir:   ${saveDir}`,
            `  Output JSONL: ${outputJsonlPath}`,
          );

          if (succeeded.length > 0) {
            allResults.push(`  Files:`);
            for (const r of succeeded) {
              allResults.push(`    - ${r.key}: ${r.imagePath}`);
            }
          }
          if (failed.length > 0) {
            allResults.push(`  Errors:`);
            for (const r of failed) {
              allResults.push(`    - ${r.key}: ${r.error}`);
            }
          }
          allResults.push("");
        } catch (err) {
          allResults.push(`${job.batchName}: Error downloading results - ${err.message}`);
        }
      }

      return {
        content: [{ type: "text", text: allResults.join("\n") }],
      };
    } catch (error) {
      return {
        isError: true,
        content: [{ type: "text", text: `Error retrieving batch results: ${error.message}` }],
      };
    }
  }
);

// ---------------------------------------------------------------------------
// Start the server via stdio
// ---------------------------------------------------------------------------

async function main() {
  try {
    getApiKey(); // Validate API key is set before starting
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("NanoBanana MCP Server running via stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
