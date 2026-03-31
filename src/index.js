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
  uploadFile, downloadFile,
  createBatchJob, getBatchStatus,
  buildJsonlFile, processBatchResults,
} from "./batch.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const DEFAULT_MODEL = "gemini-3.1-flash-image-preview";
const SUPPORTED_MODELS = [
  "gemini-3.1-flash-image-preview",
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
  version: "2.0.0",
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

// ---- Tool: upload_image -----------------------------------------------------

server.registerTool(
  "gemini_upload_image",
  {
    title: "Upload Image",
    description:
      `Upload one or more images to the Gemini Files API and return their file URIs.

Use this to pre-upload reference images before calling gemini_batch_submit.
Uploaded files persist for 48 hours on Google's servers. Pass the returned URIs
to batch_submit via the file_uris field to avoid slow base64 encoding.

Args:
  - image_paths (string, required): Comma-separated list of local file paths to upload.
    Example: "/path/to/ref1.jpg,/path/to/ref2.png"

Returns:
  - A list of file URIs (e.g. "files/abc123") mapped to each input path.`,

    inputSchema: {
      image_paths: z.string()
        .describe("Comma-separated list of local file paths to upload"),
    },

    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },

  async ({ image_paths }) => {
    try {
      const apiKey = getApiKey();
      const paths = image_paths.split(",").map(p => p.trim()).filter(Boolean);

      if (paths.length === 0) {
        return {
          isError: true,
          content: [{ type: "text", text: "Error: No image paths provided." }],
        };
      }

      const results = [];
      for (const filePath of paths) {
        if (!fs.existsSync(filePath)) {
          results.push({ path: filePath, uri: null, error: "File not found" });
          continue;
        }

        try {
          const mimeType = detectMimeType(filePath);
          const uri = await uploadFile(filePath, apiKey, mimeType);
          results.push({ path: filePath, uri, error: null });
        } catch (err) {
          results.push({ path: filePath, uri: null, error: err.message });
        }
      }

      const succeeded = results.filter(r => r.uri);
      const failed = results.filter(r => r.error);

      const lines = [
        `Uploaded ${succeeded.length}/${results.length} images:`,
      ];

      for (const r of succeeded) {
        lines.push(`  ${path.basename(r.path)} → ${r.uri}`);
      }
      if (failed.length > 0) {
        lines.push(`Errors:`);
        for (const r of failed) {
          lines.push(`  ${r.path}: ${r.error}`);
        }
      }

      // Return just the URIs as a comma-separated string for easy passing to batch_submit
      if (succeeded.length > 0) {
        lines.push(``);
        lines.push(`file_uris: ${succeeded.map(r => r.uri).join(",")}`);
      }

      return {
        content: [{ type: "text", text: lines.join("\n") }],
      };
    } catch (error) {
      return {
        isError: true,
        content: [{ type: "text", text: `Error uploading images: ${error.message}` }],
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

Builds a JSONL file, uploads it to the Gemini Files API, and submits a batch job.
The job processes asynchronously (usually completes within minutes).
Returns the batch ID — the caller is responsible for tracking it (e.g. in Supabase).

Args:
  - requests (string, required): JSON array of request objects. Each object must have:
      - key (string): Unique identifier for this request (used as output filename)
      - prompt (string): The image generation prompt
      - aspect_ratio (string, optional): Aspect ratio for this image
      - image_paths (string, optional): Comma-separated input image paths for editing (slow — base64 encodes each)
    Example: [{"key":"sunset-cat","prompt":"A cat watching a sunset","aspect_ratio":"16:9"}]
  - file_uris (string, optional): Comma-separated Gemini Files API URIs of pre-uploaded reference images
    (e.g. "files/abc123,files/def456"). These are shared across ALL requests in the batch.
    Use gemini_upload_image first to get URIs. Much faster than image_paths for large batches.
  - model (string, optional): Gemini model ID. Defaults to "${DEFAULT_MODEL}".
  - image_size (string, optional): Output image resolution. Values: "1K" (1024px), "2K" (2048px), "4K" (4096px). Defaults to "2K".
  - display_name (string, optional): Human-readable name for the batch job.

Returns:
  - The batch name/ID, request count, and model used.`,

    inputSchema: {
      requests: z.string()
        .describe('JSON array of {key, prompt, aspect_ratio?, image_paths?} objects'),
      file_uris: z.string()
        .optional()
        .describe('Comma-separated Gemini file URIs of pre-uploaded reference images shared across all requests (e.g. "files/abc,files/def")'),
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

  async ({ requests, file_uris, model, image_size, display_name }) => {
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

      // 1. Build JSONL file (temp file, cleaned up after upload)
      // Parse shared file URIs if provided
      const sharedFileUris = file_uris
        ? file_uris.split(",").map(u => u.trim()).filter(Boolean)
        : [];

      const jsonlPath = buildJsonlFile(parsedRequests, useModel, image_size || "2K", sharedFileUris);

      // 2. Upload to Gemini Files API
      const fileName = await uploadFile(jsonlPath, apiKey);

      // 3. Clean up temp JSONL file
      try { fs.unlinkSync(jsonlPath); } catch {}

      // 4. Create batch job
      const batchJob = await createBatchJob(useModel, fileName, apiKey, jobDisplayName);
      const batchName = batchJob.name || batchJob.metadata?.name;

      return {
        content: [{
          type: "text",
          text: [
            `Batch job submitted successfully!`,
            ``,
            `  Batch ID:      ${batchName}`,
            `  Display Name:  ${jobDisplayName}`,
            `  Requests:      ${parsedRequests.length}`,
            `  Keys:          ${parsedRequests.map(r => r.key).join(", ")}`,
            `  Model:         ${useModel}`,
            `  Image Size:    ${image_size || "2K"}`,
            `  Input File:    ${fileName}`,
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
      `Check the status of a batch image generation job.

Args:
  - batch_name (string, required): The batch ID to check (e.g. "batches/abc123").

Returns:
  - state: BATCH_STATE_PENDING, BATCH_STATE_RUNNING, BATCH_STATE_SUCCEEDED, BATCH_STATE_FAILED, etc.
  - output_file: The Gemini Files API reference for downloading results (when succeeded).
  - stats: Request count and success count.
  - Timing: create time, end time.`,

    inputSchema: {
      batch_name: z.string()
        .describe('The batch ID to check (e.g. "batches/abc123")'),
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
      const status = await getBatchStatus(batch_name, apiKey);

      // Google returns state and details inside metadata
      const meta = status.metadata || {};
      const state = meta.state || status.state || "UNKNOWN";
      const outputFile = status.response?.responsesFile
        || meta.output?.responsesFile || null;
      const stats = meta.batchStats || null;

      const lines = [
        `Batch:     ${batch_name}`,
        `State:     ${state}`,
        `Model:     ${meta.model || "unknown"}`,
        meta.displayName ? `Name:      ${meta.displayName}` : "",
        stats ? `Requests:  ${stats.successfulRequestCount || 0}/${stats.requestCount || "?"}` : "",
        outputFile ? `Output:    ${outputFile}` : "",
        meta.createTime ? `Created:   ${meta.createTime}` : "",
        meta.endTime ? `Completed: ${meta.endTime}` : "",
        status.error ? `Error:     ${JSON.stringify(status.error)}` : "",
      ];

      return {
        content: [{
          type: "text",
          text: lines.filter(Boolean).join("\n"),
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

Checks the batch status, retrieves the output JSONL from Gemini, decodes each image,
and saves them to the specified output directory using the key as the filename.

Args:
  - batch_name (string, required): The batch ID (e.g. "batches/abc123").
  - output_dir (string, required): Directory where images will be saved.

Returns:
  - List of saved image paths (key → file path) and any errors.`,

    inputSchema: {
      batch_name: z.string()
        .describe('The batch ID (e.g. "batches/abc123")'),
      output_dir: z.string()
        .describe("Directory where images will be saved"),
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

      // Check batch status first
      const status = await getBatchStatus(batch_name, apiKey);
      const meta = status.metadata || {};
      const state = meta.state || status.state;

      if (!state?.includes("SUCCEEDED")) {
        return {
          content: [{
            type: "text",
            text: `Batch ${batch_name} is not complete yet (state: ${state}).`,
          }],
        };
      }

      // Get the output file reference
      const outputFile = status.response?.responsesFile
        || meta.output?.responsesFile;

      if (!outputFile) {
        return {
          isError: true,
          content: [{ type: "text", text: `Batch ${batch_name} succeeded but no output file found.` }],
        };
      }

      // Download output JSONL from Gemini Files API
      const outputContent = await downloadFile(outputFile, apiKey);

      // Process results and save images to output_dir
      const results = processBatchResults(outputContent, output_dir);

      const succeeded = results.filter(r => r.imagePath);
      const failed = results.filter(r => r.error);

      const resultLines = [
        `Batch ${batch_name} results:`,
        `  Images saved: ${succeeded.length}/${results.length}`,
        `  Output dir:   ${output_dir}`,
      ];

      if (succeeded.length > 0) {
        resultLines.push(`  Files:`);
        for (const r of succeeded) {
          resultLines.push(`    - ${r.key}: ${r.imagePath}`);
        }
      }
      if (failed.length > 0) {
        resultLines.push(`  Errors:`);
        for (const r of failed) {
          resultLines.push(`    - ${r.key}: ${r.error}`);
        }
      }

      return {
        content: [{ type: "text", text: resultLines.join("\n") }],
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
