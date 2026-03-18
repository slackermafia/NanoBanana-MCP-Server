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
async function callGemini({ model, parts, responseModalities, aspectRatio }) {
  const apiKey = getApiKey();
  const url = `${GEMINI_API_BASE}/${model}:generateContent?key=${apiKey}`;

  const generationConfig = {
    responseModalities: responseModalities ?? ["TEXT", "IMAGE"],
  };

  if (aspectRatio) {
    generationConfig.imageConfig = { aspectRatio };
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
  version: "1.1.0",
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

  async ({ prompt, aspect_ratio, model, output_path }) => {
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

  async ({ prompt, image_paths, image_base64_list, aspect_ratio, model, output_path }) => {
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
