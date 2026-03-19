# NanoBanana MCP Server

An MCP (Model Context Protocol) server that connects to the Google Gemini API to generate and edit images using the **Nano Banana Pro** image generation model.

## Features

- **Text-to-Image Generation** — Describe an image and get it generated via the Gemini API.
- **Image Editing** — Provide one or more existing images and a text instruction to edit or transform them.
- **Multi-Image Input** — Send multiple images for blending, style transfer, collages, and more.
- **Batch Mode** — Submit many prompts at once at **50% reduced cost**. Jobs run async and results are polled/downloaded automatically.
- **Aspect Ratio Control** — Force output to a specific aspect ratio (1:1, 16:9, 9:16, etc.).
- **File Output** — Save generated images directly to disk with key-based filenames.
- **Job Tracking** — Batch jobs are persisted to `data/batch_jobs.json` with full state, input JSONL, and output references.

## Prerequisites

- **Node.js** >= 18
- A **Google Gemini API key** — get one at [Google AI Studio](https://aistudio.google.com/apikey)

## Installation

```bash
git clone https://github.com/slackermafia/NanoBanana-MCP-Server.git
cd NanoBanana-MCP-Server
npm install
```

## Configuration

Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Claude Desktop / Cowork

Add this to your MCP server configuration:

```json
{
  "mcpServers": {
    "nanobanana": {
      "command": "node",
      "args": ["/absolute/path/to/NanoBanana-MCP-Server/src/index.js"],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Tools

### `gemini_generate_image`

Generate an image from a text prompt (synchronous, single image).

| Parameter      | Type   | Required | Description                                |
|----------------|--------|----------|--------------------------------------------|
| `prompt`       | string | Yes      | Detailed description of the image to create |
| `aspect_ratio` | string | No       | Output aspect ratio (e.g. `16:9`, `1:1`, `9:16`) |
| `model`        | string | No       | Gemini model ID (default: `gemini-3-pro-image-preview`) |
| `output_path`  | string | No       | File path to save the generated image       |

### `gemini_edit_image`

Edit one or more images using a text instruction (synchronous).

| Parameter          | Type   | Required | Description                                          |
|--------------------|--------|----------|------------------------------------------------------|
| `prompt`           | string | Yes      | Text instruction describing the edit                  |
| `image_paths`      | string | No*      | Comma-separated list of file paths to input images    |
| `image_base64_list`| string | No*      | JSON array of `{"data","mimeType"}` objects            |
| `aspect_ratio`     | string | No       | Output aspect ratio                                   |
| `model`            | string | No       | Gemini model ID                                       |
| `output_path`      | string | No       | File path to save the edited image                    |

\* You must provide at least one image via `image_paths` or `image_base64_list`.

### `gemini_batch_submit`

Submit a batch of image generation requests at **50% reduced cost**. Jobs run asynchronously (typically completes within 24 hours).

| Parameter      | Type   | Required | Description                                          |
|----------------|--------|----------|------------------------------------------------------|
| `requests`     | string | Yes      | JSON array of request objects (see below)             |
| `output_dir`   | string | Yes      | Directory where completed images will be saved        |
| `model`        | string | No       | Gemini model ID                                       |
| `display_name` | string | No       | Human-readable name for the batch job                 |

Each request object in the `requests` array:

```json
{
  "key": "pink-flamingo",
  "prompt": "A neon pink flamingo sign on a dark wall",
  "aspect_ratio": "1:1",
  "image_paths": "/optional/reference/image.jpg"
}
```

The `key` is used as the output filename — so `"pink-flamingo"` produces `pink-flamingo.jpg`. This is how you match input prompts to output images.

A JSONL input file is saved to `data/` for debugging, and the job ID is tracked in `data/batch_jobs.json`.

### `gemini_batch_status`

Check the status of pending batch jobs.

| Parameter    | Type   | Required | Description                                          |
|--------------|--------|----------|------------------------------------------------------|
| `batch_name` | string | No       | Specific batch ID (e.g. `batches/abc123`). Omit to check all. |

Returns the current state of each job: `JOB_STATE_PENDING`, `JOB_STATE_RUNNING`, `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`, or `JOB_STATE_CANCELLED`.

### `gemini_batch_results`

Download and save images from completed batch jobs.

| Parameter    | Type   | Required | Description                                          |
|--------------|--------|----------|------------------------------------------------------|
| `batch_name` | string | No       | Specific batch ID. Omit to process all completed jobs. |
| `output_dir` | string | No       | Override the output directory from submission time.    |

Downloads the output JSONL from Gemini, decodes each image, and saves it using the `key` as the filename. Also saves the raw output JSONL to `data/` for debugging.

## Batch Workflow

```
1. Submit batch     →  gemini_batch_submit (creates JSONL, uploads, starts job)
2. Wait             →  Job runs async on Google's side (up to 24h, usually faster)
3. Check status     →  gemini_batch_status (poll for completion)
4. Download results →  gemini_batch_results (saves images to output_dir as {key}.jpg)
```

A Cowork scheduled task (`nanobanana-batch-poll`) can be set up to automatically poll every hour and download results when jobs complete.

## File Structure

```
NanoBanana-MCP-Server/
├── src/
│   ├── index.js          # MCP server with all 5 tools
│   └── batch.js          # Batch API helpers, JSONL builder, job tracking
├── data/
│   ├── batch_jobs.json   # Tracked batch jobs (state, IDs, paths)
│   ├── batch_input_*.jsonl   # Input JSONL files (for debugging)
│   └── batch_output_*.jsonl  # Output JSONL files (for debugging)
├── package.json
└── README.md
```

## Supported Aspect Ratios

`1:1`, `3:2`, `2:3`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`

## License

MIT
