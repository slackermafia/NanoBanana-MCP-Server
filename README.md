# NanoBanana MCP Server

An MCP (Model Context Protocol) server that connects to the Google Gemini API to generate and edit images using the **Nano Banana Pro** image generation models.

## Features

- **Text-to-Image Generation** — Describe an image and get it generated via the Gemini API.
- **Image Editing** — Provide one or more existing images and a text instruction to edit or transform them.
- **Multi-Image Input** — Send multiple images for blending, style transfer, collages, and more.
- **Aspect Ratio Control** — Force output to a specific aspect ratio (1:1, 16:9, 9:16, etc.).
- **File Output** — Optionally save generated images directly to disk.
- **Nano Banana Pro** — Uses the `gemini-3-pro-image-preview` model by default.

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

Generate an image from a text prompt.

| Parameter      | Type   | Required | Description                                |
|----------------|--------|----------|--------------------------------------------|
| `prompt`       | string | Yes      | Detailed description of the image to create |
| `aspect_ratio` | string | No       | Output aspect ratio (e.g. `16:9`, `1:1`, `9:16`) |
| `model`        | string | No       | Gemini model ID (default: `gemini-3-pro-image-preview`) |
| `output_path`  | string | No       | File path to save the generated image       |

**Supported aspect ratios:** `1:1`, `3:2`, `2:3`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`

### `gemini_edit_image`

Edit one or more images using a text instruction.

| Parameter          | Type   | Required | Description                                          |
|--------------------|--------|----------|------------------------------------------------------|
| `prompt`           | string | Yes      | Text instruction describing the edit                  |
| `image_paths`      | string | No*      | Comma-separated list of file paths to input images    |
| `image_base64_list`| string | No*      | JSON array of `{"data","mimeType"}` objects            |
| `aspect_ratio`     | string | No       | Output aspect ratio (e.g. `16:9`, `1:1`, `9:16`)     |
| `model`            | string | No       | Gemini model ID                                       |
| `output_path`      | string | No       | File path to save the edited image                    |

\* You must provide at least one image via `image_paths` or `image_base64_list`.

**Multi-image examples:**
- `image_paths: "/path/to/photo.jpg,/path/to/style.png"` with prompt `"Apply the style of the second image to the first"`
- `image_paths: "/path/img1.jpg,/path/img2.jpg"` with prompt `"Combine these into a side-by-side collage"`

## License

MIT
