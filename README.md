# **Super-OCRs-Demo**

> A Gradio-based demo application for comparing state-of-the-art OCR models: DeepSeek-OCR, Dots.OCR, HunyuanOCR, and Nanonets-OCR2-3B. Users can upload images, select models, apply custom prompts, and generate recognized text or visual grounding results. Supports tasks like free OCR, markdown conversion, figure parsing, and object location.

## Features

- **Multi-Model Comparison**: Switch between DeepSeek-OCR (with resolution and task options), Dots.OCR, HunyuanOCR, and Nanonets-OCR2-3B for flexible OCR workflows.
- **Image Upload and Processing**: Supports direct upload or clipboard paste; handles various image formats with PIL.
- **Customizable Prompts**: Tailor queries for text extraction, detection, or specific tasks (e.g., "Extract all text" or "Locate the red car").
- **DeepSeek-Specific Tools**: Resolution presets (Tiny to Gundam), task types (Free OCR, Markdown, Parse Figure, Locate Object), and bounding box visualization.
- **Advanced Generation Controls**: Adjust max new tokens (up to 8192), temperature, top-p, and top-k for fine-tuned outputs.
- **Streaming Output**: Real-time text generation for Dots.OCR and Nanonets-OCR2-3B; non-streaming for others.
- **Visual Results**: DeepSeek outputs annotated images with bounding boxes or grounding visuals.
- **Custom Theme**: SteelBlueTheme for a modern, gradient-based UI with enhanced readability.
- **Examples and Queueing**: Built-in example images; supports queued inferences for up to 30 concurrent users.

## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (recommended for bfloat16 models; falls back to CPU).
- Git for cloning submodules.
- Hugging Face account (optional, for model caching via `huggingface_hub`).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Super-OCRs-Demo.git
   cd Super-OCRs-Demo
   ```

2. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   flash-attn @ https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.8/cu12/abiFALSE/cp310/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4
   git+https://github.com/huggingface/accelerate.git
   git+https://github.com/huggingface/diffusers.git
   git+https://github.com/huggingface/peft.git
   huggingface_hub
   qwen-vl-utils
   sentencepiece
   opencv-python
   torch==2.8.0
   torchvision
   supervision
   matplotlib
   easydict
   kernels
   einops
   spaces
   addict
   hf_xet
   numpy
   av
   ```

3. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860` (or the provided URL if using Spaces).

## Usage

1. **Select Model**: Choose from the radio buttons (default: DeepSeek-OCR).
   - DeepSeek: Adjust resolution (e.g., "Gundam (Recommended)") and task (e.g., "Convert to Markdown").
   - Others: Use the custom prompt textbox for queries like "Detect and extract all text with coordinates."

2. **Upload Image**: Drag-and-drop or paste an image (supports examples like receipts, figures, or documents).

3. **Configure Settings**:
   - For "Locate Object" in DeepSeek, enter reference text (e.g., "the title").
   - Tune advanced sliders for generation quality.

4. **Run Inference**: Click "Perform OCR" to process. Outputs stream to the textbox (with copy button); DeepSeek may show an annotated image.

5. **View Results**:
   - Text: Raw OCR output, markdown, or formatted coordinates.
   - Image: Bounding boxes in red for detected elements (DeepSeek only).

### Example Workflow
- Upload a receipt image.
- Select Dots.OCR, prompt: "Extract items and prices."
- Adjust temperature to 0.1 for deterministic results.
- Output: Structured text list.

## Supported Models

| Model Name                  | Key Capabilities                          | Notes                                      |
|-----------------------------|-------------------------------------------|--------------------------------------------|
| DeepSeek-OCR-Latest-BF16.I64 | Free OCR, Markdown, Figure Parsing, Object Location | Visual grounding with bounding boxes; resolution presets. |
| Dots.OCR-Latest-BF16       | General text extraction; streaming       | Qwen-based; custom prompts for flexibility. |
| HunyuanOCR                  | Detection and recognition with coordinates | Tencent model; handles Chinese/English well. |
| Nanonets-OCR2-3B            | High-accuracy extraction; streaming      | Qwen2.5-VL; suitable for complex layouts.  |

## Troubleshooting

- **Model Loading Errors**: Ensure CUDA is installed for GPU; use `torch.float32` fallback if bfloat16 fails.
- **Out of Memory**: Reduce resolution in DeepSeek or max_new_tokens; clear cache with `torch.cuda.empty_cache()`.
- **Import Issues**: Install `spaces` only if deploying to Hugging Face Spaces; mock it locally.
- **Generation Loops**: Hunyuan may repeat; cleaned automatically via `clean_repeated_substrings`.
- **UI Visibility**: Model changes toggle DeepSeek-specific groups dynamically.
- **Queue Full**: Increase `max_size` in `demo.queue()` for high traffic.

## Contributing

Contributions welcome! Open issues for bugs or features (e.g., more models, export to JSON). Fork, branch, and PR with tests.
Repository: [https://github.com/PRITHIVSAKTHIUR/Super-OCRs-Demo.git](https://github.com/PRITHIVSAKTHIUR/Super-OCRs-Demo.git)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by Prithiv Sakthi. Report issues via the repository.
