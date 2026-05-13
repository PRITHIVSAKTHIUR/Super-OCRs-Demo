import os
import random
import uuid
import json
import time
import asyncio
import re
import tempfile
import ast
import html
from threading import Thread
from typing import Iterable, Optional

import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import requests

# Import spaces if available, otherwise mock it
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    TextIteratorStreamer,
    HunYuanVLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration, 
)
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
# --- Theme Definition ---
colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()
css = """
#main-title h1 { font-size: 2.3em !important; }
#output-title h2 { font-size: 2.1em !important; }
"""

# --- Model Loading ---

# 1. DeepSeek-OCR
MODEL_DS = "prithivMLmods/DeepSeek-OCR-Latest-BF16.I64" # - (deepseek-ai/DeepSeek-OCR)
print(f"Loading {MODEL_DS}...")
tokenizer_ds = AutoTokenizer.from_pretrained(MODEL_DS, trust_remote_code=True)
model_ds = AutoModel.from_pretrained(
    MODEL_DS, trust_remote_code=True, use_safetensors=True
).to(device).eval()
if device.type == 'cuda':
    model_ds = model_ds.to(torch.bfloat16)

# 2. Dots.OCR
MODEL_DOTS = "prithivMLmods/Dots.OCR-Latest-BF16" # - (rednote-hilab/dots.ocr)
print(f"Loading {MODEL_DOTS}...")
processor_dots = AutoProcessor.from_pretrained(MODEL_DOTS, trust_remote_code=True)
model_dots = AutoModelForCausalLM.from_pretrained(
    MODEL_DOTS,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
).eval()

# 3. HunyuanOCR
MODEL_HUNYUAN = "tencent/HunyuanOCR"
print(f"Loading {MODEL_HUNYUAN}...")
processor_hy = AutoProcessor.from_pretrained(MODEL_HUNYUAN, use_fast=False)
model_hy = HunYuanVLForConditionalGeneration.from_pretrained(
    MODEL_HUNYUAN,
    attn_implementation="eager", # Use eager to avoid SDPA issues if old torch
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
).eval()

# 4. Nanonets-OCR2-3B
MODEL_ID_X = "nanonets/Nanonets-OCR2-3B"
print(f"Loading {MODEL_ID_X}...")
processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True)
model_x = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" # or .to(device)
).eval()

print("✅ All models loaded successfully.")

# --- Helper Functions ---

def clean_repeated_substrings(text):
    """Clean repeated substrings in text (for Hunyuan)"""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:] 
        count = 0
        i = n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]  
    return text

def find_result_image(path):
    for filename in os.listdir(path):
        if "grounding" in filename or "result" in filename:
            try:
                return Image.open(os.path.join(path, filename))
            except Exception as e:
                print(f"Error opening result image: {e}")
    return None

# --- Main Inference Logic ---

@spaces.GPU
def run_model(
    model_choice, 
    image, 
    ds_task_type, 
    ds_model_size, 
    ds_ref_text, 
    custom_prompt,
    max_new_tokens,
    temperature,
    top_p,
    top_k
):
    if image is None:
        yield "Please upload an image.", None
        return

    # === DeepSeek-OCR Logic ===
    if model_choice == "DeepSeek-OCR-Latest-BF16.I64":
        # Prepare Prompt based on Task
        if ds_task_type == "Free OCR":
            prompt = "<image>\nFree OCR."
        elif ds_task_type == "Convert to Markdown":
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
        elif ds_task_type == "Parse Figure":
            prompt = "<image>\nParse the figure."
        elif ds_task_type == "Locate Object by Reference":
            if not ds_ref_text or ds_ref_text.strip() == "":
                yield "Error: For 'Locate', you must provide Reference Text.", None
                return
            prompt = f"<image>\nLocate <|ref|>{ds_ref_text.strip()}<|/ref|> in the image."
        else:
            prompt = "<image>\nFree OCR."

        with tempfile.TemporaryDirectory() as output_path:
            temp_image_path = os.path.join(output_path, "temp_image.png")
            image.save(temp_image_path)
            
            # Size config
            size_configs = {
                "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
                "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
                "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
                "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
                "Gundam (Recommended)": {"base_size": 1024, "image_size": 640, "crop_mode": True},
            }
            config = size_configs.get(ds_model_size, size_configs["Gundam (Recommended)"])

            text_result = model_ds.infer(
                tokenizer_ds,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=output_path,
                base_size=config["base_size"],
                image_size=config["image_size"],
                crop_mode=config["crop_mode"],
                save_results=True,
                test_compress=True,
                eval_mode=True,
            )

            # Draw Bounding Boxes if present
            result_image_pil = None
            pattern = re.compile(r"<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>")
            matches = list(pattern.finditer(text_result))

            if matches:
                image_with_bboxes = image.copy()
                draw = ImageDraw.Draw(image_with_bboxes)
                w, h = image.size
                for match in matches:
                    coords_norm = [int(c) for c in match.groups()]
                    x1 = int(coords_norm[0] / 1000 * w)
                    y1 = int(coords_norm[1] / 1000 * h)
                    x2 = int(coords_norm[2] / 1000 * w)
                    y2 = int(coords_norm[3] / 1000 * h)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                result_image_pil = image_with_bboxes
            else:
                result_image_pil = find_result_image(output_path)
            
            yield text_result, result_image_pil

    # === Dots.OCR Logic ===
    elif model_choice == "Dots.OCR-Latest-BF16":
        query = custom_prompt if custom_prompt else "Extract all text from this image."
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ]
        }]
        
        prompt_full = processor_dots.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor_dots(text=[prompt_full], images=[image], return_tensors="pt", padding=True).to(model_dots.device)
        
        streamer = TextIteratorStreamer(processor_dots, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": int(top_k),
        }
        
        thread = Thread(target=model_dots.generate, kwargs=generation_kwargs)
        thread.start()
        
        buffer = ""
        for new_text in streamer:
            buffer += new_text.replace("<|im_end|>", "")
            yield buffer, None

    # === HunyuanOCR Logic ===
    elif model_choice == "HunyuanOCR":
        query = custom_prompt if custom_prompt else "检测并识别图片中的文字，将文本坐标格式化输出。"
        # Hunyuan template structure
        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": query},
                ],
            }
        ]
        
        # Note: Hunyuan processor expects specific handling
        texts = [processor_hy.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        inputs = processor_hy(text=texts, images=image, padding=True, return_tensors="pt")
        inputs = inputs.to(model_hy.device)
        
        # Generate (Not streaming for Hunyuan usually)
        with torch.no_grad():
            generated_ids = model_hy.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False 
            )
            
        input_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = generated_ids[:, input_len:]
        output_text = processor_hy.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        final_text = clean_repeated_substrings(output_text)
        yield final_text, None

    # === Nanonets-OCR2-3B Logic ===
    elif model_choice == "Nanonets-OCR2-3B":
        query = custom_prompt if custom_prompt else "Extract the text from this image."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ],
            }
        ]
        
        # Prepare inputs for Qwen2.5-VL based architecture
        text = processor_x.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor_x(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(model_x.device)

        streamer = TextIteratorStreamer(processor_x, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": int(top_k),
        }

        thread = Thread(target=model_x.generate, kwargs=generation_kwargs)
        thread.start()
        
        buffer = ""
        for new_text in streamer:
            buffer += new_text.replace("<|im_end|>", "")
            yield buffer, None

image_examples = [
    ["examples/1.jpg"],
    ["examples/2.jpg"],
    ["examples/3.jpg"],
]

with gr.Blocks() as demo:
    gr.Markdown("# **Super-OCRs-Demo**", elem_id="main-title")
    gr.Markdown("Compare DeepSeek-OCR, Dots.OCR, HunyuanOCR, and Nanonets-OCR2-3B in one space.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Global Inputs
            model_choice = gr.Dropdown(
                choices=["HunyuanOCR", "DeepSeek-OCR-Latest-BF16.I64", "Dots.OCR-Latest-BF16", "Nanonets-OCR2-3B"],
                label="Select Model",
                value="DeepSeek-OCR-Latest-BF16.I64"
            )
            image_input = gr.Image(type="pil", label="Upload Image", sources=["upload", "clipboard"], height=350)
            
            # DeepSeek Specific Options
            with gr.Group(visible=True) as ds_group:
                ds_model_size = gr.Dropdown(
                    choices=["Tiny", "Small", "Base", "Large", "Gundam (Recommended)"], 
                    value="Large", label="DeepSeek Resolution"
                )
                ds_task_type = gr.Dropdown(
                    choices=["Free OCR", "Convert to Markdown", "Parse Figure", "Locate Object by Reference"], 
                    value="Convert to Markdown", label="Task Type"
                )
                ds_ref_text = gr.Textbox(label="Reference Text (for 'Locate' task only)", placeholder="e.g., the title, red car...", visible=False)
            
            # General Prompt (for Dots/Hunyuan/Nanonets)
            with gr.Group(visible=False) as prompt_group:
                custom_prompt = gr.Textbox(label="Custom Query / Prompt", placeholder="Extract text...", lines=2, value="Convert to Markdown precisely.")

            with gr.Accordion("Advanced Settings", open=False):
                max_new_tokens = gr.Slider(minimum=128, maximum=8192, value=2048, step=128, label="Max New Tokens")
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="Temperature")
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top P")
                top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top K")

            submit_btn = gr.Button("Perform OCR", variant="primary")
            
            gr.Examples(examples=image_examples, inputs=image_input)

        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Recognized Text / Markdown", lines=15, interactive=True)
            output_image = gr.Image(label="Visual Grounding Result (DeepSeek Only)", type="pil")

    def update_visibility(model):
        is_ds = (model == "DeepSeek-OCR-Latest-BF16.I64")
        return gr.Group(visible=is_ds), gr.Group(visible=not is_ds)

    def toggle_ref_text(task):
        return gr.Textbox(visible=(task == "Locate Object by Reference"))

    model_choice.change(fn=update_visibility, inputs=model_choice, outputs=[ds_group, prompt_group])
    ds_task_type.change(fn=toggle_ref_text, inputs=ds_task_type, outputs=ds_ref_text)

    submit_btn.click(
        fn=run_model,
        inputs=[
            model_choice, image_input, ds_task_type, ds_model_size, ds_ref_text, 
            custom_prompt, max_new_tokens, temperature, top_p, top_k
        ],
        outputs=[output_text, output_image]
    )

if __name__ == "__main__":
    demo.queue(max_size=30).launch(css=css, theme=steel_blue_theme, mcp_server=True, ssr_mode=False, show_error=True)
