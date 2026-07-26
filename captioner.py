import csv
import json
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, AutoModelForCausalLM

JOYCAPTION_MODEL = "fancyfeast/llama-joycaption-beta-one-hf-llava"
FLORENCE_MODEL = "microsoft/Florence-2-large"

_engines = {
    "joycaption": {"processor": None, "model": None},
    "florence": {"processor": None, "model": None}
}

def load_engine(engine_choice):
    if engine_choice == "JoyCaption (4-bit)":
        if _engines["joycaption"]["processor"] is None:
            print("\nLoading Quantized JoyCaption...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            _engines["joycaption"]["processor"] = AutoProcessor.from_pretrained(JOYCAPTION_MODEL)
            _engines["joycaption"]["model"] = LlavaForConditionalGeneration.from_pretrained(
                JOYCAPTION_MODEL, 
                quantization_config=quantization_config, 
                device_map=0
            )
            _engines["joycaption"]["model"].eval()
        return _engines["joycaption"]["processor"], _engines["joycaption"]["model"]
        
    elif engine_choice == "Florence-2 (Large)":
        if _engines["florence"]["processor"] is None:
            print("\nLoading Florence-2...")
            _engines["florence"]["processor"] = AutoProcessor.from_pretrained(
                FLORENCE_MODEL, trust_remote_code=True
            )
            _engines["florence"]["model"] = AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL, trust_remote_code=True, torch_dtype=torch.float16
            ).cuda()
            _engines["florence"]["model"].eval()
        return _engines["florence"]["processor"], _engines["florence"]["model"]

def get_joycaption_prompt(subject_type, text_style, trigger_word, extra_features=None):
    if extra_features is None:
        extra_features = []

    if text_style == "Flux (Natural Language)":
        format_rules = "Write highly detailed, descriptive natural language sentences. Do not use comma-separated tags."
    else:
        format_rules = "Use strict, highly detailed, comma-separated Danbooru-style tags. Do not write sentences or stories."

    base_prompt = f"Create a comprehensive training caption.\n{format_rules}\nDo not describe image quality.\nBe extremely observant and detailed.\n\n"
    
    feature_prompts = []
    if "Camera Angle & Framing" in extra_features:
        feature_prompts.append("camera angle, perspective, and shot framing (e.g. close-up, wide shot, low angle)")
    if "Lighting & Shadows" in extra_features:
        feature_prompts.append("lighting setup, light sources, highlights, contrast, and shadows")
    if "Background & Environment" in extra_features:
        feature_prompts.append("detailed background environment, scenery, location, and atmosphere")

    extra_str = ""
    if feature_prompts:
        extra_str = "Also specifically describe: " + ", ".join(feature_prompts) + ".\n"
    
    if subject_type == "Character":
        prompt = base_prompt + extra_str + "Focus on: intricate clothing details, fabric textures, accessories, pose, hand position.\n"
    elif subject_type == "Clothing":
        prompt = base_prompt + extra_str + "Focus on: character pose, and expression. CRITICAL: Do NOT describe the specific clothing, garments, or fashion style being trained.\n"
    elif subject_type == "Style":
        prompt = base_prompt + extra_str + "Focus on: subject features, colors, and composition. CRITICAL: Do NOT describe the artistic style, brushstrokes, or medium.\n"
    elif subject_type == "Pose":
        prompt = base_prompt + extra_str + "Focus on: facial expression, and detailed clothing. CRITICAL: Do NOT describe the pose, action, or limb positioning.\n"
    else:
        prompt = base_prompt + extra_str + "Focus on: hair, eyes, expression, detailed clothing, fabric textures, and pose.\n"
        
    if trigger_word:
        prompt += f"\nAlways include the trigger word '{trigger_word}' at the very beginning of the caption.\n"
            
    return prompt

def generate_with_joycaption(processor, model, image, subject_type, text_style, trigger_word, extra_features):
    prompt = get_joycaption_prompt(subject_type, text_style, trigger_word, extra_features)
    convo = [
        {"role": "system", "content": "You are an expert image captioner for AI model training."},
        {"role": "user", "content": prompt}
    ]
    convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=180, do_sample=False, use_cache=True)[0]
    
    generate_ids = generate_ids[inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

def generate_with_florence(processor, model, image, trigger_word):
    task_prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    caption = parsed_answer[task_prompt]
    
    if trigger_word:
        caption = f"{trigger_word}, {caption}"
        
    return caption.strip()

def build_final_caption(generated_caption, custom_tags=""):
    parts = []
    if custom_tags:
        parts.append(custom_tags.strip())
    if generated_caption:
        parts.append(generated_caption)
    return ", ".join(parts) if "," in generated_caption or "," in custom_tags else " ".join(parts)

def caption_dataset(missing_images, caption_engine, subject_type, text_style, trigger_word, custom_tags, extra_features, output_format, progress_callback=None):
    created_count, skipped_count, failed_count = 0, 0, 0
    total_images = len(missing_images)
    
    processor, model = load_engine(caption_engine)
    
    # Store data for master files (CSV/JSON)
    dataset_captions = {}
    dataset_dir = missing_images[0].parent if missing_images else None
    
    for index, image_path in enumerate(missing_images, start=1):
        if output_format == "Sidecar (.txt)":
            txt_path = image_path.with_suffix(".txt")
            if txt_path.exists():
                skipped_count += 1
                continue
            
        if progress_callback:
            progress_callback(index, total_images, image_path.name)

        try:
            image = Image.open(image_path).convert("RGB")
            
            if caption_engine == "JoyCaption (4-bit)":
                generated_text = generate_with_joycaption(processor, model, image, subject_type, text_style, trigger_word, extra_features)
            else:
                generated_text = generate_with_florence(processor, model, image, trigger_word)
                
            final_caption = build_final_caption(generated_text, custom_tags)
            
            # Write to disk based on format selection
            if output_format == "Sidecar (.txt)":
                txt_path.write_text(final_caption, encoding="utf-8")
            else:
                dataset_captions[image_path.name] = final_caption
                
            created_count += 1
            
        except Exception as error:
            failed_count += 1
            print(f"\nERROR: {image_path.name}\n{error}")
            
    # Write master files if selected
    if output_format == "Master Metadata (.csv)" and dataset_dir and dataset_captions:
        csv_path = dataset_dir / "captions.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["image", "caption"])
            for img_name, cap in dataset_captions.items():
                writer.writerow([img_name, cap])
                
    elif output_format == "Master Metadata (.json)" and dataset_dir and dataset_captions:
        json_path = dataset_dir / "captions.json"
        json_path.write_text(json.dumps(dataset_captions, indent=4), encoding="utf-8")
            
    return {"created": created_count, "skipped": skipped_count, "failed": failed_count}