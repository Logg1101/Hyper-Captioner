import csv
import json
import re

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

META_TAG_BLACKLIST = {
    "highres", "absurdres", "superabsurdres", "incredibly absurdres", "lowres",
    "bad anatomy", "bad hands", "missing fingers", "watermark", "signature",
    "artist name", "username", "web address", "logo", "dated", "copyright",
    "character request", "translation request", "meta", "sample", "scan",
    "source anime", "source manga", "source game", "source original", "spoiler",
    "patreon username", "twitter username",
    # Quality and camera noise that degrades LoRA quality:
    "photo (medium)", "photo", "selfie", "virtual reality", "screencap",
    # Common LLM hallucinated words that are not valid Danbooru tags:
    "bloodstain", "blood stain"
}

def clean_tags(tags_input, keep_underscores=True, filter_poisons=True, trigger_word="", allow_synonyms=True):
    if not tags_input:
        return ""

    if isinstance(tags_input, str):
        raw_list = tags_input.split(",")
    else:
        raw_list = []
        for item in tags_input:
            if item:
                raw_list.extend(str(item).split(","))

    cleaned = []
    seen_lower = set()
    specified_eye_colors = set()
    specified_hair_colors = set()
    specified_breast_sizes = set()

    for item in raw_list:
        raw = item.strip().strip("`'\"!?:;~. \t\n\r").strip("_")
        if not raw or not re.search(r"[a-zA-Z0-9]", raw):
            continue

        norm = re.sub(r"[\s_]+", " ", raw).strip()
        lower = norm.lower()

        if filter_poisons:
            if re.match(r"^(artist|copyright|meta)\s*:", lower):
                continue
            if re.match(r"^character\s*:", lower):
                if trigger_word:
                    continue
                norm = re.sub(r"^character\s*:\s*", "", norm, flags=re.IGNORECASE).strip()
                lower = norm.lower()
            if lower in META_TAG_BLACKLIST or lower.replace(" ", "_") in META_TAG_BLACKLIST:
                continue

        # Prevent contradictory attributes unless allow_synonyms is enabled for rich high-fidelity tagging
        eye_match = re.match(r"^(\w+)\s+eyes$", lower)
        if eye_match:
            color = eye_match.group(1)
            if not allow_synonyms:
                if specified_eye_colors and color not in specified_eye_colors:
                    continue
            specified_eye_colors.add(color)

        hair_color_match = re.match(r"^(\w+)\s+hair$", lower)
        if hair_color_match and hair_color_match.group(1) not in {"long", "short", "medium", "curly", "straight", "wavy", "spiky"}:
            color = hair_color_match.group(1)
            if not allow_synonyms:
                if specified_hair_colors and color not in specified_hair_colors:
                    continue
            specified_hair_colors.add(color)

        breast_match = re.match(r"^(flat chest|small breasts|medium breasts|large breasts|huge breasts)$", lower)
        if breast_match:
            size = breast_match.group(1)
            if specified_breast_sizes and size not in specified_breast_sizes:
                continue
            specified_breast_sizes.add(size)

        if lower not in seen_lower:
            seen_lower.add(lower)
            cleaned.append(norm)

    pruned = []
    if not allow_synonyms:
        # Strict legacy redundancy pruning
        has_specific_breasts = bool(specified_breast_sizes) or any(t in seen_lower for t in ["large breasts", "medium breasts", "small breasts", "huge breasts", "flat chest"])
        has_specific_dress = any(t.endswith(" dress") and t != "dress" for t in seen_lower)
        has_specific_jewelry = any(t in seen_lower for t in ["necklace", "gold chain necklace", "earrings", "bracelet", "ring", "choker", "wristband"])
        has_specific_necklace = "gold chain necklace" in seen_lower
        has_specific_hair = bool(specified_hair_colors) or any(t.endswith(" hair") and t != "hair" for t in seen_lower) or any(t in seen_lower for t in ["long hair", "short hair", "bangs", "ponytail", "twintails"])
        has_nurse = any(t in seen_lower for t in ["nurse", "nurse uniform", "nurse cap"])
        is_single_subject = any(t in seen_lower for t in ["1girl", "1boy", "solo"]) and not any(t in seen_lower for t in ["2girls", "2boys", "multiple girls", "multiple boys"])

        for t in cleaned:
            tl = t.lower()
            if tl == "breasts" and has_specific_breasts:
                continue
            if tl == "dress" and has_specific_dress:
                continue
            if tl == "jewelry" and has_specific_jewelry:
                continue
            if tl == "gold jewelry" and has_specific_necklace:
                continue
            if tl == "hair" and has_specific_hair:
                continue
            if tl == "doctor" and has_nurse and is_single_subject:
                continue

            tag_str = t.replace(" ", "_") if keep_underscores else t
            pruned.append(tag_str)
    else:
        # High-fidelity multi-layer mode: preserves rich granular tags and coexisting descriptors
        for t in cleaned:
            tl = t.lower()
            if tl in {"breasts", "hair"}:
                continue
            tag_str = t.replace(" ", "_") if keep_underscores else t
            pruned.append(tag_str)

    return ", ".join(pruned)

JOYCAPTION_MODEL = "fancyfeast/llama-joycaption-beta-one-hf-llava"
FLORENCE_MODEL = "microsoft/Florence-2-large"

_engines = {
    "joycaption": {"processor": None, "model": None},
    "florence": {"processor": None, "model": None}
}

def load_hf_resource(loader_fn, model_name_or_path, **kwargs):
    """
    Attempt to load from local cache first (local_files_only=True) to avoid unnecessary
    network requests, rate limits, or DNS/connection errors. Falls back to online hub if not found.
    """
    try:
        return loader_fn(model_name_or_path, local_files_only=True, **kwargs)
    except Exception:
        return loader_fn(model_name_or_path, local_files_only=False, **kwargs)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return "cpu"

def load_engine(engine_choice):
    device = get_device()
    if engine_choice == "JoyCaption (4-bit)":
        if _engines["joycaption"]["processor"] is None:
            print("\nLoading Optimized JoyCaption (4-bit NF4 + SDPA)...")
            if torch.cuda.is_available():
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
                    torch.backends.cudnn.allow_tf32 = True

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"]
            )
            processor = load_hf_resource(AutoProcessor.from_pretrained, JOYCAPTION_MODEL)
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token_id is None:
                processor.tokenizer.pad_token_id = 128001

            model = load_hf_resource(
                LlavaForConditionalGeneration.from_pretrained,
                JOYCAPTION_MODEL, 
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="sdpa",
                device_map="auto" if torch.cuda.is_available() else None
            )
            model.eval()
            _engines["joycaption"]["processor"] = processor
            _engines["joycaption"]["model"] = model
        return _engines["joycaption"]["processor"], _engines["joycaption"]["model"]
        
    elif engine_choice == "Florence-2 (Large)":
        if _engines["florence"]["processor"] is None:
            print("\nLoading Florence-2...")
            processor = load_hf_resource(
                AutoProcessor.from_pretrained,
                FLORENCE_MODEL, trust_remote_code=True
            )
            model = load_hf_resource(
                AutoModelForCausalLM.from_pretrained,
                FLORENCE_MODEL, trust_remote_code=True, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(device)
            # Tie missing embedding weights from shared weight (BART architecture)
            shared = model.language_model.model.shared.weight
            model.language_model.model.encoder.embed_tokens.weight = shared
            model.language_model.model.decoder.embed_tokens.weight = shared
            model.language_model.lm_head.weight = shared
            model.eval()
            _engines["florence"]["processor"] = processor
            _engines["florence"]["model"] = model
        return _engines["florence"]["processor"], _engines["florence"]["model"]

_SUBJECT_PROMPTS = {
    "Character": (
        "Output comma-separated Stable Diffusion training tags for a CHARACTER LoRA.\n"
        "Rules:\n"
        "- Tag variable attributes ONLY: clothing, pose, expression, accessories, environment, framing.\n"
        "- Do NOT tag stable physical features (hair color, eye color, body type) — the trigger word handles those.\n"
        "- No synonym pairs like 'red eyes, crimson eyes' or 'standing, standing pose'.\n"
        "- No flowery language. No sentences.\n"
        "Return ONLY the comma-separated tags."
    ),
    "Style": (
        "Output comma-separated Stable Diffusion training tags for a STYLE LoRA.\n"
        "Rules:\n"
        "- Describe ONLY the visual rendering style: art medium, shading technique, line quality, color palette approach, texture fidelity, lighting mood.\n"
        "- Do NOT tag character identity, hair, clothing, pose, objects, or story content.\n"
        "- No sentences.\n"
        "Return ONLY the comma-separated style tags."
    ),
    "Pose": (
        "Output comma-separated Stable Diffusion training tags for a POSE LoRA.\n"
        "Rules:\n"
        "- Describe ONLY body position: stance, posture, limb placement, weight distribution, gesture, orientation, shot framing.\n"
        "- Do NOT tag character identity, hair, clothing, art style, or background.\n"
        "- No sentences.\n"
        "Return ONLY the comma-separated pose tags."
    ),
    "Concept": (
        "Output comma-separated Stable Diffusion training tags for a CONCEPT LoRA.\n"
        "Rules:\n"
        "- Tag the specific concept and enough surrounding context to isolate it (framing, count tags).\n"
        "- Do NOT tie the concept to a specific character's identity, hair, or eye color.\n"
        "- No sentences.\n"
        "Return ONLY the comma-separated tags."
    ),
    "Lighting": (
        "Output comma-separated Stable Diffusion training tags for a LIGHTING LoRA.\n"
        "Rules:\n"
        "- Describe ONLY observable lighting: direction, softness/hardness, intensity, shadow quality, highlights, rim light, contrast, color temperature, light source type.\n"
        "- Do NOT tag character, clothing, pose, or art style.\n"
        "- No sentences.\n"
        "Return ONLY the comma-separated lighting tags."
    ),
    "General": (
        "Output comma-separated Danbooru-style tags (lowercase_underscores) for this image.\n"
        "Rules:\n"
        "- Tag only what is clearly visible and relevant.\n"
        "- No duplicate or synonym tags. No invented details. No sentences.\n"
        "Return ONLY the comma-separated tags."
    ),
}

def get_joycaption_prompt(subject_type, text_style, trigger_word, extra_features=None, quality_boosters=True):
    if text_style == "Flux (Natural Language)":
        base = "Write a detailed description for this image in natural language prose. Do not use comma-separated tags. Describe the subject, appearance, clothing, pose, lighting, and environment directly."
        if trigger_word:
            base += f" Always refer to the main subject as '{trigger_word}'."
        return base

    return _SUBJECT_PROMPTS.get(subject_type, _SUBJECT_PROMPTS["General"])

def generate_with_joycaption(processor, model, images, subject_type, text_style, trigger_word, extra_features, keep_underscores=True, filter_poisons=True, max_tokens=None, quality_boosters=True):
    is_single = not isinstance(images, (list, tuple))
    image_list = [images] if is_single else images

    prompt = get_joycaption_prompt(subject_type, text_style, trigger_word, extra_features, quality_boosters=quality_boosters)
    convo = [
        {
            "role": "system",
            "content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
        },
        {
            "role": "user",
            "content": prompt.strip(),
        }
    ]
    convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    
    if max_tokens is None:
        max_tokens = 450 if text_style == "Flux (Natural Language)" else 350

    device = get_device()
    inputs = processor(
        text=[convo_string] * len(image_list),
        images=image_list,
        return_tensors="pt",
        padding=True
    ).to(device)
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    with torch.inference_mode():
        if text_style == "Flux (Natural Language)":
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=True,
            )
        else:
            # Deterministic greedy decoding with exact EOS stopping to prevent runaway hallucination
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                eos_token_id=[128001, 128009],
                pad_token_id=processor.tokenizer.pad_token_id or 128001,
                do_sample=False,
                repetition_penalty=1.02,
                use_cache=True,
            )

    results = []
    input_len = inputs["input_ids"].shape[1]
    for i in range(len(image_list)):
        gen = generate_ids[i][input_len:]
        text = processor.tokenizer.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if text_style != "Flux (Natural Language)":
            text = clean_tags(
                text,
                keep_underscores=keep_underscores,
                filter_poisons=filter_poisons,
                trigger_word=trigger_word,
                allow_synonyms=False
            )
        results.append(text)

    return results[0] if is_single else results

def generate_with_florence(processor, model, image):
    task_prompt = "<MORE_DETAILED_CAPTION>"
    device = get_device()
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    inputs["pixel_values"] = processor.image_processor(image, return_tensors="pt")["pixel_values"]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].half() if torch.cuda.is_available() else inputs["pixel_values"].float()
    
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            use_cache=False
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    caption = parsed_answer.get(task_prompt, generated_text)
    return caption.strip()

def build_final_caption(generated_caption, custom_tags="", trigger_word="", wd14_tags=None, keep_underscores=True, filter_poisons=True, is_natural_language=False, subject_type="General", quality_boosters=True):
    if is_natural_language:
        parts = []
        if trigger_word and trigger_word.strip():
            parts.append(trigger_word.strip())
        if custom_tags and custom_tags.strip():
            parts.append(custom_tags.strip())
        if generated_caption and generated_caption.strip():
            parts.append(generated_caption.strip())
        return ", ".join(parts)

    tag_sources = []
    if trigger_word and trigger_word.strip():
        tag_sources.append(trigger_word.strip())
    if custom_tags and custom_tags.strip():
        tag_sources.append(custom_tags.strip())

    cleaned_gen = clean_tags(
        generated_caption,
        keep_underscores=keep_underscores,
        filter_poisons=filter_poisons,
        trigger_word=trigger_word,
        allow_synonyms=False
    )
    if cleaned_gen:
        tag_sources.append(cleaned_gen)

    # WD14 supplement: skip for Style/Pose/Lighting — those modes must not pick up
    # character/clothing ground-truth tags that pollute the training target.
    use_wd14_supplement = subject_type not in {"Style", "Pose", "Lighting"}
    if wd14_tags and use_wd14_supplement:
        existing = {
            t.strip().lower().replace(" ", "_")
            for t in (",".join(tag_sources)).split(",")
            if t.strip()
        }
        additional_wd14 = [
            wt for wt in wd14_tags
            if (wt_clean := wt.strip().lower().replace(" ", "_"))
            and wt_clean not in existing
            and wt_clean not in META_TAG_BLACKLIST
        ]
        if additional_wd14:
            cleaned_wd14 = clean_tags(
                additional_wd14,
                keep_underscores=keep_underscores,
                filter_poisons=filter_poisons,
                trigger_word=trigger_word,
                allow_synonyms=False
            )
            if cleaned_wd14:
                tag_sources.append(cleaned_wd14)

    if quality_boosters and subject_type in {"Character", "General"}:
        tag_sources.append("masterpiece, best_quality, highly_detailed, intricate_details")

    return clean_tags(
        tag_sources,
        keep_underscores=keep_underscores,
        filter_poisons=filter_poisons,
        trigger_word=trigger_word,
        allow_synonyms=False
    )

def caption_dataset(missing_images, caption_engine="JoyCaption (4-bit)", subject_type="General", text_style="High-Fidelity Danbooru (Multi-Layer / Illustrious / Pony)", trigger_word="", custom_tags="", extra_features=None, output_format="Sidecar (.txt)", batch_size=1, tag_format="clean_tags (underscores: blue_eyes)", filter_poisons=True, use_wd14=True, wd14_threshold=0.35, quality_boosters=True, max_tokens=None, progress_callback=None):
    created_count, skipped_count, failed_count = 0, 0, 0
    total_images = len(missing_images)
    
    processor, model = load_engine(caption_engine)
    
    # Store data for master files (CSV/JSON)
    dataset_captions = {}
    dataset_dir = missing_images[0].parent if missing_images else None

    # Filter out already existing captions if in sidecar mode
    to_process = []
    for img_path in missing_images:
        if output_format == "Sidecar (.txt)":
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                skipped_count += 1
                continue
        to_process.append(img_path)

    processed_count = skipped_count
    actual_batch_size = max(1, int(batch_size)) if caption_engine == "JoyCaption (4-bit)" else 1
    keep_underscores = ("underscores" in tag_format.lower() or "clean_tags" in tag_format.lower()) if text_style != "Flux (Natural Language)" else False
    is_nat_lang = (text_style == "Flux (Natural Language)")

    for idx in range(0, len(to_process), actual_batch_size):
        batch_paths = to_process[idx:idx + actual_batch_size]
        batch_images = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(img)
                valid_paths.append(p)
            except Exception as e:
                failed_count += 1
                processed_count += 1
                print(f"\nERROR opening {p.name}: {e}")
                if progress_callback:
                    progress_callback(processed_count, total_images, p.name)

        if not batch_images:
            continue

        # Optional WD14 classifier pass for pose, garments, and lighting ground truth
        batch_wd14 = [[] for _ in batch_images]
        if use_wd14 and not is_nat_lang:
            try:
                from wd14 import predict_wd14_tags
                batch_wd14 = [predict_wd14_tags(img, threshold=wd14_threshold) for img in batch_images]
            except Exception as e:
                print(f"\nWarning: WD14 tagging failed: {e}")

        try:
            if caption_engine == "JoyCaption (4-bit)":
                batch_captions = generate_with_joycaption(
                    processor, model, batch_images, subject_type, text_style, trigger_word, extra_features,
                    keep_underscores=keep_underscores, filter_poisons=filter_poisons,
                    max_tokens=max_tokens, quality_boosters=quality_boosters
                )
                if isinstance(batch_captions, str):
                    batch_captions = [batch_captions]
            else:
                batch_captions = [generate_with_florence(processor, model, img) for img in batch_images]

            for i, (path, gen_text) in enumerate(zip(valid_paths, batch_captions)):
                wd14_current = batch_wd14[i] if i < len(batch_wd14) else None
                final_caption = build_final_caption(
                    gen_text,
                    custom_tags=custom_tags,
                    trigger_word=trigger_word,
                    wd14_tags=wd14_current,
                    keep_underscores=keep_underscores,
                    filter_poisons=filter_poisons,
                    is_natural_language=is_nat_lang,
                    subject_type=subject_type,
                    quality_boosters=quality_boosters
                )
                if output_format == "Sidecar (.txt)":
                    path.with_suffix(".txt").write_text(final_caption, encoding="utf-8")
                else:
                    dataset_captions[path.name] = final_caption

                created_count += 1
                processed_count += 1
                if progress_callback:
                    progress_callback(processed_count, total_images, path.name)

        except Exception as error:
            failed_count += len(valid_paths)
            processed_count += len(valid_paths)
            print(f"\nERROR processing batch: {error}")
            if progress_callback and valid_paths:
                progress_callback(processed_count, total_images, valid_paths[-1].name)
            
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

