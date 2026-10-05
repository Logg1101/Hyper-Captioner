"""
JoyCaption engine (LLaVA-based vision-language model) with 4-bit quantization and SDPA attention.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

from hyper_captioner.config import DEFAULT_JOYCAPTION_MODEL, ModelMissingError, load_hf_resource
from hyper_captioner.core.types import CaptionMode, LoRAStrategy, ModelSource
from hyper_captioner.core.vram import clean_vram, is_cuda_available, resolve_torch_device
from hyper_captioner.engines.base import BaseCaptioner

logger = logging.getLogger(__name__)


def build_system_prompt(
    caption_mode: CaptionMode = CaptionMode.HYBRID,
    lora_strategy: LoRAStrategy = LoRAStrategy.CHARACTER,
    trigger_word: str = "",
    extra_instructions: str = "",
) -> str:
    """Builds an optimized instruction prompt for JoyCaption based on target LoRA and caption mode."""
    if caption_mode == CaptionMode.NATURAL:
        prompt = (
            "Write a concise, visually grounded natural language description of this image for LoRA dataset training. "
            "Focus strictly on what is visually observable: subjects, physical appearance, clothing, pose, action, "
            "lighting, and environment. Do not use flowery metaphors, speculative emotional states, or invisible backstories. "
        )
        if trigger_word:
            prompt += f"Refer to the primary subject as '{trigger_word}'. "
        return prompt.strip()

    if caption_mode == CaptionMode.TAG:
        if lora_strategy == LoRAStrategy.STYLE:
            return (
                "Output comma-separated descriptive tags for a STYLE LoRA dataset.\n"
                "Rules:\n"
                "- Focus strictly on visual rendering: art medium, linework quality, shading technique, color palette, lighting atmosphere.\n"
                "- Do NOT tag character identity, specific clothing details, or narrative storyline.\n"
                "- Return ONLY comma-separated tags."
            )
        elif lora_strategy == LoRAStrategy.CONCEPT:
            return (
                "Output comma-separated descriptive tags for a CONCEPT LoRA dataset.\n"
                "Rules:\n"
                "- Focus strictly on the primary concept, object, or mechanism and its direct interaction/context.\n"
                "- Do NOT tie the concept to specific character facial features unless necessary.\n"
                "- Return ONLY comma-separated tags."
            )
        else:
            return (
                "Output comma-separated Danbooru-style tags describing what is clearly visible in this image.\n"
                "Rules:\n"
                "- Include character count, key clothing items, pose, observable action, framing, and environment.\n"
                "- No duplicates, no speculative details, no sentences.\n"
                "- Return ONLY comma-separated tags."
            )

    # DEFAULT: HYBRID TRAINING MODE
    # Generates rich, visually grounded comma-separated phrases suitable for attribute fusion
    base = (
        "Analyze this image and output structured, visually grounded descriptive concepts separated by commas.\n"
        "Describe only what is visibly supported:\n"
        "- Subject & Count: (e.g. 1girl, solo)\n"
        "- Appearance: (e.g. black long hair, red eyes)\n"
        "- Clothing & Accessories: (e.g. maid uniform, form-fitting dress, thigh-highs, long gloves)\n"
        "- Pose & Action: (e.g. standing, wiping a window, reaching forward)\n"
        "- Camera & Framing: (e.g. side view, eye level, cowboy shot)\n"
        "- Setting & Environment: (e.g. indoors, large window, classroom)\n"
        "- Lighting: (e.g. natural daylight, soft shadows)\n"
        "Rules:\n"
        "- Output comma-separated phrases only.\n"
        "- Never hallucinate hidden items or unconfirmed character identities.\n"
        "- No conversational filler or preamble."
    )
    if trigger_word:
        base += f"\n- Name the main subject '{trigger_word}'."
    if extra_instructions:
        base += f"\n- {extra_instructions}"

    return base


def build_stage1_extraction_prompt(mode_instructions: str = "") -> str:
    """
    Builds a structured Stage 1 visual fact extraction prompt for JoyCaption.
    Instructs the model to output a strict JSON object mapping semantic categories
    to lists of factual, visible details, routing ambiguous or uncertain details
    to the 'uncertain' category.
    """
    prompt = (
        "Analyze this image and extract all factual, visible visual elements as a strict JSON object.\n"
        "Return ONLY a valid JSON object with the following category keys mapping to lists of concise strings:\n"
        "{\n"
        '  "identity": ["subject counts, gender/demographics, e.g. 1girl, solo"],\n'
        '  "appearance": ["hair color/style, eye color, facial features, body traits"],\n'
        '  "clothing": ["garments, accessories, footwear, hats, jewelry"],\n'
        '  "pose": ["body posture, gestures, physical actions"],\n'
        '  "expression": ["facial expressions, emotions visibly shown"],\n'
        '  "composition": ["shot framing, e.g. cowboy shot, close-up, full body"],\n'
        '  "camera": ["camera angle, perspective, depth of field"],\n'
        '  "environment": ["setting, background elements, indoors/outdoors, furniture"],\n'
        '  "lighting": ["light sources, lighting quality, shadows, highlights"],\n'
        '  "material": ["textures and material properties, e.g. silk, leather, metallic"],\n'
        '  "rendering": ["rendering techniques, shading, line art qualities"],\n'
        '  "style": ["art medium, visual genre, aesthetic style"],\n'
        '  "objects": ["held props, distinct scene items, tools, weapons"],\n'
        '  "color": ["dominant colors, color palettes, accents"],\n'
        '  "texture": ["surface textures, patterns, finishes"],\n'
        '  "concept": ["thematic elements, motifs, abstract visual ideas"],\n'
        '  "uncertain": ["any ambiguous, occluded, or partially visible items that cannot be definitively identified"]\n'
        "}\n\n"
        "Rules:\n"
        "1. Extract ONLY visually verifiable facts directly present in the image.\n"
        "2. Do NOT invent, assume, or hallucinate invisible backstory, specific unconfirmed character names, or unconfirmed lore.\n"
        "3. Any ambiguous, partially obscured, or uncertain details MUST be placed in the 'uncertain' list.\n"
        "4. Output valid JSON only, without commentary or markdown wrap."
    )
    if mode_instructions:
        prompt += f"\n\nAdditional Mode Instructions:\n{mode_instructions.strip()}"
    return prompt


class JoyCaptionEngine(BaseCaptioner):
    """
    JoyCaption Vision-Language Model backend.
    Uses 4-bit NF4 quantization for low VRAM usage and SDPA for fast inference.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_JOYCAPTION_MODEL,
        model_source: ModelSource = ModelSource.LOCAL_ONLY,
        load_in_4bit: bool = True,
    ):
        self.model_id = model_id
        self.model_source = model_source
        self.load_in_4bit = load_in_4bit
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[LlavaForConditionalGeneration] = None
        self.device = resolve_torch_device()

    def load(self, model_id: Optional[str] = None):
        if model_id:
            self.model_id = model_id

        if self.model is not None and self.processor is not None:
            return

        logger.info(f"Loading JoyCaption Model: {self.model_id} (Source: {self.model_source.value})...")

        # Speed optimizations for CUDA
        if is_cuda_available():
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
                torch.backends.cudnn.allow_tf32 = True

        # Load Processor
        self.processor = load_hf_resource(
            AutoProcessor.from_pretrained,
            self.model_id,
            model_source=self.model_source,
        )
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = 128001

        # Quantization Config
        quant_config = None
        if self.load_in_4bit and is_cuda_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
            )

        # Load Model
        self.model = load_hf_resource(
            LlavaForConditionalGeneration.from_pretrained,
            self.model_id,
            model_source=self.model_source,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if is_cuda_available() else torch.float32,
            attn_implementation="sdpa",
            device_map="auto" if (is_cuda_available() and quant_config is not None) else None,
        )
        if quant_config is None and is_cuda_available():
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("JoyCaption model loaded successfully.")

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        clean_vram()
        logger.info("JoyCaption model unloaded and VRAM cleared.")

    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 350,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[str]:
        """
        Runs batch inference with JoyCaption.
        images: list of PIL Images
        prompt: instruction prompt
        temperature: 0.0 for deterministic greedy decoding (best for tags/hybrid), >0.0 for sampling
        """
        if self.model is None or self.processor is None:
            self.load()

        convo = [
            {
                "role": "system",
                "content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
            },
            {
                "role": "user",
                "content": prompt.strip(),
            },
        ]
        convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[convo_string] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        )
        if is_cuda_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.inference_mode():
            if temperature > 0.0:
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    use_cache=True,
                )
            else:
                # Deterministic greedy generation with explicit EOS tokens
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=[128001, 128009],
                    pad_token_id=self.processor.tokenizer.pad_token_id or 128001,
                    do_sample=False,
                    repetition_penalty=1.02,
                    use_cache=True,
                )

        input_len = inputs["input_ids"].shape[1]
        results = []
        for i in range(len(images)):
            gen_tokens = gen_ids[i][input_len:]
            text = self.processor.tokenizer.decode(
                gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            results.append(text)

        return results
