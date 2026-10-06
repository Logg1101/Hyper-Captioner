"""
Florence-2 Visual Grounding and Object Detection engine (On-demand staged verification).
"""

import logging
from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from hyper_captioner.config import DEFAULT_FLORENCE_MODEL, load_hf_resource
from hyper_captioner.core.types import ModelSource
from hyper_captioner.core.vram import clean_vram, is_cuda_available, resolve_torch_device
from hyper_captioner.engines.base import BaseGrounding

logger = logging.getLogger(__name__)


class Florence2Engine(BaseGrounding):
    """
    Florence-2 model for on-demand visual grounding, object detection,
    and resolving ambiguous or conflicting attributes.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_FLORENCE_MODEL,
        model_source: ModelSource = ModelSource.LOCAL_ONLY,
    ):
        self.model_id = model_id
        self.model_source = model_source
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.device = resolve_torch_device()

    def load(self, model_id: Optional[str] = None):
        if model_id:
            self.model_id = model_id

        if self.model is not None and self.processor is not None:
            return

        logger.info(f"Loading Florence-2 Engine on-demand: {self.model_id}...")

        self.processor = load_hf_resource(
            AutoProcessor.from_pretrained,
            self.model_id,
            model_source=self.model_source,
            trust_remote_code=True,
        )

        dtype = torch.float16 if is_cuda_available() else torch.float32
        self.model = load_hf_resource(
            AutoModelForCausalLM.from_pretrained,
            self.model_id,
            model_source=self.model_source,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)

        # Tie shared embedding weights (BART architecture)
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "model"):
            lm_m = self.model.language_model.model
            if hasattr(lm_m, "shared"):
                shared = lm_m.shared.weight
                if hasattr(lm_m, "encoder"):
                    lm_m.encoder.embed_tokens.weight = shared
                if hasattr(lm_m, "decoder"):
                    lm_m.decoder.embed_tokens.weight = shared
                if hasattr(self.model.language_model, "lm_head"):
                    self.model.language_model.lm_head.weight = shared

        self.model.eval()
        logger.info("Florence-2 loaded successfully.")

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        clean_vram()
        logger.info("Florence-2 unloaded and VRAM freed.")

    def run_task(self, image: Image.Image, task_prompt: str, text_input: str = "") -> Any:
        if self.model is None or self.processor is None:
            self.load()

        prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs["pixel_values"] = self.processor.image_processor(image, return_tensors="pt")["pixel_values"]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if is_cuda_available():
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                use_cache=False,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        return parsed.get(task_prompt, parsed)

    def detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Returns detected objects with bounding boxes using <OD>."""
        res = self.run_task(image, "<OD>")
        bboxes = res.get("bboxes", [])
        labels = res.get("labels", [])
        output = []
        for box, label in zip(bboxes, labels):
            output.append({"box": box, "label": label})
        return output

    def verify_phrase_grounding(self, image: Image.Image, phrase: str) -> Dict[str, Any]:
        """
        Verifies whether a specific phrase/concept is grounded in the image
        using <CAPTION_TO_PHRASE_GROUNDING>.
        """
        res = self.run_task(image, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=phrase)
        bboxes = res.get("bboxes", [])
        labels = res.get("labels", [])
        grounded = len(bboxes) > 0
        return {
            "phrase": phrase,
            "grounded": grounded,
            "count": len(bboxes),
            "bboxes": bboxes,
            "labels": labels,
        }

    def verify_attributes(
        self, image: Image.Image, candidate_attributes: List[str]
    ) -> Dict[str, bool]:
        """Verifies candidate attributes against visual grounding."""
        results: Dict[str, bool] = {}
        for attr in candidate_attributes:
            grounding = self.verify_phrase_grounding(image, attr)
            results[attr] = grounding["grounded"]
        return results
