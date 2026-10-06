"""
WD14 ONNX Tagger engine supporting SmilingWolf SwinV2, ViT, and ConvNeXt models.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

from hyper_captioner.config import DEFAULT_WD14_MODEL, ModelMissingError
from hyper_captioner.core.types import ModelSource, WD14Device
from hyper_captioner.core.vram import resolve_wd14_provider
from hyper_captioner.engines.base import BaseTagger

logger = logging.getLogger(__name__)


class WD14Engine(BaseTagger):
    """
    SmilingWolf WD14 ONNX tagger engine.
    Supports Auto/CUDA/CPU execution providers, Danbooru category parsing,
    and distinct general and character confidence thresholds.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_WD14_MODEL,
        device_choice: WD14Device = WD14Device.AUTO,
        model_source: ModelSource = ModelSource.LOCAL_ONLY,
    ):
        self.model_id = model_id
        self.device_choice = device_choice
        self.model_source = model_source
        self.session: Optional[ort.InferenceSession] = None
        self.tags_df: Optional[pd.DataFrame] = None
        self.general_indices: np.ndarray = np.array([])
        self.character_indices: np.ndarray = np.array([])
        self.general_tags: List[str] = []
        self.character_tags: List[str] = []
        self.target_size: int = 448
        self.active_provider: str = ""

    def load(self, model_id: Optional[str] = None):
        if model_id:
            self.model_id = model_id

        if self.session is not None:
            return

        logger.info(f"Loading WD14 Tagger: {self.model_id} (Source: {self.model_source.value})...")
        local_only = (self.model_source == ModelSource.LOCAL_ONLY)

        try:
            csv_path = hf_hub_download(
                repo_id=self.model_id,
                filename="selected_tags.csv",
                local_files_only=local_only,
            )
            model_path = hf_hub_download(
                repo_id=self.model_id,
                filename="model.onnx",
                local_files_only=local_only,
            )
        except Exception as e:
            if local_only:
                msg = (
                    f"WD14 model files for '{self.model_id}' not found in local cache.\n"
                    f"Switch Model Source to 'Local + Download Missing Models' to download once."
                )
                logger.error(msg)
                raise ModelMissingError(msg) from e
            # Retry with download
            csv_path = hf_hub_download(repo_id=self.model_id, filename="selected_tags.csv")
            model_path = hf_hub_download(repo_id=self.model_id, filename="model.onnx")

        # Parse tags
        self.tags_df = pd.read_csv(csv_path)
        # Category 0 = general tags, 4 = character tags
        gen_mask = self.tags_df["category"] == 0
        char_mask = self.tags_df["category"] == 4

        self.general_indices = self.tags_df[gen_mask].index.to_numpy()
        self.general_tags = self.tags_df.loc[gen_mask, "name"].tolist()

        self.character_indices = self.tags_df[char_mask].index.to_numpy()
        self.character_tags = self.tags_df.loc[char_mask, "name"].tolist()

        # Target size from model architecture (ViT-v2 usually 384 or 448, SwinV2 448)
        if "vit" in self.model_id.lower() and "v2" in self.model_id.lower():
            self.target_size = 448
        else:
            self.target_size = 448

        # Execution provider
        provider = resolve_wd14_provider(self.device_choice)
        available_providers = ort.get_available_providers()
        if provider not in available_providers:
            logger.warning(f"Requested provider {provider} not available in ONNX Runtime. Falling back to CPU.")
            provider = "CPUExecutionProvider"

        self.session = ort.InferenceSession(model_path, providers=[provider])
        self.active_provider = provider
        logger.info(f"WD14 Tagger loaded successfully with provider: {provider}")

    def unload(self):
        if self.session is not None:
            del self.session
            self.session = None
        self.tags_df = None
        logger.info("WD14 Tagger unloaded from memory.")

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        width, height = image.size
        max_dim = max(width, height)
        scale = self.target_size / max_dim
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))

        resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        # Pad with white canvas
        padded = Image.new("RGB", (self.target_size, self.target_size), (255, 255, 255))
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        padded.paste(resized, (paste_x, paste_y))

        # Convert to BGR float32 expected by SmilingWolf taggers
        img_array = np.array(padded, dtype=np.float32)
        img_array = img_array[:, :, ::-1]  # RGB to BGR
        return np.expand_dims(img_array, axis=0)

    def predict(
        self,
        image: Image.Image,
        general_threshold: float = 0.35,
        character_threshold: float = 0.60,
    ) -> List[Tuple[str, float, int]]:
        """
        Runs tagger inference.
        Returns sorted list of (tag_name, confidence_score, category_id) tuples.
        Category 0 = General, Category 4 = Character.
        """
        if self.session is None:
            self.load()

        input_tensor = self.preprocess_image(image)
        input_name = self.session.get_inputs()[0].name
        raw_output = self.session.run(None, {input_name: input_tensor})[0][0]

        results: List[Tuple[str, float, int]] = []

        # Character tags (Category 4)
        for idx, tag in zip(self.character_indices, self.character_tags):
            score = float(raw_output[idx])
            if score >= character_threshold:
                results.append((tag, score, 4))

        # General tags (Category 0)
        for idx, tag in zip(self.general_indices, self.general_tags):
            score = float(raw_output[idx])
            if score >= general_threshold:
                results.append((tag, score, 0))

        # Sort: Characters first (descending confidence), then General tags (descending confidence)
        results.sort(key=lambda x: (0 if x[2] == 4 else 1, -x[1]))
        return results
