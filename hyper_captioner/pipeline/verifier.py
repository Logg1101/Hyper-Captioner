"""
Visual Grounding Verifier: Uses Florence-2 on-demand to resolve attribute conflicts
and verify physical grounding of uncertain tags.
"""

import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image

from hyper_captioner.core.types import ModelSource, TagItem
from hyper_captioner.engines.florence2 import Florence2Engine

logger = logging.getLogger(__name__)


class VisualGroundingVerifier:
    """
    On-demand verifier using Florence-2.
    Only instantiated and run when attribute conflict resolution
    or explicit user grounding checks are requested.
    """

    def __init__(self, model_source: ModelSource = ModelSource.LOCAL_ONLY):
        self.model_source = model_source
        self.florence: Optional[Florence2Engine] = None

    def _ensure_florence(self):
        if self.florence is None:
            self.florence = Florence2Engine(model_source=self.model_source)
            self.florence.load()

    def unload(self):
        if self.florence is not None:
            self.florence.unload()
            self.florence = None

    def verify_grounding(self, image: Image.Image, tag_text: str) -> Dict:
        """Checks if a tag is physically grounded in the image via Florence-2."""
        self._ensure_florence()

        query = tag_text.lower().strip().replace("_", " ")
        detailed = self.florence.run_task(image, "<DETAILED_CAPTION>")
        dense = self.florence.run_task(image, "<DENSE_REGION_CAPTION>")

        dense_labels = dense.get("labels", []) if isinstance(dense, dict) else []

        # Check for query or its primary keyword in detailed caption or dense region labels
        in_detailed = query in detailed.lower()
        in_dense = any(query in label.lower() for label in dense_labels)

        # Token match for multi-word concepts
        tokens = [w for w in query.split() if len(w) > 2]
        token_match = any(all(tok in label.lower() for tok in tokens) for label in dense_labels) if tokens else False

        grounded = in_detailed or in_dense or token_match

        return {
            "tag": tag_text,
            "grounded": grounded,
            "detailed_caption": detailed,
            "dense_labels": dense_labels,
        }

    def resolve_contradiction(
        self, image: Image.Image, item_a: TagItem, item_b: TagItem
    ) -> List[TagItem]:
        """
        Runs visual grounding to verify which conflicting attribute is real.
        If both are visible, preserves both. Otherwise preserves the grounded attribute.
        """
        self._ensure_florence()

        grounding_a = self.verify_grounding(image, item_a.text)
        grounding_b = self.verify_grounding(image, item_b.text)

        is_a_grounded = grounding_a.get("grounded", False)
        is_b_grounded = grounding_b.get("grounded", False)

        if is_a_grounded and not is_b_grounded:
            return [item_a]
        elif is_b_grounded and not is_a_grounded:
            return [item_b]
        elif is_a_grounded and is_b_grounded:
            # Both are genuinely visible in the image!
            return [item_a, item_b]
        else:
            # Fallback to source priority and confidence
            return [item_a if item_a.confidence >= item_b.confidence else item_b]
