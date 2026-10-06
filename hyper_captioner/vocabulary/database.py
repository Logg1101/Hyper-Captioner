"""
Dataset-wide vocabulary indexer, frequency analyzer, and batch tag normalizer.
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatasetVocabularyIndex:
    """
    Indexes tag frequencies across an entire dataset and provides
    atomic dataset-wide tag replacement, normalization, and cleanup.
    """

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.tag_counts: Counter[str] = Counter()
        self.file_tag_map: Dict[Path, List[str]] = {}

    def index(self, dataset_path: Optional[Path] = None) -> Counter[str]:
        """
        Reads every .txt caption in the dataset, parses individual tags,
        and computes global frequency counts.
        """
        path = Path(dataset_path or self.dataset_path)
        if not path or not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        self.dataset_path = path
        self.tag_counts.clear()
        self.file_tag_map.clear()

        txt_files = sorted(path.rglob("*.txt"))
        for txt_file in txt_files:
            try:
                content = txt_file.read_text(encoding="utf-8").strip()
                if not content:
                    continue

                # Parse comma-separated tags
                tags = [t.strip() for t in content.split(",") if t.strip()]
                self.file_tag_map[txt_file] = tags

                for tag in tags:
                    norm = tag.strip().lower()
                    self.tag_counts[norm] += 1

            except Exception as e:
                logger.warning(f"Failed to read caption {txt_file}: {e}")

        logger.info(f"Indexed {len(self.file_tag_map)} caption files. Found {len(self.tag_counts)} distinct tags.")
        return self.tag_counts

    def get_top_tags(self, limit: int = 100) -> List[Tuple[str, int]]:
        """Returns the most frequent tags across the dataset."""
        return self.tag_counts.most_common(limit)

    def find_variants(self, query: str) -> List[Tuple[str, int]]:
        """
        Finds variant spellings, substrings, or synonyms of a query in the indexed tags.
        e.g., query='thigh' -> [('thigh high socks', 42), ('thigh-highs', 8), ('thighhighs', 31)]
        """
        q = query.lower().strip().replace("_", " ").replace("-", " ")
        q_tokens = set(q.split())
        results: List[Tuple[str, int]] = []

        for tag, count in self.tag_counts.items():
            t_clean = tag.replace("_", " ").replace("-", " ")
            t_tokens = set(t_clean.split())
            if q in t_clean or any(token in t_tokens for token in q_tokens):
                results.append((tag, count))

        results.sort(key=lambda x: -x[1])
        return results

    def replace_tag(self, old_tag: str, new_tag: str, dataset_path: Optional[Path] = None) -> int:
        """
        Atomically replaces old_tag with new_tag across all .txt files in the dataset.
        Returns count of modified files.
        """
        return self.batch_normalize({old_tag: new_tag}, dataset_path=dataset_path)

    def delete_tag(self, target_tag: str, dataset_path: Optional[Path] = None) -> int:
        """
        Removes target_tag completely from all .txt files in the dataset.
        Returns count of modified files.
        """
        target = target_tag.lower().strip()
        path = Path(dataset_path or self.dataset_path)
        modified_count = 0

        for txt_file in path.rglob("*.txt"):
            try:
                content = txt_file.read_text(encoding="utf-8").strip()
                if not content:
                    continue

                raw_tags = [t.strip() for t in content.split(",") if t.strip()]
                new_tags = [t for t in raw_tags if t.lower().strip() != target]

                if len(raw_tags) != len(new_tags):
                    new_caption = ", ".join(new_tags)
                    txt_file.write_text(new_caption, encoding="utf-8")
                    modified_count += 1
            except Exception as e:
                logger.error(f"Error updating {txt_file}: {e}")

        # Re-index to maintain fresh state
        self.index(path)
        return modified_count

    def batch_normalize(self, tag_mapping: Dict[str, str], dataset_path: Optional[Path] = None) -> int:
        """
        Applies a dictionary of {old_tag: new_canonical_tag} replacements
        across all .txt files in the dataset.
        """
        path = Path(dataset_path or self.dataset_path)
        if not path or not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        norm_map = {k.lower().strip(): v.strip() for k, v in tag_mapping.items() if k.strip()}
        if not norm_map:
            return 0

        modified_count = 0
        txt_files = sorted(path.rglob("*.txt"))

        for txt_file in txt_files:
            try:
                content = txt_file.read_text(encoding="utf-8").strip()
                if not content:
                    continue

                tags = [t.strip() for t in content.split(",") if t.strip()]
                changed = False
                updated_tags: List[str] = []
                seen_in_file = set()

                for tag in tags:
                    t_lower = tag.lower().strip()
                    # Check exact match or underscored match
                    replacement = norm_map.get(t_lower) or norm_map.get(t_lower.replace(" ", "_")) or norm_map.get(t_lower.replace("_", " "))
                    chosen = replacement if replacement is not None else tag

                    if chosen != tag:
                        changed = True

                    # Deduplicate in file if replacement mapped two tags to the same canonical
                    chosen_lower = chosen.lower()
                    if chosen_lower not in seen_in_file:
                        seen_in_file.add(chosen_lower)
                        updated_tags.append(chosen)

                if changed:
                    new_caption = ", ".join(updated_tags)
                    txt_file.write_text(new_caption, encoding="utf-8")
                    modified_count += 1

            except Exception as e:
                logger.error(f"Error modifying {txt_file}: {e}")

        # Re-index
        self.index(path)
        return modified_count
