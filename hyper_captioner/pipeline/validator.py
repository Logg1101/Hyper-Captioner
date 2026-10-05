"""
Semantic Validator with Conservative Quality Control and Safe Auto-Repair Boundary.

Evaluates candidate captions against StructuredVisualFacts, active CaptionMode contracts,
and TriggerConfig:
1. Detects HARD contradictions (mutually exclusive physical states -> ERROR -> REJECTED).
2. Detects POTENTIAL contradictions (nuanced coexisting states -> WARNING -> preserved).
3. Safely auto-repairs low-risk deterministic issues:
   - Case-insensitive trigger deduplication and placement (PREPEND, APPEND, WRAP, OMIT).
   - Duplicate token deduplication.
   - Punctuation, comma, and empty section normalization.
   - Subjective hype adjectives / quality buzzwords removal.
   - Conversational filler prose removal.
4. Strictly enforces safe auto-repair boundaries:
   - Inviolable user locks (`locked=True`) are NEVER removed during auto-repair.
   - The validator NEVER invents visual replacements for uncertain semantic facts.
"""

import re
from typing import ClassVar, List, Optional, Set, Tuple

from hyper_captioner.caption_modes.base import BaseCaptionMode
from hyper_captioner.core.types import (
    CaptionToken,
    FactItem,
    SemanticCategory,
    StructuredVisualFacts,
    TriggerConfig,
    TriggerPlacement,
    ValidationIssue,
    ValidationReport,
    ValidationStatus,
)
from hyper_captioner.pipeline.semantic_filter import SemanticFilter


class SemanticValidator:
    """
    Conservative Semantic Validator and Safe Auto-Repair Engine.
    """

    # HARD Contradictions: Mutually exclusive physical states (ERROR severity)
    HARD_CONTRADICTIONS: ClassVar[List[Tuple[str, re.Pattern, re.Pattern, str]]] = [
        # Standing vs Sitting
        (
            "standing_vs_sitting",
            re.compile(r"\bstanding\b", re.IGNORECASE),
            re.compile(r"\bsitting(\s+(on|in))?\b", re.IGNORECASE),
            "standing vs. sitting",
        ),
        # Standing vs Lying
        (
            "standing_vs_lying",
            re.compile(r"\bstanding\b", re.IGNORECASE),
            re.compile(r"\b(lying|laying)(\s+(on|in|down))?\b", re.IGNORECASE),
            "standing vs. lying",
        ),
        # Sitting vs Lying
        (
            "sitting_vs_lying",
            re.compile(r"\bsitting\b", re.IGNORECASE),
            re.compile(r"\b(lying|laying)(\s+(on|in|down))?\b", re.IGNORECASE),
            "sitting vs. lying",
        ),
        # Eyes open vs Eyes closed
        (
            "eyes_open_vs_closed",
            re.compile(r"\b(eyes?\s+open|open\s+eyes?)\b", re.IGNORECASE),
            re.compile(r"\b(eyes?\s+closed|closed\s+eyes?)\b", re.IGNORECASE),
            "eyes open vs. eyes closed",
        ),
        # Indoors vs Outdoors
        (
            "indoors_vs_outdoors",
            re.compile(r"\bindoor(s)?\b", re.IGNORECASE),
            re.compile(r"\boutdoor(s)?\b", re.IGNORECASE),
            "indoors vs. outdoors",
        ),
        # Day vs Night
        (
            "day_vs_night",
            re.compile(r"\b(day|daytime|sunny\s+day)\b", re.IGNORECASE),
            re.compile(r"\b(night|nighttime|night\s+sky)\b", re.IGNORECASE),
            "day vs. night",
        ),
    ]

    # POTENTIAL Contradictions: Nuanced coexisting states (WARNING severity, never deleted)
    POTENTIAL_CONTRADICTIONS: ClassVar[List[Tuple[str, re.Pattern, re.Pattern, str]]] = [
        # Looking at viewer vs Closed eyes
        (
            "looking_viewer_vs_closed_eyes",
            re.compile(r"\blooking\s+at\s+viewer\b", re.IGNORECASE),
            re.compile(r"\b(eyes?\s+closed|closed\s+eyes?)\b", re.IGNORECASE),
            "looking at viewer and closed eyes",
        ),
        # Smiling vs Neutral expression
        (
            "smiling_vs_neutral_expression",
            re.compile(r"\b(smiling|smile)\b", re.IGNORECASE),
            re.compile(r"\bneutral\s+expression\b", re.IGNORECASE),
            "smiling and neutral expression",
        ),
        # Front view vs Side view
        (
            "front_view_vs_side_view",
            re.compile(r"\bfront\s+view\b", re.IGNORECASE),
            re.compile(r"\bside\s+view\b", re.IGNORECASE),
            "front view and side view",
        ),
    ]

    # Conversational filler prose patterns
    FILLER_PROSE_PATTERNS: ClassVar[List[re.Pattern]] = [
        re.compile(
            r"\bhere\s+is\s+(an?\s+)?(image|picture|photo|illustration)\s+(of|showing)\s*",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bthe\s+(image|picture|photo|illustration)\s+(shows|depicts|features|displays)\s*",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bthis\s+is\s+(an?\s+)?(image|picture|photo|illustration)\s+(of|showing)\s*",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bthis\s+(image|picture|photo)\s+(shows|depicts|features|displays)\s*",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(in\s+)?this\s+(image|picture|photo),?\s*",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bin\s+the\s+(image|picture|photo),?\s*",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(an?)\s+(image|picture|photo|illustration)\s+of\s*",
            re.IGNORECASE,
        ),
    ]

    # Subjective quality hype buzzwords
    HYPE_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"\b(masterpiece|stunning|breathtaking|beautiful|gorgeous|incredible|"
        r"amazing|flawless|perfection|best\s+quality|top\s+quality|high\s+quality|"
        r"ultra\s+high\s+res|ultra-high\s+res|hyperrealistic|hyper-realistic|"
        r"ultra\s+realistic|ultra-realistic|absurdres|award\s+winning|"
        r"trending\s+on\s+artstation|unbelievable)\b",
        re.IGNORECASE,
    )

    def __init__(self):
        self._filter = SemanticFilter()

    def _collect_locked_texts(
        self,
        facts: Optional[StructuredVisualFacts],
        tokens: Optional[List[CaptionToken]],
    ) -> Set[str]:
        """Extract set of normalized texts for inviolable user locked facts."""
        locked_texts: Set[str] = set()

        if facts is not None:
            locked_fact_ids = {fact.id for fact in facts.all_facts() if fact.locked}
            for fact in facts.all_facts():
                if fact.locked:
                    locked_texts.add(fact.text.strip().lower())
        else:
            locked_fact_ids = set()

        if tokens is not None:
            for tok in tokens:
                if tok.transformation == "locked_override":
                    locked_texts.add(tok.text.strip().lower())
                elif tok.source_fact_ids and any(fid in locked_fact_ids for fid in tok.source_fact_ids):
                    locked_texts.add(tok.text.strip().lower())

        return locked_texts

    def _detect_hard_contradictions(
        self,
        caption: str,
        facts: Optional[StructuredVisualFacts],
    ) -> List[ValidationIssue]:
        """Detect mutually exclusive physical states within caption and against facts."""
        issues: List[ValidationIssue] = []

        # 1. Contradictions within the caption itself
        for _, p1, p2, desc in self.HARD_CONTRADICTIONS:
            m1 = p1.search(caption)
            m2 = p2.search(caption)
            if m1 and m2:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="HARD_CONTRADICTION",
                        message=f"Mutually exclusive states detected in caption: {desc}",
                        token=f"{m1.group(0)}, {m2.group(0)}",
                    )
                )

        # 2. Contradictions between caption and source facts
        if facts is not None:
            fact_texts = [f.text for f in facts.all_facts() if not f.is_uncertain]
            for _, p1, p2, desc in self.HARD_CONTRADICTIONS:
                f_has_1 = any(p1.search(ft) for ft in fact_texts)
                f_has_2 = any(p2.search(ft) for ft in fact_texts)
                c_has_1 = bool(p1.search(caption))
                c_has_2 = bool(p2.search(caption))

                if (f_has_1 and c_has_2 and not f_has_2) or (
                    f_has_2 and c_has_1 and not f_has_1
                ):
                    issues.append(
                        ValidationIssue(
                            severity="ERROR",
                            code="HARD_CONTRADICTION",
                            message=f"Caption contradicts source facts on mutually exclusive states: {desc}",
                            token=caption,
                        )
                    )

        return issues

    def _detect_potential_contradictions(self, caption: str) -> List[ValidationIssue]:
        """Detect nuanced coexisting states that warrant a WARNING but must NOT be deleted."""
        issues: List[ValidationIssue] = []

        for _, p1, p2, desc in self.POTENTIAL_CONTRADICTIONS:
            m1 = p1.search(caption)
            m2 = p2.search(caption)
            if m1 and m2:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        code="POTENTIAL_CONTRADICTION",
                        message=f"Potential contradictory states detected: {desc}",
                        token=f"{m1.group(0)}, {m2.group(0)}",
                    )
                )

        return issues

    def _check_category_leakage(
        self,
        caption: str,
        tokens: Optional[List[CaptionToken]],
        mode: Optional[BaseCaptionMode],
        locked_texts: Set[str],
    ) -> List[ValidationIssue]:
        """Check for severe category leakage under the active mode contract."""
        issues: List[ValidationIssue] = []
        if mode is None:
            return issues

        mode_name = mode.name.lower()

        # Style Mode: Check for character/clothing/appearance leaks
        if mode_name == "style":
            # Check segments in caption string
            segments = [s.strip() for s in caption.split(",") if s.strip()]
            for seg in segments:
                if seg.lower() in locked_texts:
                    continue
                if self._filter.is_style_or_rendering(seg):
                    continue
                if self._filter.is_content(seg):
                    issues.append(
                        ValidationIssue(
                            severity="ERROR",
                            code="CONTENT_LEAK_IN_STYLE",
                            message=f"Content leaked into style mode caption: '{seg}'",
                            token=seg,
                        )
                    )

            # Check structured tokens if provided
            if tokens:
                for tok in tokens:
                    if tok.text.strip().lower() in locked_texts:
                        continue
                    if self._filter.is_style_or_rendering(tok.text):
                        continue
                    if (
                        tok.primary_category in mode.exclude_categories
                        or self._filter.is_content(tok.text)
                    ):
                        if not any(
                            i.code == "CONTENT_LEAK_IN_STYLE" and i.token == tok.text
                            for i in issues
                        ):
                            issues.append(
                                ValidationIssue(
                                    severity="ERROR",
                                    code="CONTENT_LEAK_IN_STYLE",
                                    message=f"Content token leaked into style mode: '{tok.text}'",
                                    token=tok.text,
                                )
                            )

        # Other modes: Check token category exclusion
        elif tokens:
            for tok in tokens:
                if tok.text.strip().lower() in locked_texts:
                    continue
                # QUALITY category represents subjective hype which is handled via safe auto-repair
                if tok.primary_category == SemanticCategory.QUALITY:
                    continue
                if tok.primary_category in mode.exclude_categories:
                    issues.append(
                        ValidationIssue(
                            severity="ERROR",
                            code="CATEGORY_LEAKAGE",
                            message=f"Category '{tok.primary_category.value}' excluded in {mode.name} mode: '{tok.text}'",
                            token=tok.text,
                        )
                    )

        return issues

    def _clean_filler_prose(self, caption: str) -> Tuple[str, List[ValidationIssue]]:
        """Remove conversational filler phrases from caption."""
        issues: List[ValidationIssue] = []
        cleaned = caption

        for pattern in self.FILLER_PROSE_PATTERNS:
            match = pattern.search(cleaned)
            if match:
                filler_text = match.group(0)
                cleaned = pattern.sub("", cleaned)
                issues.append(
                    ValidationIssue(
                        severity="INFO",
                        code="FILLER_PROSE_REMOVED",
                        message=f"Removed conversational filler prose: '{filler_text.strip()}'",
                        token=filler_text.strip(),
                    )
                )

        return cleaned, issues

    def _repair_tokens_and_triggers(
        self,
        caption: str,
        trigger_cfg: Optional[TriggerConfig],
        locked_texts: Set[str],
    ) -> Tuple[str, List[ValidationIssue]]:
        """
        Perform deterministic auto-repairs:
        - Conversational filler prose removal.
        - Punctuation, extra commas, whitespace normalization.
        - Subjective hype adjectives / buzzwords removal (respecting locked items).
        - Duplicate token deduplication.
        - Trigger word case-insensitive deduplication and placement.
        """
        issues: List[ValidationIssue] = []

        # 1. Clean conversational filler prose
        cleaned_cap, filler_issues = self._clean_filler_prose(caption)
        issues.extend(filler_issues)

        # 2. Split into token segments by commas and normalize punctuation
        raw_segments = [s.strip() for s in re.split(r",+", cleaned_cap) if s.strip()]

        # 3. Clean subjective hype from non-locked segments
        cleaned_segments: List[str] = []
        for seg in raw_segments:
            seg_lower = seg.lower()
            if seg_lower in locked_texts:
                cleaned_segments.append(seg)
                continue

            # Check if segment matches hype
            if self.HYPE_PATTERN.search(seg):
                # Check if it's purely a hype phrase
                pure_hype_match = self.HYPE_PATTERN.fullmatch(seg.strip())
                if pure_hype_match:
                    issues.append(
                        ValidationIssue(
                            severity="INFO",
                            code="SUBJECTIVE_HYPE_REMOVED",
                            message=f"Removed subjective hype: '{seg}'",
                            token=seg,
                        )
                    )
                    continue

                # Strip hype words modifying an underlying noun
                stripped = self.HYPE_PATTERN.sub("", seg).strip()
                stripped = re.sub(r"\s+", " ", stripped).strip()
                if stripped:
                    issues.append(
                        ValidationIssue(
                            severity="INFO",
                            code="SUBJECTIVE_HYPE_REMOVED",
                            message=f"Stripped subjective hype: '{seg}' -> '{stripped}'",
                            token=seg,
                            suggested_repair=stripped,
                        )
                    )
                    cleaned_segments.append(stripped)
                else:
                    issues.append(
                        ValidationIssue(
                            severity="INFO",
                            code="SUBJECTIVE_HYPE_REMOVED",
                            message=f"Removed subjective hype: '{seg}'",
                            token=seg,
                        )
                    )
            else:
                cleaned_segments.append(seg)

        # 4. Trigger deduplication & placement
        if trigger_cfg is not None and trigger_cfg.word and trigger_cfg.word.strip():
            canonical_word = trigger_cfg.word.strip()
            canonical_lower = canonical_word.lower()
            case_sens = trigger_cfg.case_sensitive
            placement = trigger_cfg.placement

            def is_trig(t: str) -> bool:
                return (
                    t == canonical_word
                    if case_sens
                    else t.lower() == canonical_lower
                )

            trig_count = sum(1 for t in cleaned_segments if is_trig(t))
            non_trig_segments = [t for t in cleaned_segments if not is_trig(t)]

            # Deduplicate non-trigger segments (case-insensitive preserving first)
            deduped_non_trig: List[str] = []
            seen_tokens: Set[str] = set()
            for t in non_trig_segments:
                t_key = t.lower()
                if t_key in seen_tokens:
                    issues.append(
                        ValidationIssue(
                            severity="INFO",
                            code="DUPLICATE_TOKEN_REMOVED",
                            message=f"Removed duplicate token: '{t}'",
                            token=t,
                        )
                    )
                else:
                    seen_tokens.add(t_key)
                    deduped_non_trig.append(t)

            # Assemble based on placement
            if placement == TriggerPlacement.PREPEND:
                repaired_tokens = [canonical_word] + deduped_non_trig
            elif placement == TriggerPlacement.APPEND:
                repaired_tokens = deduped_non_trig + [canonical_word]
            elif placement == TriggerPlacement.WRAP:
                if deduped_non_trig:
                    repaired_tokens = (
                        [canonical_word] + deduped_non_trig + [canonical_word]
                    )
                else:
                    repaired_tokens = [canonical_word]
            elif placement == TriggerPlacement.OMIT:
                repaired_tokens = deduped_non_trig
            else:
                repaired_tokens = [canonical_word] + deduped_non_trig

            # Check if trigger was adjusted or relocated
            trigger_changed = False
            if trig_count > 1:
                trigger_changed = True
            elif trig_count == 0 and placement != TriggerPlacement.OMIT:
                trigger_changed = True
            elif placement == TriggerPlacement.OMIT and trig_count > 0:
                trigger_changed = True
            elif placement == TriggerPlacement.PREPEND:
                if not raw_segments or raw_segments[0].strip().lower() != canonical_lower or raw_segments[0].strip() != canonical_word:
                    trigger_changed = True
            elif placement == TriggerPlacement.APPEND:
                if not raw_segments or raw_segments[-1].strip().lower() != canonical_lower or raw_segments[-1].strip() != canonical_word:
                    trigger_changed = True
            elif placement == TriggerPlacement.WRAP:
                if (
                    len(raw_segments) < 2
                    or raw_segments[0].strip().lower() != canonical_lower
                    or raw_segments[-1].strip().lower() != canonical_lower
                ):
                    trigger_changed = True

            if trigger_changed:
                issues.append(
                    ValidationIssue(
                        severity="INFO",
                        code="TRIGGER_REPAIRED",
                        message=f"Repaired trigger placement/deduplication: '{canonical_word}'",
                        token=canonical_word,
                    )
                )

            repaired_caption = ", ".join(repaired_tokens)
        else:
            # No trigger configured: deduplicate tokens
            deduped_segments: List[str] = []
            seen_tokens: Set[str] = set()
            for t in cleaned_segments:
                t_key = t.lower()
                if t_key in seen_tokens:
                    issues.append(
                        ValidationIssue(
                            severity="INFO",
                            code="DUPLICATE_TOKEN_REMOVED",
                            message=f"Removed duplicate token: '{t}'",
                            token=t,
                        )
                    )
                else:
                    seen_tokens.add(t_key)
                    deduped_segments.append(t)

            repaired_tokens = deduped_segments
            repaired_caption = ", ".join(repaired_tokens)

        # Check for malformed punctuation / whitespace difference
        if not issues and repaired_caption != caption.strip():
            issues.append(
                ValidationIssue(
                    severity="INFO",
                    code="PUNCTUATION_CLEANED",
                    message="Cleaned malformed punctuation and empty sections",
                    suggested_repair=repaired_caption,
                )
            )

        return repaired_caption, issues

    def validate(
        self,
        caption: str,
        tokens: Optional[List[CaptionToken]] = None,
        mode: Optional[BaseCaptionMode] = None,
        facts: Optional[StructuredVisualFacts] = None,
        trigger_cfg: Optional[TriggerConfig] = None,
    ) -> ValidationReport:
        """
        Validate candidate caption and perform safe deterministic auto-repairs.

        Args:
            caption: Candidate caption string to validate.
            tokens: Optional list of CaptionToken objects from builder.
            mode: Optional active BaseCaptionMode contract.
            facts: Optional source StructuredVisualFacts.
            trigger_cfg: Optional TriggerConfig.

        Returns:
            ValidationReport with ValidationStatus, issues list, and repaired_caption.
        """
        all_issues: List[ValidationIssue] = []

        # 1. Identify inviolable locked user items
        locked_texts = self._collect_locked_texts(facts, tokens)

        # 2. Check HARD contradictions (Mutually exclusive -> ERROR)
        hard_issues = self._detect_hard_contradictions(caption, facts)
        all_issues.extend(hard_issues)

        # 3. Check POTENTIAL contradictions (Nuanced coexisting -> WARNING, preserved)
        potential_issues = self._detect_potential_contradictions(caption)
        all_issues.extend(potential_issues)

        # 4. Check Mode Contract Category Leakage (e.g. style mode content leak -> ERROR)
        leakage_issues = self._check_category_leakage(caption, tokens, mode, locked_texts)
        all_issues.extend(leakage_issues)

        # 5. Deterministic Safe Auto-Repair
        repaired_caption, repair_issues = self._repair_tokens_and_triggers(
            caption, trigger_cfg, locked_texts
        )
        all_issues.extend(repair_issues)

        # 6. Determine final validation status
        has_errors = any(i.severity == "ERROR" for i in all_issues)
        if has_errors:
            status = ValidationStatus.REJECTED
        else:
            # If no errors: check whether any repair took place
            has_repairs = (
                repaired_caption != caption.strip()
                or any(i.severity == "INFO" for i in all_issues)
            )
            if has_repairs:
                status = ValidationStatus.REPAIRED
            else:
                status = ValidationStatus.VALID

        return ValidationReport(
            status=status,
            issues=all_issues,
            repaired_caption=repaired_caption,
        )
