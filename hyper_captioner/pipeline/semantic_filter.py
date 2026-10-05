"""
Meaning-Based Content vs. Style Discriminator & Semantic Filter.

Enforces the foundational principle:
HyperCaptioner distinguishes between what an image contains (content)
and how the image renders that content (style) based on semantic meaning.
Categories serve as routing defaults, not absolute semantic truth.
"""

import re
from dataclasses import dataclass, field
from typing import ClassVar

from hyper_captioner.caption_modes.base import BaseCaptionMode
from hyper_captioner.core.types import (
    FactItem,
    SemanticCategory,
    StructuredVisualFacts,
    TriggerConfig,
)


@dataclass
class SemanticFilterResult:
    """
    Result of semantic filtering containing accepted facts
    and rejected facts with detailed rejection reasons.
    """

    accepted: list[FactItem] = field(default_factory=list)
    rejected: list[tuple[FactItem, str]] = field(default_factory=list)

    @property
    def accepted_facts(self) -> list[FactItem]:
        return self.accepted

    @property
    def rejected_facts(self) -> list[tuple[FactItem, str]]:
        return self.rejected


class SemanticFilter:
    """
    Meaning-Based Content vs. Style Discriminator and Multi-Mode Semantic Filter.

    Enforces:
    1. Inviolable user locks (`locked=True`) take absolute precedence over all exclusions.
    2. Authoritative uncertainty (`is_uncertain=True` or `SemanticCategory.UNCERTAINTY`)
       is strictly rejected with reason 'uncertainty'.
    3. Subjective quality hype ('masterpiece', 'stunning', etc.) is rejected with 'subjective_hype'.
    4. Redundant trigger word and stable trait absorption is pruned with 'redundant_trigger_trait'.
    5. Meaning-based Content vs. Style discrimination:
       - Style mode strips content leaks ('pink hair', 'maid dress') while preserving
         rendering techniques, material response, and texture fidelity ('individually resolved hair strands').
       - Character mode suppresses esoteric 3D/rendering jargon ('octane render', 'subsurface scattering').
       - Outfit, Pose, and Concept modes respect mode-specific focus and exclude irrelevant clutter.
    """

    # Subjective hype adjectives and quality buzzwords
    HYPE_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"\b(masterpiece|stunning|breathtaking|beautiful|gorgeous|incredible|"
        r"amazing|flawless|perfection|best\s+quality|top\s+quality|high\s+quality|"
        r"ultra\s+high\s+res|ultra-high\s+res|hyperrealistic|hyper-realistic|"
        r"ultra\s+realistic|ultra-realistic|absurdres|award\s+winning|"
        r"trending\s+on\s+artstation|unbelievable)\b",
        re.IGNORECASE,
    )

    # Esoteric 3D engine / render passes jargon to suppress in character/general modes
    RENDERING_JARGON_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"\b(octane\s+render|octane|unreal\s+engine(\s*[45])?|ue[45]|raytracing|"
        r"ray\s*tracing|ray-tracing|raytraced|ray-traced|subsurface\s+scattering|"
        r"cycles\s+render|blender\s+cycles|v-ray|vray|redshift(\s+render)?|arnold\s+render|"
        r"lumion|rendered\s+in\s+blender|zbrush(\s+sculpt)?|"
        r"cinema\s+4d|c4d|cg\s+render|volumetric\s+rendering)\b",
        re.IGNORECASE,
    )

    # Patterns indicating Style, Rendering, Lighting Physics, Material Response, or Texture Fidelity
    STYLE_RENDERING_PATTERNS: ClassVar[list[str]] = [
        # Shading & Line Art Techniques
        r"\b(cel|cell)\s+shading\b",
        r"\b(soft|flat|hard|smooth|gradient)\s+shading\b",
        r"\bclean\s+line\s*art\b",
        r"\bline\s*art\b",
        r"\bline\s*work\b",
        r"\b(clean|thick|thin|sketchy|delicate|crisp|bold)\s+lines?\b",
        r"\b(cross\s*hatching|hatching|stippling|screentone|halftone)\b",
        r"\b(chiaroscuro|impasto|brushwork|brush\s*strokes?|palette\s*knife)\b",
        r"\bwatercolor(\s+wash)?\b",
        r"\bwatercolour(\s+wash)?\b",
        r"\b(ink|color)\s+wash\b",
        r"\b(oil|acrylic|gouache|digital)\s+(painting|illustration)\b",
        r"\b(vector\s*art|pixel\s*art|pencil\s*sketch|charcoal\s*sketch)\b",
        r"\b(flat\s*colors?|monochrome|grayscale|sepia)\b",
        r"\b(anime|manga|comic\s*book|graphic\s*novel|pop\s*art|minimalist|surrealist)\s+style\b",
        r"\b(photorealistic|stylized|impressionist)\b",
        # Surface & Material Reflectance Response
        r"\b(diffused\s+)?specular(\s+(highlights?|reflections?|response))?\b",
        r"\b(specular|subtle|blown-out|surface|edge|rim)\s+highlights?\b",
        r"\b(metallic|surface)\s+reflections?\b",
        r"\b(matte|glossy|rough|lustrous|polished|cracked|weathered)\s+(\w+\s+)*surface\b",
        r"\bsurface\s+(texture|response|reflectance|finish)\b",
        r"\b(subsurface|subsurface-like)(\s+(scattering|skin|glow|rendering))?\b",
        r"\b(sheen|luster|lustre|iridescence|iridescent|fresnel|anisotropic|albedo)\b",
        r"\bambient\s+occlusion\b",
        # Micro-Fidelity, Texture Resolution, and Surface Rendering
        r"\bindividually\s+(resolved|detailed|rendered)(\s+\w+)*\s+strands?\b",
        r"\b(fine|intricate|detailed|resolved)\s+strands?\b",
        r"\bskin\s+rendering\b",
        r"\btexture\s+(fidelity|quality|resolution)\b",
        r"\b(film|paper)\s+grain\b",
        # Lighting Physics & Optical Properties
        r"\b(rim|volumetric|dappled|diffuse|ambient|split|bounce)\s+light(ing)?\b",
        r"\b(light\s+shafts|god\s+rays|crepuscular\s+rays|backlighting|backlit)\b",
        r"\b(harsh|soft|cast)\s+shadows?\b",
        r"\b(lens\s+flare|bloom|chromatic\s+aberration|bokeh|depth\s+of\s+field|vignette)\b",
        # Composition & Camera Framing
        r"\b(cowboy\s+shot|close-up|extreme\s+close-up|wide\s+shot|medium\s+shot|full\s+body\s+shot)\b",
        r"\b(dutch\s+angle|low\s+angle|high\s+angle|birds?-eye\s+view|worms?-eye\s+view|overhead\s+shot|macro\s+shot)\b",
        r"\b(rule\s+of\s+thirds|dynamic\s+perspective|isometric\s+view)\b",
    ]

    # Patterns indicating Content (Subject identity, hair/eye color, garments, props, actions)
    CONTENT_PATTERNS: ClassVar[list[str]] = [
        # Character counts / demographics / identities
        r"\b(\d+girls?|\d+boys?|solo|multiple\s+girls|multiple\s+boys|woman|man|female|male|girl|boy|child|baby|person|human|character)\b",
        # Hair / Eye color + Anatomy
        r"\b(pink|blue|blonde|blond|brown|black|white|silver|red|green|purple|orange|yellow|golden|grey|gray|aqua|cyan|navy)\s+(hair|eyes?)\b",
        r"\b(hair|eyes?|face|lips?|mouth|nose|ears?|elf\s+ears?|pointed\s+ears?|cheeks?|neck|chest|breasts?|cleavage|navel|belly|abs|waist|hips?|thighs?|legs?|feet|foot|arms?|hands?|fingers?|claws?|wings?|horns?|tails?|halos?|fur|scales?|freckles?|scars?|tattoos?)\b",
        r"\b(ponytail|twintails|twin\s+tails|braid|bun|bangs|bob\s+cut|ahoge)\b",
        # Clothing / Garments / Accessories
        r"\b(dress|maid\s+dress|sundress|gown|skirt|pleated\s+skirt|miniskirt|jacket|leather\s+jacket|denim\s+jacket|coat|trench\s+coat|blazer|hoodie|sweater|cardigan|shirt|t-shirt|blouse|corset|vest|pants|trousers|jeans|shorts|leggings|tights|stockings|thighhighs|socks|boots|shoes|sneakers|sandals|heels|gloves|mittens|hat|cap|beret|beanie|helmet|mask|scarf|cape|cloak|robe|kimono|yukata|swimsuit|bikini|uniform|school\s+uniform|sailor\s+suit|suit|tuxedo|armor|belt|buckle|collar|choker|necklace|earrings|bracelet|ring|ribbon|bow|hair\s+ribbon|tie|necktie|bowtie|sash|apron|veil)\b",
        # Props / Scene Objects
        r"\b(sword|katana|blade|shield|gun|pistol|rifle|staff|wand|bow\s+and\s+arrow|knife|dagger|weapon|book|grimoire|tome|scroll|cup|mug|bottle|food|flower|bouquet|rose|guitar|instrument|phone|camera|umbrella|parasol|bag|backpack|purse|chair|bed|table|desk|car|vehicle|motorcycle|bike|boat)\b",
        # Interactive Actions / Poses
        r"\b(holding|carrying|wielding|wearing|sitting|standing|kneeling|lying|crouching|leaning|walking|running|jumping|dancing|looking\s+at|looking\s+away|smiling|laughing|crying|blushing|winking)\b",
    ]

    def __init__(self):
        self._style_rendering_regexes = [
            re.compile(p, re.IGNORECASE) for p in self.STYLE_RENDERING_PATTERNS
        ]
        self._content_regexes = [
            re.compile(p, re.IGNORECASE) for p in self.CONTENT_PATTERNS
        ]

    def is_subjective_hype(self, text: str) -> bool:
        """Check if text contains subjective quality hype buzzwords."""
        return bool(self.HYPE_PATTERN.search(text))

    def is_rendering_jargon(self, text: str) -> bool:
        """Check if text contains esoteric 3D engine or render pass jargon."""
        return bool(self.RENDERING_JARGON_PATTERN.search(text))

    def is_style_or_rendering(self, text: str) -> bool:
        """
        Check if text describes rendering method, surface response,
        lighting physics, or texture fidelity based on meaning.
        """
        return any(regex.search(text) for regex in self._style_rendering_regexes)

    def is_content(self, text: str) -> bool:
        """
        Check if text describes what the image contains (character identity,
        clothing garment, props, bodily features).
        Style/rendering phrases (e.g. individually resolved strands, surface response)
        are never classified as content.
        """
        if self.is_style_or_rendering(text):
            return False
        return any(regex.search(text) for regex in self._content_regexes)

    def filter_facts(
        self,
        mode: BaseCaptionMode,
        facts: StructuredVisualFacts | list[FactItem],
        trigger_cfg: TriggerConfig | None = None,
    ) -> SemanticFilterResult:
        """
        Filter extracted visual facts based on semantic meaning and mode contracts.

        Args:
            mode: Caption mode contract defining category boundaries and focus.
            facts: Extracted visual facts (StructuredVisualFacts or list of FactItem).
            trigger_cfg: Optional trigger word configuration.

        Returns:
            SemanticFilterResult containing accepted facts and rejected facts with reasons.
        """
        if isinstance(facts, StructuredVisualFacts):
            fact_list = facts.all_facts()
        elif isinstance(facts, list):
            fact_list = facts
        else:
            fact_list = []

        accepted: list[FactItem] = []
        rejected: list[tuple[FactItem, str]] = []
        seen_ids: set[str] = set()

        mode_name = mode.name.lower() if hasattr(mode, "name") else ""

        for fact in fact_list:
            fid = fact.id if fact.id else f"{fact.primary_category}:{fact.text}"
            if fid in seen_ids:
                continue
            seen_ids.add(fid)

            # 1. Inviolable User Lock priority (locked=True always accepted)
            if fact.locked:
                accepted.append(fact)
                continue

            # 2. Authoritative Uncertainty exclusion
            if (
                fact.is_uncertain
                or fact.primary_category == SemanticCategory.UNCERTAINTY
            ):
                rejected.append((fact, "uncertainty"))
                continue

            # 3. Subjective Quality Hype exclusion
            if (
                fact.primary_category == SemanticCategory.QUALITY
                or self.is_subjective_hype(fact.text)
            ):
                rejected.append((fact, "subjective_hype"))
                continue

            # 4. Trigger Trait Redundancy & Deduplication
            if trigger_cfg and trigger_cfg.word:
                trig_word = trigger_cfg.word.strip()
                fact_text = fact.text.strip()
                is_exact_match = (
                    fact_text == trig_word
                    if trigger_cfg.case_sensitive
                    else fact_text.lower() == trig_word.lower()
                )
                if is_exact_match:
                    rejected.append((fact, "redundant_trigger_trait"))
                    continue

                if trigger_cfg.absorb_stable_traits and fact.is_stable:
                    rejected.append((fact, "redundant_trigger_trait"))
                    continue

            # 5. Meaning-Based Discrimination & Mode-Specific Gating
            if mode_name == "style":
                # Style mode: strictly exclude content domain facts, preserve rendering
                if self.is_style_or_rendering(fact.text):
                    accepted.append(fact)
                    continue

                if self.is_content(fact.text):
                    rejected.append((fact, "content_leak_in_style"))
                    continue

                # Fallback to category contract for neutral/unclassified facts
                if mode.is_category_allowed(fact.primary_category):
                    accepted.append(fact)
                else:
                    is_content_cat = fact.primary_category in {
                        SemanticCategory.IDENTITY,
                        SemanticCategory.APPEARANCE,
                        SemanticCategory.CLOTHING,
                        SemanticCategory.OBJECTS,
                    }
                    reason = "content_leak_in_style" if is_content_cat else "excluded_category"
                    rejected.append((fact, reason))

            elif mode_name == "character":
                # Character mode: suppress esoteric 3D/rendering jargon
                if self.is_rendering_jargon(fact.text):
                    rejected.append((fact, "rendering_jargon"))
                    continue

                if mode.is_category_allowed(fact.primary_category):
                    accepted.append(fact)
                else:
                    rejected.append((fact, "excluded_category"))

            else:
                # Outfit, Pose, Concept, and other general modes
                if mode.is_category_allowed(fact.primary_category):
                    accepted.append(fact)
                else:
                    rejected.append((fact, "excluded_category"))

        return SemanticFilterResult(accepted=accepted, rejected=rejected)
