"""
Unit tests for Stage 1 Structured Visual Fact Extractor (Approach C Hybrid Dual-Parser).
Covers:
1. Primary JSON parser with syntax repair and code fence stripping.
2. Tagged semantic category block fallback parser.
3. Secondary fused comma-separated fallback parser.
4. Authoritative uncertainty handling (is_uncertain=True).
5. WD14 auxiliary tag enrichment, categorization, provenance, and deduplication.
6. Empty and malformed input graceful degradation.
7. JoyCaption Stage 1 prompt generation.
"""

import pytest
from hyper_captioner.core.types import FactItem, SemanticCategory, StructuredVisualFacts
from hyper_captioner.engines.joycaption import build_stage1_extraction_prompt
from hyper_captioner.pipeline.stage1_extractor import Stage1Extractor


@pytest.fixture
def extractor():
    return Stage1Extractor()


# ============================================================================
# 1. Clean JSON Extraction Tests
# ============================================================================


def test_extract_clean_json(extractor):
    raw_json = """{
        "identity": ["1girl", "solo"],
        "appearance": ["long black hair", "crimson red eyes"],
        "clothing": ["gothic maid dress", "white frilled apron"],
        "pose": ["standing upright"],
        "expression": ["gentle smile"],
        "environment": ["ornate library", "bookshelves"],
        "lighting": ["soft warm candlelight"]
    }"""

    facts = extractor.extract(raw_json)
    assert isinstance(facts, StructuredVisualFacts)
    assert facts.parse_method == "json"

    # Check categories
    identities = facts.get_category(SemanticCategory.IDENTITY)
    assert len(identities) == 2
    assert [f.text for f in identities] == ["1girl", "solo"]
    assert all(f.source == "joycaption" for f in identities)
    assert all(f.primary_category == SemanticCategory.IDENTITY for f in identities)

    clothing = facts.get_category(SemanticCategory.CLOTHING)
    assert len(clothing) == 2
    assert [f.text for f in clothing] == ["gothic maid dress", "white frilled apron"]

    # Verify ID formatting and confidence
    for item in facts.all_facts():
        assert item.id.startswith(f"{item.primary_category.value}_")
        assert not item.is_uncertain
        if len(item.text.split()) > 1:
            assert item.confidence == 0.92
        else:
            assert item.confidence == 0.88


# ============================================================================
# 2. JSON Syntax Repair & Code Fence Stripping Tests
# ============================================================================


def test_extract_json_with_code_fences_and_trailing_commas(extractor):
    raw_output = """```json
    {
        "appearance": ["silver ponytail", "amber eyes",],
        "clothing": ["oversized hoodie", "black leggings",],
        "lighting": ["neon rim lighting",],
    }
    ```"""

    facts = extractor.extract(raw_output)
    assert facts.parse_method == "json"
    appearance = facts.get_category(SemanticCategory.APPEARANCE)
    assert len(appearance) == 2
    assert [f.text for f in appearance] == ["silver ponytail", "amber eyes"]

    lighting = facts.get_category(SemanticCategory.LIGHTING)
    assert len(lighting) == 1
    assert lighting[0].text == "neon rim lighting"


def test_extract_json_with_single_quotes_and_unquoted_keys(extractor):
    raw_output = """
    Here is the visual fact analysis:
    {
        identity: ['1boy', 'solo'],
        'appearance': ['short spiky brown hair'],
        'clothing': ['leather motorcycle jacket', 'dark jeans',],
    }
    """

    facts = extractor.extract(raw_output)
    assert facts.parse_method == "json"

    identities = facts.get_category(SemanticCategory.IDENTITY)
    assert len(identities) == 2
    assert [f.text for f in identities] == ["1boy", "solo"]

    clothing = facts.get_category(SemanticCategory.CLOTHING)
    assert len(clothing) == 2
    assert [f.text for f in clothing] == ["leather motorcycle jacket", "dark jeans"]


# ============================================================================
# 3. Tagged Semantic Category Block Fallback Tests
# ============================================================================


def test_extract_tagged_semantic_blocks_bracket_headers(extractor):
    raw_blocks = """
    [IDENTITY]: 1girl, solo
    [APPEARANCE]: blonde twintails, blue eyes
    [CLOTHING]: sailor uniform, pleated skirt
    [ENVIRONMENT]: classroom, chalkboard
    [LIGHTING]: natural sunlight from window
    """

    facts = extractor.extract(raw_blocks)
    assert facts.parse_method == "tagged_block"

    appearance = facts.get_category(SemanticCategory.APPEARANCE)
    assert len(appearance) == 2
    assert [f.text for f in appearance] == ["blonde twintails", "blue eyes"]

    clothing = facts.get_category(SemanticCategory.CLOTHING)
    assert len(clothing) == 2
    assert [f.text for f in clothing] == ["sailor uniform", "pleated skirt"]

    env = facts.get_category(SemanticCategory.ENVIRONMENT)
    assert len(env) == 2
    assert [f.text for f in env] == ["classroom", "chalkboard"]


def test_extract_tagged_semantic_blocks_dash_and_colon_headers(extractor):
    raw_blocks = """
    - Identity: 1girl
    - Clothing: red kimono, floral sash
    - Pose: kneeling on tatami mat
    - Style: traditional ukiyo-e woodblock print
    """

    facts = extractor.extract(raw_blocks)
    assert facts.parse_method == "tagged_block"

    identity = facts.get_category(SemanticCategory.IDENTITY)
    assert len(identity) == 1
    assert identity[0].text == "1girl"

    pose = facts.get_category(SemanticCategory.POSE)
    assert len(pose) == 1
    assert pose[0].text == "kneeling on tatami mat"

    style = facts.get_category(SemanticCategory.STYLE)
    assert len(style) == 1
    assert style[0].text == "traditional ukiyo-e woodblock print"


# ============================================================================
# 4. Secondary Fused Fallback Tests
# ============================================================================


def test_extract_fused_fallback_comma_separated(extractor):
    raw_text = "1girl, solo, long purple hair, gothic dress, sitting, dark background"

    facts = extractor.extract(raw_text)
    assert facts.parse_method == "fused_fallback"

    identities = facts.get_category(SemanticCategory.IDENTITY)
    assert any(f.text == "1girl" for f in identities)

    appearance = facts.get_category(SemanticCategory.APPEARANCE)
    assert any(f.text == "long purple hair" for f in appearance)

    clothing = facts.get_category(SemanticCategory.CLOTHING)
    assert any(f.text == "gothic dress" for f in clothing)

    pose = facts.get_category(SemanticCategory.POSE)
    assert any(f.text == "sitting" for f in pose)


# ============================================================================
# 5. Authoritative Uncertainty Handling Tests
# ============================================================================


def test_extract_uncertainty_in_json(extractor):
    raw_json = """{
        "identity": ["1girl"],
        "clothing": ["red coat"],
        "uncertain": ["small pendant or brooch", "metallic keychain in pocket"],
        "uncertainty": ["partially obscured shoes"]
    }"""

    facts = extractor.extract(raw_json)
    assert facts.parse_method == "json"

    uncertain_items = facts.get_category(SemanticCategory.UNCERTAINTY)
    assert len(uncertain_items) == 3
    assert all(item.is_uncertain is True for item in uncertain_items)
    assert all(item.primary_category == SemanticCategory.UNCERTAINTY for item in uncertain_items)

    uncertain_texts = {item.text for item in uncertain_items}
    assert "small pendant or brooch" in uncertain_texts
    assert "metallic keychain in pocket" in uncertain_texts
    assert "partially obscured shoes" in uncertain_texts


def test_extract_uncertainty_in_tagged_blocks(extractor):
    raw_blocks = """
    [CLOTHING]: trench coat, fedora
    [UNCERTAINTY]: possibly wearing gloves, unclear object in background
    """

    facts = extractor.extract(raw_blocks)
    assert facts.parse_method == "tagged_block"

    uncertain_items = facts.get_category(SemanticCategory.UNCERTAINTY)
    assert len(uncertain_items) == 2
    assert all(item.is_uncertain is True for item in uncertain_items)
    assert all(item.primary_category == SemanticCategory.UNCERTAINTY for item in uncertain_items)


# ============================================================================
# 6. Text Cleaning & Provenance Tests
# ============================================================================


def test_text_cleaning_leading_articles_and_quotes(extractor):
    raw_json = """{
        "appearance": ["\\\"a dark ponytail\\\"", "'an emerald green eye'"],
        "clothing": ["the pleated wool skirt", "  a silk scarf  "],
        "objects": ["an ancient book", "the wooden staff"]
    }"""

    facts = extractor.extract(raw_json)
    appearance = [f.text for f in facts.get_category(SemanticCategory.APPEARANCE)]
    assert "dark ponytail" in appearance
    assert "emerald green eye" in appearance

    clothing = [f.text for f in facts.get_category(SemanticCategory.CLOTHING)]
    assert "pleated wool skirt" in clothing
    assert "silk scarf" in clothing

    objects = [f.text for f in facts.get_category(SemanticCategory.OBJECTS)]
    assert "ancient book" in objects
    assert "wooden staff" in objects


# ============================================================================
# 7. WD14 Auxiliary Tag Grounding & Enrichment Tests
# ============================================================================


def test_wd14_enrichment_and_provenance(extractor):
    joy_json = """{
        "identity": ["1girl", "solo"],
        "clothing": ["maid dress"]
    }"""

    wd14_predictions = [
        # Character tags (cat_id == 4)
        ("1girl", 0.99, 4),  # Demographic count -> IDENTITY (dup of JoyCaption)
        ("solo", 0.98, 4),   # Demographic count -> IDENTITY (dup of JoyCaption)
        ("hatsune_miku", 0.88, 4),  # Specific unconfirmed character name -> FILTER OUT
        # General tags (cat_id == 0)
        ("blue_hair", 0.92, 0),        # Appearance
        ("twin_tails", 0.89, 0),       # Appearance
        ("white_apron", 0.85, 0),      # Clothing
        ("maid_headdress", 0.83, 0),   # Clothing
        ("standing", 0.91, 0),         # Pose
        ("smile", 0.80, 0),            # Expression
        ("cowboy_shot", 0.78, 0),      # Composition
        ("indoors", 0.88, 0),          # Environment
        ("soft_lighting", 0.72, 0),    # Lighting
    ]

    facts = extractor.extract(joy_json, wd14_tags=wd14_predictions)

    # 1. Check unconfirmed character name filtered out
    all_texts = [f.text for f in facts.all_facts()]
    assert "hatsune miku" not in all_texts
    assert "hatsune_miku" not in all_texts

    # 2. Check no duplicate 1girl / solo
    identities = facts.get_category(SemanticCategory.IDENTITY)
    assert len(identities) == 2
    assert [f.text for f in identities] == ["1girl", "solo"]
    assert all(f.source == "joycaption" for f in identities)

    # 3. Check WD14 items categorized with provenance
    appearance = facts.get_category(SemanticCategory.APPEARANCE)
    app_texts = [f.text for f in appearance]
    assert "blue hair" in app_texts
    assert "twin tails" in app_texts
    for item in appearance:
        if item.text in {"blue hair", "twin tails"}:
            assert item.source == "wd14"
            assert item.confidence in {0.92, 0.89}

    clothing = facts.get_category(SemanticCategory.CLOTHING)
    clo_texts = [f.text for f in clothing]
    assert "maid dress" in clo_texts  # from joycaption
    assert "white apron" in clo_texts  # from wd14
    assert "maid headdress" in clo_texts  # from wd14

    pose = facts.get_category(SemanticCategory.POSE)
    assert any(f.text == "standing" and f.source == "wd14" for f in pose)

    expression = facts.get_category(SemanticCategory.EXPRESSION)
    assert any(f.text == "smile" and f.source == "wd14" for f in expression)

    composition = facts.get_category(SemanticCategory.COMPOSITION)
    assert any(f.text == "cowboy shot" and f.source == "wd14" for f in composition)

    environment = facts.get_category(SemanticCategory.ENVIRONMENT)
    assert any(f.text == "indoors" and f.source == "wd14" for f in environment)

    lighting = facts.get_category(SemanticCategory.LIGHTING)
    assert any(f.text == "soft lighting" and f.source == "wd14" for f in lighting)


def test_wd14_deduplication_exact_match(extractor):
    joy_json = """{
        "appearance": ["black hair", "red eyes"],
        "clothing": ["school uniform"]
    }"""

    wd14_predictions = [
        ("black_hair", 0.95, 0),       # Exact match with "black hair" -> Skip
        ("red_eyes", 0.93, 0),         # Exact match with "red eyes" -> Skip
        ("school_uniform", 0.91, 0),   # Exact match with "school uniform" -> Skip
        ("pleated_skirt", 0.82, 0),    # New tag -> Add
    ]

    facts = extractor.extract(joy_json, wd14_tags=wd14_predictions)
    appearance = facts.get_category(SemanticCategory.APPEARANCE)
    assert len(appearance) == 2
    assert all(f.source == "joycaption" for f in appearance)

    clothing = facts.get_category(SemanticCategory.CLOTHING)
    assert len(clothing) == 2
    sources = {f.text: f.source for f in clothing}
    assert sources["school uniform"] == "joycaption"
    assert sources["pleated skirt"] == "wd14"


# ============================================================================
# 8. Empty & Malformed Input Handling
# ============================================================================


def test_extract_empty_or_whitespace_input(extractor):
    facts1 = extractor.extract("")
    assert isinstance(facts1, StructuredVisualFacts)
    assert len(facts1.all_facts()) == 0

    facts2 = extractor.extract("   \n\t  ")
    assert isinstance(facts2, StructuredVisualFacts)
    assert len(facts2.all_facts()) == 0


def test_extract_none_like_input(extractor):
    facts = extractor.extract("None")
    assert isinstance(facts, StructuredVisualFacts)
    assert len(facts.all_facts()) == 0


# ============================================================================
# 9. JoyCaption Stage 1 Prompt Construction Tests
# ============================================================================


def test_build_stage1_extraction_prompt():
    prompt = build_stage1_extraction_prompt()
    assert isinstance(prompt, str)
    assert "JSON" in prompt
    assert "identity" in prompt
    assert "appearance" in prompt
    assert "clothing" in prompt
    assert "uncertain" in prompt or "uncertainty" in prompt
    assert "only visually verifiable facts" in prompt.lower() or "factual" in prompt.lower()

    # With mode instructions appended
    prompt_custom = build_stage1_extraction_prompt(mode_instructions="Focus heavily on fabric textures and weave.")
    assert "Focus heavily on fabric textures and weave." in prompt_custom
