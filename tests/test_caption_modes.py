"""
Unit tests for BaseCaptionMode and the 5 Core Caption Modes:
CharacterMode, StyleMode, OutfitMode, PoseMode, ConceptMode,
as well as dynamic registry lookup and category contracts.
"""

import pytest
from hyper_captioner.core.types import (
    CaptionFormat,
    CaptionModeType,
    CaptionToken,
    FactItem,
    SemanticCategory,
    StructuredVisualFacts,
    TriggerConfig,
    TriggerPlacement,
)


def test_registry_import_and_discovery():
    from hyper_captioner.caption_modes.registry import (
        get_caption_mode,
        list_caption_modes,
    )

    modes = list_caption_modes()
    assert isinstance(modes, list)
    expected = {"character", "style", "outfit", "pose", "concept"}
    assert expected.issubset(set(modes))


def test_registry_get_mode_case_insensitive_and_enum():
    from hyper_captioner.caption_modes.character import CharacterMode
    from hyper_captioner.caption_modes.registry import get_caption_mode

    # String exact
    mode1 = get_caption_mode("character")
    assert isinstance(mode1, CharacterMode)
    assert mode1.name == "character"

    # String uppercase
    mode2 = get_caption_mode("CHARACTER")
    assert isinstance(mode2, CharacterMode)

    # CaptionModeType enum
    mode3 = get_caption_mode(CaptionModeType.CHARACTER)
    assert isinstance(mode3, CharacterMode)

    # Unknown mode raises KeyError
    with pytest.raises((KeyError, ValueError)):
        get_caption_mode("nonexistent_mode")


def test_registry_custom_registration():
    from hyper_captioner.caption_modes.base import BaseCaptionMode
    from hyper_captioner.caption_modes.registry import (
        get_caption_mode,
        list_caption_modes,
        register_caption_mode,
    )

    class CustomMode(BaseCaptionMode):
        def __init__(self):
            super().__init__(
                name="custom_test",
                description="Custom test mode",
                include_categories={SemanticCategory.CONCEPT},
                exclude_categories={SemanticCategory.IDENTITY},
            )

        def build_extraction_instructions(self) -> str:
            return "Custom instructions"

        def filter_facts(self, facts: StructuredVisualFacts, trigger_cfg: TriggerConfig):
            return [f for f in facts.all_facts() if self.is_category_allowed(f.primary_category)]

        def format_tokens(self, tokens, format_type):
            return ", ".join(t.text for t in tokens)

    custom_instance = CustomMode()
    register_caption_mode(custom_instance)

    assert "custom_test" in list_caption_modes()
    retrieved = get_caption_mode("custom_test")
    assert retrieved.name == "custom_test"


def test_character_mode_contracts_and_filtering():
    from hyper_captioner.caption_modes.registry import get_caption_mode

    mode = get_caption_mode("character")
    assert mode.name == "character"
    assert mode.description

    # Category contracts
    assert SemanticCategory.IDENTITY in mode.include_categories
    assert SemanticCategory.APPEARANCE in mode.include_categories
    assert SemanticCategory.EXPRESSION in mode.include_categories
    assert SemanticCategory.POSE in mode.include_categories
    assert SemanticCategory.COMPOSITION in mode.include_categories
    assert SemanticCategory.CAMERA in mode.include_categories
    assert SemanticCategory.ENVIRONMENT in mode.include_categories
    assert SemanticCategory.LIGHTING in mode.include_categories
    assert SemanticCategory.CLOTHING in mode.include_categories

    assert SemanticCategory.STYLE in mode.exclude_categories
    assert SemanticCategory.RENDERING in mode.exclude_categories
    assert SemanticCategory.QUALITY in mode.exclude_categories
    assert SemanticCategory.UNCERTAINTY in mode.exclude_categories

    # Vision model extraction instructions
    instructions = mode.build_extraction_instructions()
    assert isinstance(instructions, str)
    assert len(instructions) > 20

    # Fact filtering
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.IDENTITY: [
                FactItem(id="id_1", text="1girl", primary_category=SemanticCategory.IDENTITY),
            ],
            SemanticCategory.APPEARANCE: [
                FactItem(id="app_1", text="blue eyes", primary_category=SemanticCategory.APPEARANCE, is_stable=True),
            ],
            SemanticCategory.CLOTHING: [
                FactItem(id="clo_1", text="leather jacket", primary_category=SemanticCategory.CLOTHING),
            ],
            SemanticCategory.RENDERING: [
                FactItem(id="ren_1", text="octane render", primary_category=SemanticCategory.RENDERING),
            ],
            SemanticCategory.UNCERTAINTY: [
                FactItem(id="unc_1", text="maybe silk", primary_category=SemanticCategory.UNCERTAINTY, is_uncertain=True),
            ],
            SemanticCategory.STYLE: [
                FactItem(id="sty_locked", text="anime style", primary_category=SemanticCategory.STYLE, locked=True),
            ],
        }
    )

    # Filtering without trait absorption
    trigger_cfg = TriggerConfig(word="Heroine")
    filtered = mode.filter_facts(facts, trigger_cfg)
    filtered_texts = [f.text for f in filtered]

    assert "1girl" in filtered_texts
    assert "blue eyes" in filtered_texts
    assert "leather jacket" in filtered_texts
    assert "octane render" not in filtered_texts
    assert "maybe silk" not in filtered_texts
    # Inviolable lock override
    assert "anime style" in filtered_texts

    # Filtering with trait absorption
    trigger_absorb = TriggerConfig(word="Heroine", absorb_stable_traits=True)
    filtered_absorbed = mode.filter_facts(facts, trigger_absorb)
    absorbed_texts = [f.text for f in filtered_absorbed]
    # Stable trait "blue eyes" should be absorbed
    assert "blue eyes" not in absorbed_texts
    assert "1girl" in absorbed_texts
    assert "leather jacket" in absorbed_texts


def test_style_mode_contracts_and_filtering():
    from hyper_captioner.caption_modes.registry import get_caption_mode

    mode = get_caption_mode("style")
    assert mode.name == "style"

    # Category contracts
    assert SemanticCategory.RENDERING in mode.include_categories
    assert SemanticCategory.STYLE in mode.include_categories
    assert SemanticCategory.LIGHTING in mode.include_categories
    assert SemanticCategory.MATERIAL in mode.include_categories
    assert SemanticCategory.TEXTURE in mode.include_categories
    assert SemanticCategory.COLOR in mode.include_categories
    assert SemanticCategory.COMPOSITION in mode.include_categories
    assert SemanticCategory.CAMERA in mode.include_categories

    assert SemanticCategory.IDENTITY in mode.exclude_categories
    assert SemanticCategory.APPEARANCE in mode.exclude_categories
    assert SemanticCategory.CLOTHING in mode.exclude_categories
    assert SemanticCategory.OBJECTS in mode.exclude_categories
    assert SemanticCategory.UNCERTAINTY in mode.exclude_categories

    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.APPEARANCE: [
                FactItem(id="app_1", text="blonde hair", primary_category=SemanticCategory.APPEARANCE),
            ],
            SemanticCategory.CLOTHING: [
                FactItem(id="clo_1", text="red dress", primary_category=SemanticCategory.CLOTHING),
                FactItem(id="clo_2", text="locked ribbon", primary_category=SemanticCategory.CLOTHING, locked=True),
            ],
            SemanticCategory.RENDERING: [
                FactItem(id="ren_1", text="watercolor wash", primary_category=SemanticCategory.RENDERING),
            ],
            SemanticCategory.LIGHTING: [
                FactItem(id="lig_1", text="dramatic rim lighting", primary_category=SemanticCategory.LIGHTING),
            ],
            SemanticCategory.TEXTURE: [
                FactItem(id="tex_1", text="canvas grain", primary_category=SemanticCategory.TEXTURE),
            ],
        }
    )

    filtered = mode.filter_facts(facts, TriggerConfig(word="WatercolorStyle"))
    filtered_texts = [f.text for f in filtered]

    assert "watercolor wash" in filtered_texts
    assert "dramatic rim lighting" in filtered_texts
    assert "canvas grain" in filtered_texts
    assert "blonde hair" not in filtered_texts
    assert "red dress" not in filtered_texts
    # Inviolable lock
    assert "locked ribbon" in filtered_texts


def test_outfit_mode_contracts_and_filtering():
    from hyper_captioner.caption_modes.registry import get_caption_mode

    mode = get_caption_mode("outfit")
    assert mode.name == "outfit"

    # Category contracts
    assert SemanticCategory.CLOTHING in mode.include_categories
    assert SemanticCategory.MATERIAL in mode.include_categories
    assert SemanticCategory.TEXTURE in mode.include_categories
    assert SemanticCategory.COLOR in mode.include_categories
    assert SemanticCategory.OBJECTS in mode.include_categories

    assert SemanticCategory.IDENTITY in mode.exclude_categories
    assert SemanticCategory.APPEARANCE in mode.exclude_categories
    assert SemanticCategory.ENVIRONMENT in mode.exclude_categories
    assert SemanticCategory.POSE in mode.exclude_categories
    assert SemanticCategory.UNCERTAINTY in mode.exclude_categories

    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.IDENTITY: [
                FactItem(id="id_1", text="1girl", primary_category=SemanticCategory.IDENTITY),
            ],
            SemanticCategory.APPEARANCE: [
                FactItem(id="app_1", text="green eyes", primary_category=SemanticCategory.APPEARANCE),
            ],
            SemanticCategory.CLOTHING: [
                FactItem(id="clo_1", text="pleated skirt", primary_category=SemanticCategory.CLOTHING),
                FactItem(id="clo_2", text="sailor collar", primary_category=SemanticCategory.CLOTHING),
            ],
            SemanticCategory.MATERIAL: [
                FactItem(id="mat_1", text="cotton fabric", primary_category=SemanticCategory.MATERIAL),
            ],
            SemanticCategory.ENVIRONMENT: [
                FactItem(id="env_1", text="classroom background", primary_category=SemanticCategory.ENVIRONMENT),
            ],
            SemanticCategory.POSE: [
                FactItem(id="pos_locked", text="sitting", primary_category=SemanticCategory.POSE, locked=True),
            ],
        }
    )

    filtered = mode.filter_facts(facts, TriggerConfig(word="SchoolUniform"))
    filtered_texts = [f.text for f in filtered]

    assert "pleated skirt" in filtered_texts
    assert "sailor collar" in filtered_texts
    assert "cotton fabric" in filtered_texts
    assert "1girl" not in filtered_texts
    assert "green eyes" not in filtered_texts
    assert "classroom background" not in filtered_texts
    # Inviolable lock
    assert "sitting" in filtered_texts


def test_pose_mode_contracts_and_filtering():
    from hyper_captioner.caption_modes.registry import get_caption_mode

    mode = get_caption_mode("pose")
    assert mode.name == "pose"

    # Category contracts
    assert SemanticCategory.POSE in mode.include_categories
    assert SemanticCategory.COMPOSITION in mode.include_categories
    assert SemanticCategory.CAMERA in mode.include_categories
    assert SemanticCategory.OBJECTS in mode.include_categories

    assert SemanticCategory.IDENTITY in mode.exclude_categories
    assert SemanticCategory.APPEARANCE in mode.exclude_categories
    assert SemanticCategory.CLOTHING in mode.exclude_categories
    assert SemanticCategory.STYLE in mode.exclude_categories
    assert SemanticCategory.ENVIRONMENT in mode.exclude_categories
    assert SemanticCategory.UNCERTAINTY in mode.exclude_categories

    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.POSE: [
                FactItem(id="pos_1", text="crossed arms", primary_category=SemanticCategory.POSE),
                FactItem(id="pos_2", text="head tilt", primary_category=SemanticCategory.POSE),
            ],
            SemanticCategory.COMPOSITION: [
                FactItem(id="comp_1", text="cowboy shot", primary_category=SemanticCategory.COMPOSITION),
            ],
            SemanticCategory.CAMERA: [
                FactItem(id="cam_1", text="low angle", primary_category=SemanticCategory.CAMERA),
            ],
            SemanticCategory.CLOTHING: [
                FactItem(id="clo_1", text="trench coat", primary_category=SemanticCategory.CLOTHING),
            ],
            SemanticCategory.IDENTITY: [
                FactItem(id="id_1", text="1boy", primary_category=SemanticCategory.IDENTITY),
            ],
        }
    )

    filtered = mode.filter_facts(facts, TriggerConfig(word="DynamicPose"))
    filtered_texts = [f.text for f in filtered]

    assert "crossed arms" in filtered_texts
    assert "head tilt" in filtered_texts
    assert "cowboy shot" in filtered_texts
    assert "low angle" in filtered_texts
    assert "trench coat" not in filtered_texts
    assert "1boy" not in filtered_texts


def test_concept_mode_contracts_and_filtering():
    from hyper_captioner.caption_modes.concept import ConceptMode
    from hyper_captioner.caption_modes.registry import get_caption_mode

    mode = get_caption_mode("concept")
    assert mode.name == "concept"

    # Category contracts
    assert SemanticCategory.CONCEPT in mode.include_categories
    assert SemanticCategory.OBJECTS in mode.include_categories
    assert SemanticCategory.MATERIAL in mode.include_categories
    assert SemanticCategory.TEXTURE in mode.include_categories
    assert SemanticCategory.COLOR in mode.include_categories

    assert SemanticCategory.IDENTITY in mode.exclude_categories
    assert SemanticCategory.APPEARANCE in mode.exclude_categories
    assert SemanticCategory.UNCERTAINTY in mode.exclude_categories

    # Focal concept configuration
    custom_concept_mode = ConceptMode(focal_concept="levitation magic")
    assert custom_concept_mode.focal_concept == "levitation magic"
    instructions = custom_concept_mode.build_extraction_instructions()
    assert "levitation magic" in instructions

    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.CONCEPT: [
                FactItem(id="con_1", text="floating orbs", primary_category=SemanticCategory.CONCEPT),
            ],
            SemanticCategory.MATERIAL: [
                FactItem(id="mat_1", text="glowing crystal", primary_category=SemanticCategory.MATERIAL),
            ],
            SemanticCategory.IDENTITY: [
                FactItem(id="id_1", text="1girl", primary_category=SemanticCategory.IDENTITY),
            ],
            SemanticCategory.UNCERTAINTY: [
                FactItem(id="unc_1", text="possibly sparks", primary_category=SemanticCategory.UNCERTAINTY, is_uncertain=True),
            ],
        }
    )

    filtered = custom_concept_mode.filter_facts(facts, TriggerConfig(word="LevitationSpell"))
    filtered_texts = [f.text for f in filtered]

    assert "floating orbs" in filtered_texts
    assert "glowing crystal" in filtered_texts
    assert "1girl" not in filtered_texts
    assert "possibly sparks" not in filtered_texts


def test_base_caption_mode_category_helpers():
    from hyper_captioner.caption_modes.registry import get_caption_mode

    mode = get_caption_mode("style")
    assert mode.is_category_allowed(SemanticCategory.RENDERING) is True
    assert mode.is_category_allowed(SemanticCategory.CLOTHING) is False
    assert mode.is_category_allowed("rendering") is True
    assert mode.is_category_allowed("clothing") is False

    allowed_cats = mode.validate_categories([
        SemanticCategory.RENDERING,
        SemanticCategory.CLOTHING,
        SemanticCategory.TEXTURE,
    ])
    assert allowed_cats == [SemanticCategory.RENDERING, SemanticCategory.TEXTURE]


def test_format_tokens_across_modes():
    from hyper_captioner.caption_modes.registry import get_caption_mode

    mode = get_caption_mode("character")
    tokens = [
        CaptionToken(text="1girl", primary_category=SemanticCategory.IDENTITY),
        CaptionToken(text="solo", primary_category=SemanticCategory.IDENTITY),
        CaptionToken(text="smiling", primary_category=SemanticCategory.EXPRESSION),
    ]

    tag_str = mode.format_tokens(tokens, CaptionFormat.TAGS)
    assert tag_str == "1girl, solo, smiling"

    struct_str = mode.format_tokens(tokens, CaptionFormat.STRUCTURED)
    assert "1girl" in struct_str
    assert "smiling" in struct_str

    natural_str = mode.format_tokens(tokens, CaptionFormat.NATURAL)
    assert "1girl" in natural_str
    assert "smiling" in natural_str


def test_package_exports():
    import hyper_captioner.caption_modes as cm

    assert hasattr(cm, "BaseCaptionMode")
    assert hasattr(cm, "CharacterMode")
    assert hasattr(cm, "StyleMode")
    assert hasattr(cm, "OutfitMode")
    assert hasattr(cm, "PoseMode")
    assert hasattr(cm, "ConceptMode")
    assert hasattr(cm, "get_caption_mode")
    assert hasattr(cm, "list_caption_modes")
    assert hasattr(cm, "register_caption_mode")
