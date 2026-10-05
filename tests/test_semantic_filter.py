"""
Tests for Meaning-Based Content vs. Style Discriminator & Semantic Filter.
"""


from hyper_captioner.caption_modes.registry import get_caption_mode
from hyper_captioner.core.types import (
    FactItem,
    SemanticCategory,
    StructuredVisualFacts,
    TriggerConfig,
)
from hyper_captioner.pipeline.semantic_filter import (
    SemanticFilter,
    SemanticFilterResult,
)


def test_semantic_filter_style_mode_strips_content_preserves_rendering():
    """
    Style mode must strip content (character traits, hair/eye color, clothing)
    while preserving rendering techniques, textures, and material surface response.
    """
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.APPEARANCE: [
                FactItem(id="app_1", text="pink hair", primary_category=SemanticCategory.APPEARANCE),
                FactItem(id="app_2", text="blue eyes", primary_category=SemanticCategory.APPEARANCE),
            ],
            SemanticCategory.CLOTHING: [
                FactItem(id="clo_1", text="maid dress", primary_category=SemanticCategory.CLOTHING),
            ],
            SemanticCategory.RENDERING: [
                FactItem(id="ren_1", text="soft cel shading", primary_category=SemanticCategory.RENDERING),
                FactItem(id="ren_2", text="clean line art", primary_category=SemanticCategory.RENDERING),
            ],
            SemanticCategory.TEXTURE: [
                FactItem(id="tex_1", text="individually detailed hair strands", primary_category=SemanticCategory.TEXTURE),
            ],
            SemanticCategory.MATERIAL: [
                FactItem(
                    id="mat_1",
                    text="cracked matte leather surface with diffused specular highlights",
                    primary_category=SemanticCategory.MATERIAL,
                ),
                FactItem(
                    id="mat_2",
                    text="hard-surface metallic reflections",
                    primary_category=SemanticCategory.MATERIAL,
                ),
            ],
            SemanticCategory.UNCERTAINTY: [
                FactItem(id="unc_1", text="silk satin fabric", primary_category=SemanticCategory.UNCERTAINTY, is_uncertain=True),
            ],
        }
    )

    style_mode = get_caption_mode("style")
    filter_engine = SemanticFilter()
    result = filter_engine.filter_facts(style_mode, facts, TriggerConfig(word="MyStyle"))

    assert isinstance(result, SemanticFilterResult)
    accepted_texts = [f.text for f in result.accepted]

    # Rendering, texture, and surface response properties preserved
    assert "soft cel shading" in accepted_texts
    assert "clean line art" in accepted_texts
    assert "individually detailed hair strands" in accepted_texts
    assert "cracked matte leather surface with diffused specular highlights" in accepted_texts
    assert "hard-surface metallic reflections" in accepted_texts

    # Content and uncertain items rejected
    assert "pink hair" not in accepted_texts
    assert "blue eyes" not in accepted_texts
    assert "maid dress" not in accepted_texts
    assert "silk satin fabric" not in accepted_texts

    # Verify rejection reasons
    rejections = {fact.text: reason for fact, reason in result.rejected}
    assert rejections.get("pink hair") == "content_leak_in_style"
    assert rejections.get("blue eyes") == "content_leak_in_style"
    assert rejections.get("maid dress") == "content_leak_in_style"
    assert rejections.get("silk satin fabric") == "uncertainty"


def test_meaning_based_discrimination_overrides_naive_categories():
    """
    Categories serve as routing defaults, not absolute semantic truth:
    - Content facts miscategorized under TEXTURE/RENDERING are rejected in style mode.
    - Rendering/surface facts miscategorized under APPEARANCE/CLOTHING are accepted in style mode.
    """
    facts = StructuredVisualFacts(
        facts_by_category={
            # Miscategorized content under rendering/texture categories
            SemanticCategory.TEXTURE: [
                FactItem(id="tex_leak", text="leather jacket", primary_category=SemanticCategory.TEXTURE),
            ],
            SemanticCategory.RENDERING: [
                FactItem(id="ren_leak", text="pink hair", primary_category=SemanticCategory.RENDERING),
                FactItem(id="ren_prop", text="holding a sword", primary_category=SemanticCategory.RENDERING),
            ],
            # Rendering/surface facts miscategorized under content categories
            SemanticCategory.APPEARANCE: [
                FactItem(
                    id="app_strand",
                    text="individually resolved hair strands",
                    primary_category=SemanticCategory.APPEARANCE,
                ),
                FactItem(
                    id="app_skin",
                    text="soft subsurface-like skin rendering",
                    primary_category=SemanticCategory.APPEARANCE,
                ),
            ],
        }
    )

    style_mode = get_caption_mode("style")
    filter_engine = SemanticFilter()
    result = filter_engine.filter_facts(style_mode, facts, TriggerConfig())

    accepted_texts = [f.text for f in result.accepted]
    assert "individually resolved hair strands" in accepted_texts
    assert "soft subsurface-like skin rendering" in accepted_texts

    assert "leather jacket" not in accepted_texts
    assert "pink hair" not in accepted_texts
    assert "holding a sword" not in accepted_texts

    rejections = {fact.text: reason for fact, reason in result.rejected}
    assert rejections.get("leather jacket") == "content_leak_in_style"
    assert rejections.get("pink hair") == "content_leak_in_style"
    assert rejections.get("holding a sword") == "content_leak_in_style"


def test_character_mode_keeps_lighting_camera_pose_and_suppresses_jargon():
    """
    Character mode retains lighting, camera/framing, pose, environment context,
    and subject traits, while suppressing esoteric 3D/rendering jargon.
    """
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.IDENTITY: [
                FactItem(id="id_1", text="1girl", primary_category=SemanticCategory.IDENTITY),
            ],
            SemanticCategory.APPEARANCE: [
                FactItem(id="app_1", text="brown ponytail", primary_category=SemanticCategory.APPEARANCE),
            ],
            SemanticCategory.CLOTHING: [
                FactItem(id="clo_1", text="leather jacket", primary_category=SemanticCategory.CLOTHING),
            ],
            SemanticCategory.LIGHTING: [
                FactItem(id="lit_1", text="warm afternoon sunlight", primary_category=SemanticCategory.LIGHTING),
            ],
            SemanticCategory.CAMERA: [
                FactItem(id="cam_1", text="cowboy shot", primary_category=SemanticCategory.CAMERA),
            ],
            SemanticCategory.POSE: [
                FactItem(id="pos_1", text="sitting on a bench", primary_category=SemanticCategory.POSE),
            ],
            SemanticCategory.ENVIRONMENT: [
                FactItem(id="env_1", text="cherry blossom park", primary_category=SemanticCategory.ENVIRONMENT),
            ],
            # Esoteric rendering jargon to suppress
            SemanticCategory.RENDERING: [
                FactItem(id="ren_jargon1", text="octane render", primary_category=SemanticCategory.RENDERING),
                FactItem(id="ren_jargon2", text="subsurface scattering", primary_category=SemanticCategory.RENDERING),
                FactItem(id="ren_jargon3", text="unreal engine 5", primary_category=SemanticCategory.RENDERING),
            ],
        }
    )

    character_mode = get_caption_mode("character")
    filter_engine = SemanticFilter()
    result = filter_engine.filter_facts(character_mode, facts, TriggerConfig())

    accepted_texts = [f.text for f in result.accepted]
    assert "1girl" in accepted_texts
    assert "brown ponytail" in accepted_texts
    assert "leather jacket" in accepted_texts
    assert "warm afternoon sunlight" in accepted_texts
    assert "cowboy shot" in accepted_texts
    assert "sitting on a bench" in accepted_texts
    assert "cherry blossom park" in accepted_texts

    assert "octane render" not in accepted_texts
    assert "subsurface scattering" not in accepted_texts
    assert "unreal engine 5" not in accepted_texts

    rejections = {fact.text: reason for fact, reason in result.rejected}
    assert rejections.get("octane render") == "rendering_jargon"
    assert rejections.get("subsurface scattering") == "rendering_jargon"
    assert rejections.get("unreal engine 5") == "rendering_jargon"


def test_inviolable_user_lock_priority():
    """
    If fact.locked == True, it is ALWAYS accepted, overriding all category
    exclusions, content/style discrimination, hype pruning, and uncertainty.
    """
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.CLOTHING: [
                FactItem(
                    id="clo_locked",
                    text="red ribbon",
                    primary_category=SemanticCategory.CLOTHING,
                    locked=True,
                ),
            ],
            SemanticCategory.UNCERTAINTY: [
                FactItem(
                    id="unc_locked",
                    text="maybe silk kimono",
                    primary_category=SemanticCategory.UNCERTAINTY,
                    is_uncertain=True,
                    locked=True,
                ),
            ],
            SemanticCategory.QUALITY: [
                FactItem(
                    id="hype_locked",
                    text="breathtaking masterpiece",
                    primary_category=SemanticCategory.QUALITY,
                    locked=True,
                ),
            ],
        }
    )

    style_mode = get_caption_mode("style")
    filter_engine = SemanticFilter()
    result = filter_engine.filter_facts(style_mode, facts, TriggerConfig(word="MyStyle"))

    accepted_texts = [f.text for f in result.accepted]
    assert "red ribbon" in accepted_texts
    assert "maybe silk kimono" in accepted_texts
    assert "breathtaking masterpiece" in accepted_texts
    assert len(result.rejected) == 0


def test_uncertainty_rejection():
    """
    Uncertain facts (is_uncertain=True or SemanticCategory.UNCERTAINTY)
    must be strictly rejected with reason 'uncertainty'.
    """
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.UNCERTAINTY: [
                FactItem(id="u1", text="perhaps sunglasses", primary_category=SemanticCategory.UNCERTAINTY),
            ],
            SemanticCategory.CLOTHING: [
                FactItem(
                    id="u2",
                    text="possibly velvet cloak",
                    primary_category=SemanticCategory.CLOTHING,
                    is_uncertain=True,
                ),
            ],
        }
    )

    character_mode = get_caption_mode("character")
    filter_engine = SemanticFilter()
    result = filter_engine.filter_facts(character_mode, facts, TriggerConfig())

    assert len(result.accepted) == 0
    assert len(result.rejected) == 2
    for fact, reason in result.rejected:
        assert reason == "uncertainty"


def test_subjective_hype_rejection():
    """
    Subjective hype words (masterpiece, stunning, etc.) must be detected
    and rejected with reason 'subjective_hype'.
    """
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.QUALITY: [
                FactItem(id="q1", text="masterpiece", primary_category=SemanticCategory.QUALITY),
                FactItem(id="q2", text="best quality", primary_category=SemanticCategory.QUALITY),
            ],
            SemanticCategory.APPEARANCE: [
                FactItem(id="h1", text="stunning blue eyes", primary_category=SemanticCategory.APPEARANCE),
                FactItem(id="h2", text="breathtaking scenery", primary_category=SemanticCategory.ENVIRONMENT),
                FactItem(id="h3", text="gorgeous dress", primary_category=SemanticCategory.CLOTHING),
            ],
        }
    )

    character_mode = get_caption_mode("character")
    filter_engine = SemanticFilter()
    result = filter_engine.filter_facts(character_mode, facts, TriggerConfig())

    assert len(result.accepted) == 0
    for fact, reason in result.rejected:
        assert reason == "subjective_hype"


def test_trigger_trait_absorption():
    """
    When trigger_cfg.absorb_stable_traits is True:
    - Stable traits (is_stable=True) are rejected with 'redundant_trigger_trait'.
    - Non-stable traits (is_stable=False) are retained.
    - Exact trigger word duplicates are rejected with 'redundant_trigger_trait'.
    When absorb_stable_traits is False:
    - Stable traits are retained.
    """
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.IDENTITY: [
                FactItem(id="id_trig", text="Heroine", primary_category=SemanticCategory.IDENTITY),
                FactItem(id="id_solo", text="1girl", primary_category=SemanticCategory.IDENTITY, is_stable=False),
            ],
            SemanticCategory.APPEARANCE: [
                FactItem(id="app_stable", text="blue eyes", primary_category=SemanticCategory.APPEARANCE, is_stable=True),
            ],
            SemanticCategory.CLOTHING: [
                FactItem(id="clo_var", text="leather jacket", primary_category=SemanticCategory.CLOTHING, is_stable=False),
            ],
        }
    )

    character_mode = get_caption_mode("character")
    filter_engine = SemanticFilter()

    # Case 1: Trait absorption enabled
    trigger_absorb = TriggerConfig(word="Heroine", absorb_stable_traits=True)
    result_absorb = filter_engine.filter_facts(character_mode, facts, trigger_absorb)

    accepted_absorb = [f.text for f in result_absorb.accepted]
    assert "1girl" in accepted_absorb
    assert "leather jacket" in accepted_absorb
    assert "blue eyes" not in accepted_absorb
    assert "Heroine" not in accepted_absorb

    rejections = {fact.text: reason for fact, reason in result_absorb.rejected}
    assert rejections.get("blue eyes") == "redundant_trigger_trait"
    assert rejections.get("Heroine") == "redundant_trigger_trait"

    # Case 2: Trait absorption disabled
    trigger_no_absorb = TriggerConfig(word="Heroine", absorb_stable_traits=False)
    result_no_absorb = filter_engine.filter_facts(character_mode, facts, trigger_no_absorb)

    accepted_no_absorb = [f.text for f in result_no_absorb.accepted]
    assert "1girl" in accepted_no_absorb
    assert "leather jacket" in accepted_no_absorb
    assert "blue eyes" in accepted_no_absorb  # Retained when absorb is False
    assert "Heroine" not in accepted_no_absorb  # Redundant with trigger word itself


def test_outfit_mode_filtering():
    """
    Outfit mode retains garments, materials, and trims,
    while suppressing unrelated pose and environment clutter.
    """
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.CLOTHING: [
                FactItem(id="clo_1", text="silk blouse", primary_category=SemanticCategory.CLOTHING),
                FactItem(id="clo_2", text="pleated skirt", primary_category=SemanticCategory.CLOTHING),
            ],
            SemanticCategory.MATERIAL: [
                FactItem(id="mat_1", text="silk satin", primary_category=SemanticCategory.MATERIAL),
            ],
            SemanticCategory.POSE: [
                FactItem(id="pos_1", text="standing with arms crossed", primary_category=SemanticCategory.POSE),
            ],
            SemanticCategory.ENVIRONMENT: [
                FactItem(id="env_1", text="dense pine forest", primary_category=SemanticCategory.ENVIRONMENT),
            ],
            SemanticCategory.APPEARANCE: [
                FactItem(id="app_1", text="blue eyes", primary_category=SemanticCategory.APPEARANCE),
            ],
        }
    )

    outfit_mode = get_caption_mode("outfit")
    filter_engine = SemanticFilter()
    result = filter_engine.filter_facts(outfit_mode, facts, TriggerConfig())

    accepted_texts = [f.text for f in result.accepted]
    assert "silk blouse" in accepted_texts
    assert "pleated skirt" in accepted_texts
    assert "silk satin" in accepted_texts

    assert "standing with arms crossed" not in accepted_texts
    assert "dense pine forest" not in accepted_texts
    assert "blue eyes" not in accepted_texts

    rejections = {fact.text: reason for fact, reason in result.rejected}
    assert rejections.get("standing with arms crossed") == "excluded_category"
    assert rejections.get("dense pine forest") == "excluded_category"
    assert rejections.get("blue eyes") == "excluded_category"


def test_pose_mode_filtering():
    """
    Pose mode retains body orientation, limb positioning, stance, and camera angle,
    while suppressing clothing details and character identity traits.
    """
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.POSE: [
                FactItem(id="pos_1", text="standing with arms akimbo", primary_category=SemanticCategory.POSE),
                FactItem(id="pos_2", text="looking to the side", primary_category=SemanticCategory.POSE),
            ],
            SemanticCategory.CAMERA: [
                FactItem(id="cam_1", text="low angle cowboy shot", primary_category=SemanticCategory.CAMERA),
            ],
            SemanticCategory.CLOTHING: [
                FactItem(id="clo_1", text="maid dress", primary_category=SemanticCategory.CLOTHING),
            ],
            SemanticCategory.APPEARANCE: [
                FactItem(id="app_1", text="blonde hair", primary_category=SemanticCategory.APPEARANCE),
            ],
        }
    )

    pose_mode = get_caption_mode("pose")
    filter_engine = SemanticFilter()
    result = filter_engine.filter_facts(pose_mode, facts, TriggerConfig())

    accepted_texts = [f.text for f in result.accepted]
    assert "standing with arms akimbo" in accepted_texts
    assert "looking to the side" in accepted_texts
    assert "low angle cowboy shot" in accepted_texts

    assert "maid dress" not in accepted_texts
    assert "blonde hair" not in accepted_texts

    rejections = {fact.text: reason for fact, reason in result.rejected}
    assert rejections.get("maid dress") == "excluded_category"
    assert rejections.get("blonde hair") == "excluded_category"


def test_concept_mode_filtering():
    """
    Concept mode retains concept, objects, materials, and textures,
    while suppressing identity, appearance, and uncertainty.
    """
    facts = StructuredVisualFacts(
        facts_by_category={
            SemanticCategory.CONCEPT: [
                FactItem(id="con_1", text="cyberpunk neon aesthetic", primary_category=SemanticCategory.CONCEPT),
            ],
            SemanticCategory.OBJECTS: [
                FactItem(id="obj_1", text="holographic terminal", primary_category=SemanticCategory.OBJECTS),
            ],
            SemanticCategory.IDENTITY: [
                FactItem(id="id_1", text="1girl", primary_category=SemanticCategory.IDENTITY),
            ],
        }
    )

    concept_mode = get_caption_mode("concept")
    filter_engine = SemanticFilter()
    result = filter_engine.filter_facts(concept_mode, facts, TriggerConfig())

    accepted_texts = [f.text for f in result.accepted]
    assert "cyberpunk neon aesthetic" in accepted_texts
    assert "holographic terminal" in accepted_texts
    assert "1girl" not in accepted_texts
