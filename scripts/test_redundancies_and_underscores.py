"""
Test script validating:
1. Underscore preservation (keep_underscores=True / False)
2. Subsumption pruning (dress vs lace dress, jewelry vs earrings)
3. Contradiction resolution on single subjects (medium breasts vs large breasts, short hair vs medium hair)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hyper_captioner.core.types import CaptionMode, CharacterConfig, LoRAStrategy, TagCategory, TagItem
from hyper_captioner.pipeline.builder import CaptionBuilder
from hyper_captioner.vocabulary.normalizer import VocabularyNormalizer


def test_redundancies_and_underscores():
    print("=" * 70)
    print("TESTING SUBSUMPTION PRUNING & UNDERSCORE PRESERVATION")
    print("=" * 70)

    normalizer = VocabularyNormalizer()
    builder = CaptionBuilder()

    # Reproduce exact tags from User Screenshot #1:
    # "isolde, 1girl, solo, breasts, blue eyes, medium hair, short hair, bangs, blue hair,
    #  hair between eyes, large breasts, open mouth, smile, blush, jewelry, lace dress,
    #  earrings, dress, black dress, long sleeves, necklace, looking at viewer, upper body,
    #  night sky, night, sky, indoors, window, cleavage, blue flower, blue rose, clothing cutout,
    #  hand up, lace trim, rose (flower), flower, parted lips, original, lace, teeth, rose,
    #  candle, red flower, head rest, hand on own face, red rose, blurry foreground, depth of field,
    #  hand on own cheek, rain, gem, blue gemstone, cleavage cutout"
    raw_tag_names = [
        ("isolde", 1.0, TagCategory.CHARACTER),
        ("1girl", 0.99, TagCategory.CHARACTER),
        ("solo", 0.98, TagCategory.CHARACTER),
        ("breasts", 0.95, TagCategory.APPEARANCE),
        ("large breasts", 0.92, TagCategory.APPEARANCE),
        ("blue eyes", 0.96, TagCategory.APPEARANCE),
        ("medium hair", 0.88, TagCategory.APPEARANCE),
        ("short hair", 0.70, TagCategory.APPEARANCE),
        ("bangs", 0.85, TagCategory.APPEARANCE),
        ("blue hair", 0.94, TagCategory.APPEARANCE),
        ("hair between eyes", 0.82, TagCategory.APPEARANCE),
        ("open mouth", 0.75, TagCategory.EXPRESSION),
        ("parted lips", 0.86, TagCategory.EXPRESSION),
        ("teeth", 0.65, TagCategory.EXPRESSION),
        ("smile", 0.80, TagCategory.EXPRESSION),
        ("blush", 0.83, TagCategory.EXPRESSION),
        ("jewelry", 0.92, TagCategory.CLOTHING),
        ("lace dress", 0.90, TagCategory.CLOTHING),
        ("black dress", 0.89, TagCategory.CLOTHING),
        ("dress", 0.95, TagCategory.CLOTHING),  # Generic!
        ("earrings", 0.88, TagCategory.CLOTHING),
        ("necklace", 0.87, TagCategory.CLOTHING),
        ("long sleeves", 0.84, TagCategory.CLOTHING),
        ("looking at viewer", 0.95, TagCategory.ACTION),
        ("upper body", 0.93, TagCategory.FRAMING),
        ("night sky", 0.89, TagCategory.ENVIRONMENT),
        ("night", 0.85, TagCategory.ENVIRONMENT),  # Generic!
        ("sky", 0.80, TagCategory.ENVIRONMENT),    # Generic!
        ("indoors", 0.86, TagCategory.ENVIRONMENT),
        ("window", 0.82, TagCategory.ENVIRONMENT),
        ("cleavage", 0.80, TagCategory.APPEARANCE),
        ("blue flower", 0.79, TagCategory.OBJECTS),
        ("blue rose", 0.85, TagCategory.OBJECTS),
        ("red rose", 0.81, TagCategory.OBJECTS),
        ("flower", 0.90, TagCategory.OBJECTS),     # Generic!
        ("rose (flower)", 0.88, TagCategory.OBJECTS), # Generic!
        ("rose", 0.87, TagCategory.OBJECTS),       # Generic!
        ("lace trim", 0.82, TagCategory.CLOTHING),
        ("lace", 0.89, TagCategory.CLOTHING),      # Generic!
    ]

    tag_items = [
        TagItem(text=name, source="wd14", confidence=score, category=cat)
        for name, score, cat in raw_tag_names
    ]

    print(f"\nInitial raw tags count: {len(tag_items)}")
    
    # Prune redundancies
    pruned = normalizer.prune_redundancies(tag_items, keep_underscores=True)
    pruned_names = [t.text.lower().replace("_", " ") for t in pruned]

    print(f"Pruned tags count: {len(pruned)}")

    # 1. Assert Generic "dress" is pruned because "lace dress" & "black dress" exist
    assert "dress" not in pruned_names, "Generic 'dress' was NOT pruned!"
    assert "lace dress" in pruned_names, "'lace dress' should be kept!"
    assert "black dress" in pruned_names, "'black dress' should be kept!"
    print("[PASS] Subsumption: 'dress' removed, specific 'lace dress' and 'black dress' preserved.")

    # 2. Assert Generic "jewelry" is pruned because "earrings" & "necklace" exist
    assert "jewelry" not in pruned_names, "Generic 'jewelry' was NOT pruned!"
    assert "earrings" in pruned_names
    assert "necklace" in pruned_names
    print("[PASS] Subsumption: 'jewelry' removed, specific 'earrings' and 'necklace' preserved.")

    # 3. Assert Generic "breasts" is pruned because "large breasts" exists
    assert "breasts" not in pruned_names, "Generic 'breasts' was NOT pruned!"
    assert "large breasts" in pruned_names
    print("[PASS] Subsumption: 'breasts' removed, specific 'large breasts' preserved.")

    # 4. Assert Contradictory hair length is resolved
    assert "short hair" not in pruned_names, "'short hair' should be pruned in favor of higher confidence 'medium hair'!"
    assert "medium hair" in pruned_names
    print("[PASS] Mutually Exclusive: 'short hair' pruned, higher-confidence 'medium hair' preserved.")

    # 5. Assert Generic flowers pruned
    assert "flower" not in pruned_names
    assert "rose (flower)" not in pruned_names
    assert "rose" not in pruned_names
    assert "blue rose" in pruned_names
    assert "red rose" in pruned_names
    print("[PASS] Subsumption: Generic 'flower' and 'rose' removed, specific 'blue rose' and 'red rose' preserved.")

    # 6. Assert Generic night/sky pruned
    assert "night" not in pruned_names
    assert "sky" not in pruned_names
    assert "night sky" in pruned_names
    print("[PASS] Subsumption: Generic 'night' and 'sky' removed, specific 'night sky' preserved.")

    # 7. Test CaptionBuilder with keep_underscores=True
    char_cfg = CharacterConfig(name="Isolde", trigger_word="isolde")
    res_under = builder.build(
        fused_tags=pruned,
        caption_mode=CaptionMode.HYBRID,
        lora_strategy=LoRAStrategy.CHARACTER,
        character_config=char_cfg,
        keep_underscores=True,
    )
    print("\nGenerated Caption (keep_underscores=True):")
    print(f"  \"{res_under.caption}\"")

    # Assert underscores are actually present!
    assert "blue_eyes" in res_under.caption, "Underscores missing in 'blue_eyes'!"
    assert "lace_dress" in res_under.caption, "Underscores missing in 'lace_dress'!"
    assert "black_dress" in res_under.caption, "Underscores missing in 'black_dress'!"
    assert "looking_at_viewer" in res_under.caption, "Underscores missing in 'looking_at_viewer'!"
    assert "large_breasts" in res_under.caption, "Underscores missing in 'large_breasts'!"
    assert "night_sky" in res_under.caption, "Underscores missing in 'night_sky'!"
    tags_in_cap = [t.strip() for t in res_under.caption.split(",")]
    assert "dress" not in tags_in_cap, "Generic 'dress' found in caption!"
    assert "open_mouth" not in tags_in_cap, "Generic 'open_mouth' should be pruned when parted_lips is present!"
    print("[PASS] Underscores strictly preserved in final output string.")

    # 8. Test CaptionBuilder with keep_underscores=False
    res_spaces = builder.build(
        fused_tags=pruned,
        caption_mode=CaptionMode.HYBRID,
        lora_strategy=LoRAStrategy.CHARACTER,
        character_config=char_cfg,
        keep_underscores=False,
    )
    print("\nGenerated Caption (keep_underscores=False):")
    print(f"  \"{res_spaces.caption}\"")
    assert "blue eyes" in res_spaces.caption
    assert "lace dress" in res_spaces.caption
    print("[PASS] Spaces strictly preserved when keep_underscores=False.")

    print("\n" + "=" * 70)
    print("ALL REDUNDANCY & UNDERSCORE TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_redundancies_and_underscores()
