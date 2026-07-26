from pathlib import Path

from scanner import scan_dataset
from backup import create_backup
from protector import analyze_dataset
from reporter import generate_report
from captioner import caption_dataset


def print_report(results):

    print("\n" + "=" * 60)
    print("IMAGE FLUX CAPTIONER TM")
    print("=" * 60)

    print(
        f"\nTotal Images        : "
        f"{len(results['images'])}"
    )

    print(
        f"Existing Captions   : "
        f"{len(results['with_captions'])}"
    )

    print(
        f"Missing Captions    : "
        f"{len(results['missing_captions'])}"
    )

    print(
        f"Orphan Captions     : "
        f"{len(results['orphan_captions'])}"
    )

    print("\nScan Complete.")
    print("=" * 60)


def print_protection_report(report):

    print("\nDATASET HEALTH CHECK")
    print("-" * 60)

    print(
        f"Corrupted Images : "
        f"{len(report['corrupted_images'])}"
    )

    print(
        f"Empty Captions   : "
        f"{len(report['empty_captions'])}"
    )

    print(
        f"Unreadable TXT   : "
        f"{len(report['unreadable_captions'])}"
    )

    print("-" * 60)


def main():

    print("Image Flux Captioner TM")
    print()

    dataset_path = input(
        "Enter dataset folder path: "
    ).strip()

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():

        print(
            "\nERROR: Folder does not exist."
        )

        return

    results = scan_dataset(
        dataset_path
    )

    protection_report = analyze_dataset(
        dataset_path
    )

    print_report(results)

    print_protection_report(
        protection_report
    )

    report_file = generate_report(
        dataset_path,
        results,
        protection_report
    )

    print(
        f"\nReport saved to:\n"
        f"{report_file}"
    )

    if len(results["missing_captions"]) == 0:

        print(
            "\nNothing to caption."
        )

        return

    answer = input(
        "\nCreate backup before captioning? (y/n): "
    ).strip().lower()

    if answer != "y":

        print(
            "\nOperation cancelled."
        )

        return

    create_backup(
        dataset_path
    )

    print("\nCaption Settings")
    print("-" * 60)

    character_name = input(
        "Character Name (optional): "
    ).strip()

    base_caption = input(
        "Base Caption (optional): "
    ).strip()

    print("\nStarting JoyCaption...")
    print(
        f"Images to caption: "
        f"{len(results['missing_captions'])}"
    )

    caption_result = caption_dataset(
        results["missing_captions"],
        character_name,
        base_caption
    )

    print("\nCAPTION RESULTS")
    print("-" * 60)

    print(
        f"Created Captions : "
        f"{caption_result['created']}"
    )

    print(
        f"Skipped Captions : "
        f"{caption_result['skipped']}"
    )

    print(
        f"Failed Captions  : "
        f"{caption_result['failed']}"
    )

    print("-" * 60)

    print("\nFinished.")


if __name__ == "__main__":
    main()