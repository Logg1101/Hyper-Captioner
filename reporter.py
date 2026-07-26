from pathlib import Path
from datetime import datetime


def generate_report(
    dataset_path,
    scan_results,
    protection_report
):
    dataset_path = Path(dataset_path)

    report_file = dataset_path / "scan_report.txt"

    lines = []

    lines.append("=" * 60)
    lines.append("IMAGE FLUX CAPTIONER TM REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    lines.append(
        f"Dataset: {dataset_path}"
    )

    lines.append("")

    lines.append(
        f"Total Images: {len(scan_results['images'])}"
    )

    lines.append(
        f"Existing Captions: {len(scan_results['with_captions'])}"
    )

    lines.append(
        f"Missing Captions: {len(scan_results['missing_captions'])}"
    )

    lines.append(
        f"Orphan Captions: {len(scan_results['orphan_captions'])}"
    )

    lines.append("")

    lines.append(
        f"Corrupted Images: {len(protection_report['corrupted_images'])}"
    )

    lines.append(
        f"Empty Captions: {len(protection_report['empty_captions'])}"
    )

    lines.append(
        f"Unreadable TXT Files: {len(protection_report['unreadable_captions'])}"
    )

    lines.append("")

    if scan_results["missing_captions"]:
        lines.append("MISSING CAPTIONS")
        lines.append("-" * 60)

        for file in scan_results["missing_captions"]:
            lines.append(str(file))

        lines.append("")

    if scan_results["orphan_captions"]:
        lines.append("ORPHAN CAPTIONS")
        lines.append("-" * 60)

        for file in scan_results["orphan_captions"]:
            lines.append(str(file))

        lines.append("")

    if protection_report["corrupted_images"]:
        lines.append("CORRUPTED IMAGES")
        lines.append("-" * 60)

        for file in protection_report["corrupted_images"]:
            lines.append(str(file))

        lines.append("")

    if protection_report["empty_captions"]:
        lines.append("EMPTY CAPTIONS")
        lines.append("-" * 60)

        for file in protection_report["empty_captions"]:
            lines.append(str(file))

        lines.append("")

    if protection_report["unreadable_captions"]:
        lines.append("UNREADABLE TXT FILES")
        lines.append("-" * 60)

        for file in protection_report["unreadable_captions"]:
            lines.append(str(file))

        lines.append("")

    report_file.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    return report_file