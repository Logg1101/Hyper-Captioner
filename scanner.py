from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def scan_dataset(dataset_path):
    """
    Scan a dataset folder and return:
    - images
    - images with captions
    - images missing captions
    - orphan captions
    """

    dataset_path = Path(dataset_path)

    images = []
    with_captions = []
    missing_captions = []
    orphan_captions = []

    image_stems = set()

    for file in dataset_path.rglob("*"):
        if file.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(file)
            image_stems.add(str(file.with_suffix("")))

    for image in images:
        txt_file = image.with_suffix(".txt")

        if txt_file.exists():
            with_captions.append(image)
        else:
            missing_captions.append(image)

    for txt_file in dataset_path.rglob("*.txt"):
        matching_image_found = False

        for ext in SUPPORTED_EXTENSIONS:
            if txt_file.with_suffix(ext).exists():
                matching_image_found = True
                break

        if not matching_image_found:
            orphan_captions.append(txt_file)

    return {
        "images": images,
        "with_captions": with_captions,
        "missing_captions": missing_captions,
        "orphan_captions": orphan_captions
    }