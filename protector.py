from pathlib import Path
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def analyze_dataset(dataset_path):
    dataset_path = Path(dataset_path)

    corrupted_images = []
    empty_captions = []
    unreadable_captions = []

    for file in dataset_path.rglob("*"):

        if file.suffix.lower() in SUPPORTED_EXTENSIONS:

            try:
                with Image.open(file) as img:
                    img.verify()

            except Exception:
                corrupted_images.append(file)

        elif file.suffix.lower() == ".txt":

            try:
                content = file.read_text(
                    encoding="utf-8"
                ).strip()

                if len(content) == 0:
                    empty_captions.append(file)

            except Exception:
                unreadable_captions.append(file)

    return {
        "corrupted_images": corrupted_images,
        "empty_captions": empty_captions,
        "unreadable_captions": unreadable_captions
    }