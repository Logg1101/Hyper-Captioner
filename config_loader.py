import json
from pathlib import Path


def load_config():

    config_file = Path("config.json")

    if not config_file.exists():
        raise FileNotFoundError(
            "config.json was not found."
        )

    with open(
        config_file,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    return config