from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VocabularyItem:
    sign_id: int
    meaning: str
    tts_text: str
    image_key: str


def load_vocabulary(path: Path) -> dict[int, VocabularyItem]:
    items: dict[int, VocabularyItem] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sign_id = int(row["sign_id"])
            items[sign_id] = VocabularyItem(
                sign_id=sign_id,
                meaning=row["meaning"],
                tts_text=row["tts_text"],
                image_key=row["image_key"],
            )
    return items
