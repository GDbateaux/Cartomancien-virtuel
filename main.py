from card_extractor import CardExtractor

from pathlib import Path


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    image_paths = sorted(list(data_dir.glob("*.jpg")))

    for img_path in image_paths:
        cards = CardExtractor(str(img_path), True).get_cards()
        print(f"Extracted {len(cards)} cards from {img_path.name}")
