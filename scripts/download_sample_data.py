import os
import sys
import csv
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger

logger = get_logger("download_sample_data")

CREDIBLE_TEMPLATES = [
    "Peer-reviewed study in {journal} confirms {topic} findings.",
    "Scientists at {institution} publish new research on {topic}.",
    "Official {agency} data shows {stat} in annual report.",
    "Clinical trial demonstrates {effect} in controlled study.",
    "Government report documents {topic} statistics for {year}.",
    "University research team validates {topic} hypothesis.",
    "Independent audit confirms {topic} compliance results.",
    "New {journal} meta-analysis reviews evidence on {topic}.",
]
MISINFO_TEMPLATES = [
    "SHOCKING: {topic} cover-up exposed by insiders!",
    "They do not want you to know the truth about {topic}.",
    "Scientists BAFFLED by miracle {topic} discovery!",
    "{topic} causes {disease} - mainstream media hiding it.",
    "LEAKED documents prove {agency} lied about {topic}.",
    "One weird trick {topic} doctors refuse to discuss!",
    "Illuminati connection to {topic} finally exposed!",
    "{topic} contains DANGEROUS chemicals they hide from you!",
]
JOURNALS = ["Nature", "Science", "NEJM", "Lancet", "JAMA"]
INSTITUTIONS = ["Harvard", "MIT", "Stanford", "Oxford", "WHO"]
AGENCIES = ["CDC", "FDA", "EPA", "USDA", "NHS"]
TOPICS = [
    "vaccine safety",
    "climate change",
    "food safety",
    "drug efficacy",
    "economic data",
    "cancer research",
    "water quality",
    "air pollution",
]
DISEASES = ["cancer", "autism", "diabetes", "heart disease"]
YEARS = ["2023", "2024", "2025"]
STATS = ["3.2 percent growth", "15 percent reduction", "stable trends"]
EFFECTS = ["significant reduction", "modest improvement", "no adverse effects"]


def generate_row(label, seed_offset=0):
    """
    Generate one synthetic row.

    Args:
        label (int): 0 for credible, 1 for misinformation
        seed_offset (int): offset for deterministic generation
    Returns:
        dict: row with text, label, category, source
    """
    rng = random.Random(seed_offset)
    template = rng.choice(CREDIBLE_TEMPLATES if label == 0 else MISINFO_TEMPLATES)
    text = template.format(
        journal=rng.choice(JOURNALS),
        institution=rng.choice(INSTITUTIONS),
        agency=rng.choice(AGENCIES),
        topic=rng.choice(TOPICS),
        disease=rng.choice(DISEASES),
        year=rng.choice(YEARS),
        stat=rng.choice(STATS),
        effect=rng.choice(EFFECTS),
    )
    return {
        "text": text,
        "label": label,
        "category": "credible" if label == 0 else "misinformation",
        "source": "synthetic",
    }


def write_csv(path, rows):
    """Write rows to CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "category", "source"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), path)


def main():
    """Generate train/val/test CSV files in data/ directory."""
    random.seed(42)
    os.makedirs("data", exist_ok=True)

    for split, n in [("train", 800), ("val", 100), ("test", 100)]:
        rows = []
        for i in range(n // 2):
            rows.append(generate_row(0, seed_offset=i))
        for i in range(n // 2):
            rows.append(generate_row(1, seed_offset=i + 10000))
        random.shuffle(rows)
        write_csv("data/%s.csv" % split, rows)

    print("Sample data written to data/train.csv, data/val.csv, data/test.csv")
    print("Total: 800 train, 100 val, 100 test samples")


if __name__ == "__main__":
    main()
