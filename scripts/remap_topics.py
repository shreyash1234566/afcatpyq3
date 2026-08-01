"""
scripts/remap_topics.py

Normalises all topic labels in data/processed/Q.json using the canonical
mapping in data/topic_map.json, then writes the cleaned file to
data/processed/Q_clean.json.

Run:
    python scripts/remap_topics.py
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
Q_PATH = ROOT / "data" / "processed" / "Q.json"
MAP_PATH = ROOT / "data" / "topic_map.json"
OUT_PATH = ROOT / "data" / "processed" / "Q_clean.json"


def build_lookup(topic_map: dict) -> dict:
    """
    Build a flat alias->canonical dict.
    {alias_lower: (section, canonical_topic)}
    """
    lookup = {}
    for section, topics in topic_map.items():
        for canonical, aliases in topics.items():
            for alias in aliases:
                lookup[alias.strip().lower()] = (section, canonical)
            # also map the canonical itself
            lookup[canonical.strip().lower()] = (section, canonical)
    return lookup


def remap(data: list, lookup: dict) -> list:
    remapped = []
    unmapped_raw = defaultdict(int)
    mapped_count = 0
    skipped_count = 0

    for q in data:
        raw_topic = q.get("topic", "").strip()
        key = raw_topic.lower()
        if key in lookup:
            sec, canonical = lookup[key]
            q = dict(q)
            q["topic"] = canonical
            q["section"] = sec
            mapped_count += 1
        else:
            unmapped_raw[raw_topic] += 1
            skipped_count += 1
        remapped.append(q)

    print(f"  Mapped:   {mapped_count}")
    print(f"  Unmapped: {skipped_count}")
    if unmapped_raw:
        print("\n  Unmapped topic labels (fix topic_map.json if important):")
        for t, c in sorted(unmapped_raw.items(), key=lambda x: -x[1]):
            print(f"    [{c:3d}] {t!r}")
    return remapped


def main():
    print(f"Loading {Q_PATH} ...")
    data = json.loads(Q_PATH.read_text(encoding="utf-8"))
    print(f"  {len(data)} total questions")

    topic_map = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    lookup = build_lookup(topic_map)
    print(f"  {len(lookup)} alias entries in topic_map")

    print("\nRemapping topics ...")
    cleaned = remap(data, lookup)

    # Validate counts
    from collections import Counter
    topics_after = Counter(q["topic"] for q in cleaned)
    print(f"\nTop 20 topics after remapping:")
    for t, c in topics_after.most_common(20):
        print(f"  {c:4d}  {t}")

    OUT_PATH.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote cleaned data to {OUT_PATH}")


if __name__ == "__main__":
    main()
