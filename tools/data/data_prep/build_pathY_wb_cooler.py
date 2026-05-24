"""Build pathY_wb_cooler.json: same 300 FiveK source images as pathY_wb.json
but with 'Apply cooler white balance...' caption instead.

This complements the warmer-wb run to test direction-diversity hypothesis (H2)
in addition to sample-count hypothesis (H1).
"""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
SRC = PROJECT / "data" / "pathY_captions" / "pathY_wb.json"
DST = PROJECT / "data" / "pathY_captions" / "pathY_wb_cooler.json"

COOLER_CAPTION = (
    "Apply cooler white balance. Keep the original composition "
    "and subject unchanged."
)


def main():
    with open(SRC, encoding="utf-8") as f:
        cfg = json.load(f)

    # Same samples, same idx, new caption
    new_samples = []
    for s in cfg["samples"]:
        new_samples.append({
            **s,
            "new_caption": COOLER_CAPTION,
            "tone_target": "wb_cooler",  # different action label
        })

    new_cfg = {
        "metadata": {
            **cfg.get("metadata", {}),
            "purpose": "Path Y: +300 FireRed cooler-WB pseudo-labels "
                      "(direction diversity supplement to pathY_wb)",
            "action": "wb_cooler",
            "caption": COOLER_CAPTION,
            "n_samples": len(new_samples),
            "source_of_images": str(SRC),
        },
        "samples": new_samples,
    }

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {len(new_samples)} samples -> {DST}")
    print(f"     caption: '{COOLER_CAPTION}'")


if __name__ == "__main__":
    main()
