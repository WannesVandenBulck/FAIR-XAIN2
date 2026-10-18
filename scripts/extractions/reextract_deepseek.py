#!/usr/bin/env python
"""
Re-extract the regenerated DeepSeek include_pa / exclude_pa narratives
(DeepSeek-V4.1-Flash, Oct 2026) with the Grok extractor, overwriting the
old extractions that belonged to the August narratives.

Uses extract_single() from extraction.py, so the extractor, prompt and
output paths are identical to all other extractions.

Progress is logged in DONE_LOG, so the script is safe to rerun: it only
retries narratives that have not been extracted successfully yet.

Run from the repo root:
    python scripts/extractions/reextract_deepseek_factual.py
"""

from pathlib import Path

from extraction import extract_single

DATASETS = ["credit", "saudi", "student", "law"]
CONDITIONS = ["include_pa", "exclude_pa"]
NARR = Path("results/narratives")
DONE_LOG = Path("results/extractions/deepseek_v41_reextraction_done.txt")
PASSES = 3  # retry failed extractions up to 3 times per run


def main():
    done = set(DONE_LOG.read_text(encoding="utf-8").split()) if DONE_LOG.exists() else set()

    all_jobs = []
    for ds in DATASETS:
        for cond in CONDITIONS:
            for f in sorted(NARR.glob(f"{ds}/{cond}/deepseek/*/instance_*.json")):
                all_jobs.append((ds, cond, int(f.stem.split("_")[1])))
    print(f"{len(all_jobs)} DeepSeek factual narratives found, {len(done)} already re-extracted")

    for p in range(1, PASSES + 1):
        todo = [j for j in all_jobs if f"{j[0]}/{j[1]}/{j[2]}" not in done]
        print(f"\n=== Pass {p}: {len(todo)} to extract ===")
        if not todo:
            break
        for n, (ds, cond, i) in enumerate(todo, 1):
            try:
                ok, _, err = extract_single(ds, i, "deepseek", "grok", condition=cond)
            except Exception as e:
                ok, err = False, str(e)
            if ok:
                key = f"{ds}/{cond}/{i}"
                done.add(key)
                with open(DONE_LOG, "a", encoding="utf-8") as fh:
                    fh.write(key + "\n")
            print(f"[{n}/{len(todo)}] {ds} {cond} #{i}: {'OK' if ok else 'FAIL ' + str(err)[:100]}")

    remaining = len(all_jobs) - len([j for j in all_jobs if f"{j[0]}/{j[1]}/{j[2]}" in done])
    print(f"\nDone. {remaining} still to extract" + (" - rerun the script." if remaining else "."))


if __name__ == "__main__":
    main()