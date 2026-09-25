"""Step 1: generate ~120 root causes (issue + canonical fix) for the CloudDesk taxonomy."""
import json

import yaml
from pydantic import BaseModel

from data.generator.llm import ROOT, generate_json, usage_totals

OUT = ROOT / "data" / "raw" / "root_causes.json"
PER_CATEGORY = 15


class RootCause(BaseModel):
    title: str            # short internal name, e.g. "403 after SSO token format change in v4.2"
    product: str          # one of the products
    symptoms: str         # what customers observe / write about (1-2 sentences)
    error_code: str       # e.g. "403", "E-SYNC-102", or "" if none
    canonical_fix: str    # what support tells the customer (concrete steps, 2-4 sentences)
    typical_priority: str # LOW|MEDIUM|HIGH|URGENT
    is_release_related: bool


def main():
    tax = yaml.safe_load((ROOT / "data/generator/taxonomy.yaml").read_text(encoding="utf-8"))
    all_rc, n = [], 0
    for cat, meta in tax["categories"].items():
        prompt = f"""You are designing a realistic support knowledge base for a fictional product.
{tax['product_description']}

Category: {cat} ({meta['desc']}).
Produce {PER_CATEGORY} DISTINCT root causes / recurring issues that customers report in this category.
Each must be specific and concrete (real-sounding feature names, settings, error codes, version numbers),
clearly different from the others, and have a canonical fix an agent would send (concrete steps, no fluff).
Products must be one of: {tax['products']}. typical_priority is one of {tax['priorities']}.
Include 2 issues tied to the v4.2 release if plausible for this category (is_release_related=true).
For FEATURE_REQUEST the canonical_fix is the standard answer (status, workaround, or how to vote)."""
        items = generate_json(prompt, schema=list[RootCause], temperature=0.8, tag=f"rc-{cat}")
        for it in items:
            n += 1
            it["id"] = f"RC-{n:03d}"
            it["category"] = cat
            all_rc.append(it)
        print(f"{cat}: {len(items)}")
    OUT.write_text(json.dumps(all_rc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"total {len(all_rc)} -> {OUT}", usage_totals())


if __name__ == "__main__":
    main()
