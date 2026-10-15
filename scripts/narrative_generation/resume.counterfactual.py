import sys, json
from pathlib import Path
from generate_narratives_counterfactual import (
    COUNTERFACTUAL_JOBS, PROVIDERS_TO_RUN, generate_counterfactual_narrative, save_result)
import generate_narratives as gn

OUT = "results/narratives"
datasets = set(sys.argv[1:]) or {"credit", "law", "saudi", "student"}

def done(path):
    try:
        d = json.load(open(path))
        return d.get("status") == "success" and len((d.get("narrative") or "").strip()) >= 200
    except Exception:
        return False

# Counterfactual gaps
for job in [j for j in COUNTERFACTUAL_JOBS if j["dataset"] in datasets]:
    for provider, model in PROVIDERS_TO_RUN:
        d = Path(OUT) / job["dataset"] / "counterfactual_pa" / job["label"] / provider / model
        todo = [i for i in job["instances"] if not done(d / f"instance_{i}.json")]
        if not todo: continue
        print(f"{job['dataset']} {job['label']} / {provider}: {len(todo)} to do")
        for i in todo:
            r = generate_counterfactual_narrative(job["dataset"], i, job["attr_column"],
                    job["override_value"], job["label"], provider=provider, model=model)
            save_result(r, OUT)
            print(f"  {i}: {r['status']}, {len(r['narrative'] or '')} chars")

# include_pa / exclude_pa gaps
for ds in datasets:
    for cond in ["include_pa", "exclude_pa"]:
        for provider, model in PROVIDERS_TO_RUN:
            for f in (Path(OUT) / ds / cond / provider / model).glob("instance_*.json"):
                if done(f): continue
                i = int(f.stem.split("_")[1])
                print(f"{ds} {cond} / {provider}: instance {i}")
                r = gn.generate_narrative(ds, i, provider=provider, model=model,
                        exclude_protected_attributes=(cond == "exclude_pa"))
                gn.save_result(r, OUT)
                print(f"  {r['status']}, {len(r['narrative'] or '')} chars")