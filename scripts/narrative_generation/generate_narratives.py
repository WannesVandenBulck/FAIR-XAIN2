#!/usr/bin/env python
"""
Code to generate narratives. 
"""

import subprocess
import sys
from datetime import datetime

# All adversely predicted instances in credit dataset (0-33)
ALL_INSTANCES = 17, 18
PROVIDERS = ["mistral"]
DATASET = "credit"
PROMPT_TYPE = "shap"

def run_command(cmd):
    """Run command and return exit code."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

def main():
    start_time = datetime.now()
    total_narratives = len(ALL_INSTANCES) * len(PROVIDERS)
    
    print("=" * 80)
    print(f"GENERATING SHAP NARRATIVES FOR CREDIT DATASET")
    print(f"Total: {total_narratives} narratives ({len(PROVIDERS)} providers × {len(ALL_INSTANCES)} instances)")
    print("=" * 80)
    print()
    
    completed = 0
    failed = 0
    
    for i, provider in enumerate(PROVIDERS, 1):
        print(f"[{i}/{len(PROVIDERS)}] Provider: {provider.upper()}")
        print(f"    Instances: 0-{len(ALL_INSTANCES)-1} ({len(ALL_INSTANCES)} total)")
        print(f"    Time: {datetime.now().strftime('%HH:%MM:%SS')}")
        
        # Build command with all instances
        cmd = [
            sys.executable,
            "scripts/make_narratives.py",
            "--dataset", DATASET,
            "--prompt-type", PROMPT_TYPE,
            "--provider", provider,
            "--instances"
        ] + [str(i) for i in ALL_INSTANCES]
        
        exit_code = run_command(cmd)
        
        if exit_code == 0:
            completed += 1
            print(f"    ✓ Success\n")
        else:
            failed += 1
            print(f"    ✗ Failed (exit code: {exit_code})\n")
    
    # Summary
    print("=" * 80)
    print("GENERATION COMPLETE")
    print(f"Completed: {completed}/{len(PROVIDERS)} providers")
    print(f"Failed: {failed}/{len(PROVIDERS)} providers")
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Total time: {int(duration.total_seconds() / 60)} minutes {int(duration.total_seconds() % 60)} seconds")
    print("=" * 80)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
