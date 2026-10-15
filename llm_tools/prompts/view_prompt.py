#!/usr/bin/env python
"""
Quick viewer for a SHAP prompt in Markdown format
 .\.venv\Scripts\python.exe .\llm_tools\prompts\view_prompt.py 0
 """

from prompt_law import build_shap_prompt
import sys

# Get instance index from command line or default to 0
instance_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

print("# SHAP Prompt\n")
print(build_shap_prompt(instance_idx))

