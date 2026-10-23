"""Session 2 demo: run the same task with different prompt types.

  python session2_prompt_types.py
  python session2_prompt_types.py "Do I need a medical certificate if my attendance is 70%?"
"""
import sys

from src.llm import generate
from src.prompts import PROMPT_TYPES

query = " ".join(sys.argv[1:]) or "I missed 30% of classes because I was ill. Can I still write exams?"

for name, template in PROMPT_TYPES.items():
    print(f"\n{'=' * 70}\n{name.upper()}\n{'-' * 70}")
    print(template.format(query=query))
    print(f"{'-' * 70}\n>>> {generate(template.format(query=query), max_tokens=300, temperature=0)}")
