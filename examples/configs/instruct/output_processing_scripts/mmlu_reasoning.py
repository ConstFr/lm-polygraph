import re

def process_output_mmlu_reasoning(output: str) -> str:
    """
    Extract the FIRST answer letter (a/b/c/d) 
    after '### Answer:' and ignore everything after.
    """
    match = re.search(
        r"###\s*answer:\s*([a-dA-D])\b",
        output,
        flags=re.IGNORECASE
    )
    if match:
        return match.group(1).lower()
    return output.strip().lower()

def process_target_mmlu_reasoning(target: str) -> str:
    return target.strip().lower()