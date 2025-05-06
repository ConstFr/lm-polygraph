import re
import string

CoT_OUTPUT_IGNORE_REGEX = re.compile(r"(?s).*Final Answer:")

def process_output_cot_hotpot(output: str) -> str:
    output = CoT_OUTPUT_IGNORE_REGEX.sub("", output).lower().strip()
    return output

def process_target_cot_hotpot(target: str) -> str:
    target = target.lower().strip()
    target = target.translate(str.maketrans("", "", string.punctuation))
    
    return target
