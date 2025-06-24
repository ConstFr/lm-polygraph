import re
import string

# COT_OUTPUT_IGNORE_REGEX = re.compile(r"(?s).*Final Answer:")
COT_TARGET_IGNORE_REGEX = re.compile(r"(?s).*\n#### ")

def process_output_cot_gsm8k(output: str) -> str:
    output.replace(",", "")
    if "Final Answer: " in output:
        output = output.split("Final Answer:")[1].split("\n")[0]
    match = re.findall(r'\d+\.?\d*', output)
    if match:
        number = float(match[0])
        return str(int(number))
    output = output.translate(str.maketrans("", "", string.punctuation))
    return output

def process_target_cot_gsm8k(target: str) -> str:
    target = COT_TARGET_IGNORE_REGEX.sub("", target).lower().strip()
    target = target.translate(str.maketrans("", "", string.punctuation))
    
    return target
