def process_output_mmlu_reasoning(output: str) -> str:
    if "Final Answer: " in output:
        output = output.split("Final Answer:")[1].split("\n")[0]
    return output.lower()

def process_target_mmlu_reasoning(target: str) -> str:
    return target.lower()
