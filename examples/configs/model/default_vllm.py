from vllm import LLM, SamplingParams
import os

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def load_model(
    model_path: str, gpu_memory_utilization: float, max_new_tokens: int, logprobs: int
):
    model = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization, max_model_len=8192)
    sampling_params = SamplingParams(max_tokens=max_new_tokens, logprobs=logprobs, stop=["Question:"])
    return model, sampling_params
