import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_TOKEN = os.getenv("HF_TOKEN")
print(os.getenv("HF_HOME"))

def load_model(model_path: str, device_map: str):

    llama_model = AutoModelForCausalLM.from_pretrained(model_path, token=HF_TOKEN, device_map=device_map)
    llama_model.eval()

    return llama_model


def load_tokenizer(model_path: str, add_bos_token: bool = True):
    llama_tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
    
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    return llama_tokenizer
