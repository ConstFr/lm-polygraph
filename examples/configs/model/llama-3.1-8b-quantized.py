import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_TOKEN = os.getenv("HF_TOKEN")

def load_model(model_path: str, device_map: str):
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    llama_model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                       quantization_config=quantization_config, 
                                                       token=HF_TOKEN)
    llama_model.eval()

    return llama_model


def load_tokenizer(model_path: str, add_bos_token: bool = True):
    llama_tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
    
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    return llama_tokenizer
