import torch
import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class AllHiddenStatesCalculator(StatCalculator):
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        # Depends on greedy generation result, and also needs the prompt text to compute prompt_len
        return ["all_hidden_states", "all_attentions", "prompt_len"], ["greedy_texts", "greedy_tokens"]

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        # doesn't support bs size > 1
        if len(texts) > 1:
            raise NotImplementedError("Batch size > 1 not supported for AllHiddenStatesCalculator")
        
        device = model.device()

        prompt_text = texts[0]
        prompt_ids = model.tokenizer(
            prompt_text, 
            return_tensors="pt", 
            padding=True, 
            add_special_tokens=True,
        )["input_ids"].to(device)  # [1, T_prompt]
        prompt_len = prompt_ids.shape[1]

        gen_ids = torch.tensor(dependencies["greedy_tokens"], dtype=torch.long).to(device)  # [1, T_gen]

        # Full sequence = prompt + generated
        full_ids = torch.cat([prompt_ids, gen_ids], dim=1)  # [1, T_total]

        fw = model(
            input_ids=full_ids,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

        # hidden_states: tuple length (L+1), each [B, T, D]  -> stack to [L+1, B, T, D]
        # hidden_states = torch.stack(list(fw.hidden_states), dim=0)
        hidden_states = torch.stack(
            [h.detach().to("cpu") for h in fw.hidden_states],
            dim=0,
        )

        # attentions: tuple length L, each [B, H, T, T] -> stack to [L, B, H, T, T]
        # attentions = torch.stack(list(fw.attentions), dim=0)
        attentions = torch.stack(
            [a.detach().to("cpu", dtype=torch.float16) for a in fw.attentions],
            dim=0,
        )

        return {
            "all_hidden_states": hidden_states.detach().cpu().numpy(),
            "all_attentions": attentions.detach().cpu().numpy(),
            "prompt_len": prompt_len,
        }
