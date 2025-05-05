import numpy as np
import math

from typing import Dict, List, Tuple

from .estimator import Estimator


def aggregate_probas_mean(
    keyword_token_probability: Dict[str, Dict[str, List[int]]], contribution_scores: Dict[str, Dict[str, int]] = None
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Aggregates token probabilities

    Parameters:
        keyword_token_probability (Dict[str, Dict[str, List[int]]]): token probs for keywords
    (example {
                "step1": {
                    "keyword1": [0.7, 0.8],
                    "keyword2": [0.9, 0.6, 0.5],
                },
                "step2": {
                    "keyword1": [0.5, 0.8],
                    "keyword3": [0.5, 0.9, 0.9],
                },
                ...
             }
    ),
        contribution_scores (Dict[str, Dict[str, int]]): contribution scores for keywords.
    Returns:
        Tuple[Dict[str, List[float]], Dict[str, List[float]]]: agg. keyword probs, agg. keyword contributions.
    (example {
                "keyword1": [(0.7 + 0.8) / 2, (0.5 + 0.8) / 2],
                "keyword2": [(0.9 + 0.6 + 0.5) / 3],
                "keyword3": [(0.5 + 0.9 + 0.9) / 3],
                ...
             }
    ),
    """
    return_keyword_dict = {}
    return_contribution_dict = {}
    for step, inner_dict in keyword_token_probability.items():
        for key, values in inner_dict.items():
            if len(values) == 0:
                continue
            # it is strange that min(values) was in original implementation for probas mean agg. strategy
            # value_to_add = min(values)
            value_to_add = np.mean(values)
            if key in return_keyword_dict:
                return_keyword_dict[key].append(value_to_add)
                return_contribution_dict[key].append(contribution_scores[step][key])
            else:
                return_keyword_dict[key] = [value_to_add]
                return_contribution_dict[key] = [contribution_scores[step][key]]
    return return_keyword_dict, return_contribution_dict


def weighted_sum(values: List[float]) -> float:
    """
    Computes a softmin weighted sum of the input values.

    Parameters:
        values (List[float]): values to be summed
    Returns:
        float: a softmin weighted sum
    """
    if len(values) == 1:
        return values[0]
    weights = [math.exp(-c) for c in values]
    sum_weights = sum(weights)
    normalized_weights = [w / sum_weights for w in weights]
    result = sum(w * c for w, c in zip(normalized_weights, values))
    return result


class ProbasMeanWithCoT(Estimator):
    """
    Enhances Probas-Mean aggregated probabilities strategy with reasoning steps.
    Only usabe for instruct-finetuned models with chat template support.
    Adapted from the original implementation in the paper https://arxiv.org/pdf/2502.17214
    """

    def __init__(
        self,
        name_postfix="",
    ):
        self.postfix = name_postfix
        super().__init__(
            [
                "input_texts",
                "greedy_texts",
                "reasoning_answer",
                "reasoning_keywords_probabilities",
                "reasoning_keywords_contributions",
            ],
            "sequence",
        )

    def __str__(self):
        return f"ProbasMeanWithCoT{self.postfix}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        prompts = stats["input_texts"]
        ues = []
        for i, question in enumerate(prompts):
            reasoning_answer = stats["reasoning_answer"][i]
            if reasoning_answer == "":
                ues.append(0.5)
                continue

            keyword_token_probability = stats["reasoning_keywords_probabilities"][i]
            if keyword_token_probability is None or keyword_token_probability == {}:
                ues.append(0.5)
                continue
            contribution_scores = stats["reasoning_keywords_contributions"][i]
            if contribution_scores is None or contribution_scores == {}:
                ues.append(0.5)
                continue

            probabilities, contribution_dict = aggregate_probas_mean(keyword_token_probability, contribution_scores)

            # softmin weighted sum of keywords probs
            probabilities = {key: weighted_sum(value) for key, value in probabilities.items()}
            # average of keywords contributions
            contributions = {key: sum(value) / len(value) for key, value in contribution_dict.items()}

            # CoT-UQ
            total_sum = sum(probabilities[key] * contributions[key] for key in probabilities)
            total_weight = sum(contributions[key] for key in contributions)
            if total_weight == 0:
                p_list = [v for v in probabilities.values()]
                confidence = sum(p_list) / len(p_list)
            else:
                confidence = total_sum / total_weight
            ues.append(1 - confidence)

        return np.array(ues)
