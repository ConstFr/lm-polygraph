import numpy as np
import math
import logging

from typing import Dict, List, Tuple

from .estimator import Estimator


log = logging.getLogger("lm_polygraph")
# logging.getLogger("httpx").setLevel(logging.WARNING)


def aggregate_probas_min(
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
            value_to_add = min(values)
            # value_to_add = np.mean(values)
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


def calculate_max_probability(log_likelihoods: np.ndarray) -> float:
        return -np.sum(log_likelihoods)


def calculate_mean_token_entropy(log_probs: np.ndarray) -> float:
    entropies = []
    for token_log_probs in log_probs:
        mask = ~np.isinf(token_log_probs)
        entropy = -np.sum(np.array(token_log_probs[mask]) * np.exp(token_log_probs[mask]))
        entropies.append(entropy)

    return np.mean(entropies)


def calculate_perplexity(log_likelihoods: np.ndarray) -> float:
    return -np.mean(log_likelihoods)


class ProbasMinWithCoT(Estimator):
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
                # "greedy_texts",
                "reasoning_answer",
                "reasoning_keywords_probabilities",
                "reasoning_keywords_contributions",
            ],
            "sequence",
        )

    def __str__(self):
        return f"ProbasMinWithCoT{self.postfix}"

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

            probabilities, contribution_dict = aggregate_probas_min(keyword_token_probability, contribution_scores)

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


class StepsMaxSequenceProbability(Estimator):
    """
    Original method leveraging clearly defined structure of the reasoning output.
    """

    def __init__(self, name_postfix="", aggregation="min"):
        self.aggregation = aggregation
        self.postfix = name_postfix + f"_{aggregation}"
        super().__init__(
            [
                "input_texts",
                "reasoning_steps",
                "greedy_log_likelihoods"
            ],
            "sequence",
        )

    def __str__(self):
        return f"StepsMaxSequenceProbability{self.postfix}"
    
    def _aggregate(self, scores):
        if self.aggregation == "mean":
            return np.mean(scores)
        elif self.aggregation == "min":
            return min(scores)
        elif self.aggregation == "max":
            return max(scores)
        elif self.aggregation == "prod":
            return np.prod(scores)
        raise NotImplementedError
    
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        prompts = stats["input_texts"]
        ues = []
        for i, question in enumerate(prompts):
            reasoning_steps = stats["reasoning_steps"][i]
            greedy_log_likelihoods = stats['greedy_log_likelihoods'][i]

            steps_probs = []
            start_index = 0

            for step_text in reasoning_steps:
                tokenized_step = stats["model"].tokenizer.tokenize(step_text)
                tokenized_step_length = len(tokenized_step)
                
                step_probs = greedy_log_likelihoods[start_index : start_index + tokenized_step_length]
                start_index += tokenized_step_length
                
                assert len(step_probs) == tokenized_step_length
                steps_probs.append(calculate_max_probability(step_probs))

            ues.append(self._aggregate(steps_probs))
        
        return np.array(ues)


class StepsMaxTokenEntropy(Estimator):
    """
    Original method leveraging clearly defined structure of the reasoning output.
    """

    def __init__(self, name_postfix="", aggregation="min"):
        self.aggregation = aggregation
        self.postfix = name_postfix + f"_{aggregation}"
        super().__init__(
            [
                "input_texts",
                "reasoning_steps",
                "greedy_log_probs"
            ],
            "sequence",
        )

    def __str__(self):
        return f"StepsMeanTokenEntropy{self.postfix}"
    
    def _aggregate(self, scores):
        if self.aggregation == "mean":
            return np.mean(scores)
        elif self.aggregation == "min":
            return min(scores)
        elif self.aggregation == "max":
            return max(scores)
        elif self.aggregation == "prod":
            return np.prod(scores)
        raise NotImplementedError
    
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        prompts = stats["input_texts"]
        ues = []
        for i, question in enumerate(prompts):
            reasoning_steps = stats["reasoning_steps"][i]
            greedy_log_probs = stats['greedy_log_probs'][i]

            steps_probs = []
            start_index = 0

            for step_text in reasoning_steps:
                tokenized_step = stats["model"].tokenizer.tokenize(step_text)
                tokenized_step_length = len(tokenized_step)
                
                step_probs = greedy_log_probs[start_index : start_index + tokenized_step_length]
                start_index += tokenized_step_length

                assert len(step_probs) == tokenized_step_length
                steps_probs.append(calculate_mean_token_entropy(step_probs))

            ues.append(self._aggregate(steps_probs))
        
        return np.array(ues)


class StepsPerplexity(Estimator):
    """
    Original method leveraging clearly defined structure of the reasoning output.
    """

    def __init__(self, name_postfix="", aggregation="min"):
        self.aggregation = aggregation
        self.postfix = name_postfix + f"_{aggregation}"
        super().__init__(
            [
                "input_texts",
                "reasoning_steps",
                "greedy_log_likelihoods"
            ],
            "sequence",
        )

    def __str__(self):
        return f"StepsPerplexity{self.postfix}"
    
    def _aggregate(self, scores):
        if self.aggregation == "mean":
            return np.mean(scores)
        elif self.aggregation == "min":
            return min(scores)
        elif self.aggregation == "max":
            return max(scores)
        elif self.aggregation == "prod":
            return np.prod(scores)
        raise NotImplementedError
    
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        prompts = stats["input_texts"]
        ues = []
        for i, question in enumerate(prompts):
            reasoning_steps = stats["reasoning_steps"][i]
            greedy_log_likelihoods = stats['greedy_log_likelihoods'][i]

            steps_probs = []
            start_index = 0

            for step_text in reasoning_steps:
                tokenized_step = stats["model"].tokenizer.tokenize(step_text)
                tokenized_step_length = len(tokenized_step)
                
                step_probs = greedy_log_likelihoods[start_index : start_index + tokenized_step_length]
                start_index += tokenized_step_length
                
                assert len(step_probs) == tokenized_step_length
                steps_probs.append(calculate_perplexity(step_probs))

            ues.append(self._aggregate(steps_probs))
        
        return np.array(ues)


class Step2QuestionNLI(Estimator):
    """
    Original method leveraging clearly defined structure of the reasoning output.
    """

    def __init__(
        self,
        aggregation="mean",
        affinity="entail",
        name_postfix="",
    ):
        self.aggregation = aggregation
        self.affinity = affinity
        self.postfix = f"{name_postfix}_{affinity}_{aggregation}"
        super().__init__(
            [
                "input_texts",
                "reasoning_steps",
                "reasoning_step_to_question_nli"
            ],
            "sequence",
        )

    def __str__(self):
        return f"Step2QuestionNLI{self.postfix}"

    def _aggregate(self, scores):
        if self.aggregation == "mean":
            return np.mean(scores)
        elif self.aggregation == "min":
            return min(scores)
        elif self.aggregation == "max":
            return max(scores)
        elif self.aggregation == "prod":
            return np.prod(scores)
        raise NotImplementedError

    def _get_scores(self, nli_scores):
        if self.affinity == "entail":
            return [nli_score['entail'] for nli_score in nli_scores]
        if self.affinity == "contra":
            return [1 - nli_score['contra'] for nli_score in nli_scores]
    
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        ues = []
        
        batch_input_texts = stats['input_texts']
        batch_reasoning_steps = stats['reasoning_steps']
        batch_reasoning_step_to_question_nli = stats['reasoning_step_to_question_nli']
        for input_text, reasoning_steps, reasoning_step_to_question_nli in zip(batch_input_texts, batch_reasoning_steps, batch_reasoning_step_to_question_nli):
            scores = self._get_scores(reasoning_step_to_question_nli)
            
            ue = self._aggregate(scores)
            ues.append(ue)
            
        return np.array(ues)


class Step2StepNLI(Estimator):
    """
    Original method leveraging clearly defined structure of the reasoning output.
    """

    def __init__(
        self,
        aggregation="mean",
        affinity="entail",
        name_postfix="",
    ):
        self.aggregation = aggregation
        self.affinity = affinity
        self.postfix = f"{name_postfix}_{affinity}_{aggregation}"
        super().__init__(
            [
                "input_texts",
                "reasoning_steps",
                "reasoning_step_to_step_nli"
            ],
            "sequence",
        )

    def __str__(self):
        return f"Step2StepNLI{self.postfix}"

    def _aggregate(self, matrix):
        if self.aggregation == "mean":
            return np.mean(matrix)
        elif self.aggregation == "min":
            return np.min(matrix)
        elif self.aggregation == "max":
            return np.max(matrix)
        elif self.aggregation == "degmat":
            W = (matrix + np.transpose(matrix)) / 2
            D = np.diag(W.sum(axis=1))
            return np.trace(len(matrix) - D) / (len(matrix) ** 2)
        raise NotImplementedError()

    def _get_matrix(self, nli_scores):
        if self.affinity == "entail":
            return np.array([[nli_score['entail'] for nli_score in step_nli_scores] for step_nli_scores in nli_scores])
        if self.affinity == "contra":
            return np.array([[1 - nli_score['contra'] for nli_score in step_nli_scores] for step_nli_scores in nli_scores])
    
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        ues = []
        
        batch_input_texts = stats['input_texts']
        batch_reasoning_steps = stats['reasoning_steps']
        batch_reasoning_step_to_step_nli = stats['reasoning_step_to_step_nli']
        for input_text, reasoning_steps, reasoning_step_to_step_nli in zip(batch_input_texts, batch_reasoning_steps, batch_reasoning_step_to_step_nli):
            matrix = self._get_matrix(reasoning_step_to_step_nli)
            
            ue = self._aggregate(matrix)
            ues.append(ue)
            
        return np.array(ues)
