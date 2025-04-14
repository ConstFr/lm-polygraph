import re
import torch
import numpy as np
from collections import defaultdict

from typing import Dict, List, Tuple, Optional

from .embeddings import get_embeddings_from_output
from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


cot_instruction = """
Please reason the following question step by step. Label each reasoning step as "Step i:", where "i" is the step number.
You need to ensure that each step builds on the previous one and contributes meaningfully toward reaching the final answer.
Once you finish all steps, put your final answer on a separate line after the reasoning steps, starting with "Final Answer:" (do not label it as a step).

Question: <QUESTION>
Response: Let's think step by step.
"""

keywords_extraction_instruction = ''' 
You will be provided with a question and a multi-step response containing reasoning steps. 
For each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.
You also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.
If you find more than one keyword in a specific step, separate them with “;”.
If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:

Question:
<QUESTION>
Multi-Step Response:
<RESPONSE>
Keywords for Each Reasoning Step:
'''


def is_effectively_empty(obj):
    
    if obj is None:
        return True

    if isinstance(obj, (int, float)) and obj == 0:
        return True

    if obj == "":
        return True

    if isinstance(obj, list):
        return all(is_effectively_empty(item) for item in obj)
    
    if isinstance(obj, dict):
        if len(obj) == 0: 
            return True
        return all(is_effectively_empty(value) for value in obj.values())
    return False


def parse_response_to_dict(response: str) -> Tuple[Optional[str], Dict[str, str], Optional[str]]:
    """
    Parse model reasoning output to highlight: reasoning answer, reasoning steps, reasoning output without answer.

    Parameters:
        response (str): reasoning output.
    Returns:
        Tuple[Optional[str], Dict[str, str], Optional[str]]: 
            - final answer (str or None),
            - dictionary of steps (e.g., {"Step 1": "Step 1: ..."}),
            - response before final answer (str or None)
    """
    steps: Dict[str, str] = {}
    final_answer: Optional[str] = None

    # Match Final Answer
    match = re.search(r"Final Answer:\s*(.+?)\s*(?=(\n|$))", response, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        response_before_final_answer = response[:match.end()].strip()
    else:
        return None, {}, None

    # Match Steps
    matches = list(re.finditer(r"(Step \d+):", response_before_final_answer))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response_before_final_answer)
        segment = response[start:end].strip()
        steps[match.group(1)] = segment

    return_response = response_before_final_answer
    return final_answer, steps, return_response


def match_final_answer_token_ids(tokenizer, original_tokens, response_tokens, original_token_ids):
    # caution
    final_answer_tokens = tokenizer.tokenize("Final answer:")

    end_index = None
    end_index_original = None

    for i in range(len(response_tokens) - len(final_answer_tokens) + 1):
        if response_tokens[i : i + len(final_answer_tokens)] == final_answer_tokens:
            start_index = i
            end_index = i + len(final_answer_tokens)
            break

    if end_index == None or end_index + 1 == len(response_tokens):
        return None, None

    for i in range(len(original_tokens) - len(final_answer_tokens) + 1):
        if original_tokens[i : i + len(final_answer_tokens)] == final_answer_tokens:
            end_index_original = i + len(final_answer_tokens)
            break

    if end_index_original == None:
        return None, None

    if response_tokens[end_index] in ["▁", "Ġ", tokenizer.tokenize(" ")]:
        end_index += 1
        end_index_original += 1

    target_tokens = response_tokens[end_index:]

    final_answer_token_ids = original_token_ids[end_index_original : end_index_original + len(target_tokens)]

    return end_index_original, final_answer_token_ids.data.cpu().numpy()


def predict(prompt, model, tokenizer, max_length_cot, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    generate_ids = model.generate(
        **inputs, 
        max_new_tokens = max_length_cot,
        temperature=temperature, 
        pad_token_id=tokenizer.eos_token_id)
    generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
    infer_res = tokenizer.decode(generate_ids)
    return infer_res


def step_exacts_2_list(response):
    # Split response into lines and filter out empty lines
    lines = response.splitlines()
    lines = [line for line in lines if line.strip()]

    keywords_by_step = []
    contributions_by_step = []
    valid_response_text = []

    for line in lines:
        # Match lines starting with "Step X:"
        match = re.search(r"Step \d+: (.+)", line)
        if match:
            if "(/" not in line or "/)" not in line:
                continue  # Skip invalid lines

            # Extract keywords with contributions
            keywords_w_contribution = match.group(1).split("; ")

            # Check for valid format and skip invalid lines
            if any("(/" not in key_w_c or "/)" not in key_w_c for key_w_c in keywords_w_contribution):
                continue

            try:
                # Extract keywords and contributions
                keywords = [key_w_c.split("(/")[0].strip() for key_w_c in keywords_w_contribution]
                contributions = [int(key_w_c.split("(/")[1].split("/)")[0].strip()) for key_w_c in keywords_w_contribution]
            except ValueError:
                return False  # Return False if contributions cannot be converted to int

            for i in contributions:
                if i > 10:
                    return False

            keywords_by_step.append(keywords)
            contributions_by_step.append(contributions)
            valid_response_text.append(line)  # Add valid lines from the original response

    # If no valid lines are found, return False
    if not valid_response_text:
        return False

    return "\n".join(valid_response_text), keywords_by_step, contributions_by_step


def find_subsequence_position(sub_sequence, long_sequence):
    len_long = long_sequence.size(0)
    len_sub = len(sub_sequence) 

    sub_sequence_tensor = torch.tensor(sub_sequence, device=long_sequence.device)
    
    for i in range(len_long - len_sub + 1):
        if torch.equal(long_sequence[i:i + len_sub], sub_sequence_tensor):
            return i 
    return -1


def clean_words(word):
    # TODO forward space token
  return word.replace(" ", "").replace(".", "").replace("\"", "").replace("\n", "").replace("_", "").replace("Ġ", "").lower()


def find_token_indices(tokens, word):
    word_len = len(word.replace(" ", ""))
    
    for start_index in range(len(tokens)):
        combined_text = ""
        end_index = start_index       
        while end_index < len(tokens) and len(combined_text) < word_len:
            combined_text += tokens[end_index]
            if clean_words(combined_text) == clean_words(word):
                return start_index, end_index
            end_index += 1
    
    return -1, -1


def is_word_in_sentence(sentence, word):
    pattern = re.escape(word)
    match = re.search(pattern, sentence, re.IGNORECASE)
    return True if match else False


class ReasoningKeywordsProbs(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
        * model output for reasoning enhanced input,
        * model answer for reasoning enhanced input,
        * token probabilities for `reasoning_answer`,
        * keywords from `reasoning_output`,
        * probabilities for `reasoning_keywords`,
        * contributions for `reasoning_keywords`,
        * step-wise token indices for `reasoning_keywords`,
        * token indices for `reasoning_keywords`.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return [
            "reasoning_output",
            "reasoning_answer",
            "reasoning_answer_tokens_probs",
            "reasoning_keywords",
            "reasoning_keywords_probabilities",
            "reasoning_keywords_contributions",
            "reasoning_keywords_token_ids",
            "reasoning_answer_token_ids",
        ], ["input_texts"]

    def __init__(self, max_retries=5, max_length_cot=128, temperature=1):
        super().__init__()
        self.max_retries = max_retries
        self.max_length_cot = max_length_cot
        self.temperature = temperature

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of reasoning enhanced process.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'reasoning_output' (List[str]): model output for reasoning enhanced input,
                - 'reasoning_answer' (List[str]): model answer for reasoning enhanced input,
                - 'reasoning_answer_tokens_probs' (List[str]): token probabilities for `reasoning_answer`,
                - 'reasoning_keywords' (List[str]): keywords from `reasoning_output`,
                - 'reasoning_keywords_probabilities' (List[Dict[str, Dict[str, List[int]]]]): probabilities for `reasoning_keywords`,
                - 'reasoning_keywords_contributions' (List[Dict[str, Dict[str, int]]]): contributions for `reasoning_keywords`,
                - 'reasoning_keywords_token_ids' (List[Dict[str, Dict[str, List[int]]]]): step-wise token indices for `reasoning_keywords`,
                - 'reasoning_answer_token_ids' (List[Dict[str, List[int]]]): token indices for `reasoning_keywords`.
        """
        result_dict = defaultdict(list)
        for question in texts:
            cot_prompt = cot_instruction.replace("<QUESTION>", question)

            inputs = model.tokenizer(cot_prompt, return_tensors="pt")
            inputs = {key: value.to(model.model.device) for key, value in inputs.items()}
            n_of_retries = 0
            while n_of_retries < self.max_retries:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_length_cot,
                    temperature=self.temperature,
                    pad_token_id=model.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # generated token ids for the question enchanced with CoT.
                generated_ids = outputs.sequences[0][len(inputs["input_ids"][0]) : -1]
                # generated text for the question enchaced with CoT
                to_parse = model.tokenizer.decode(generated_ids, skip_special_tokens=True)

                llm_answer, steps_dict, response = parse_response_to_dict(to_parse)
                if generated_ids.size(0) >= self.max_length_cot:
                    # log.debug(f'New Reasoning Tokens Are Too Much, Current try is {n_of_retries + 1}')
                    n_of_retries += 1
                    continue
                elif generated_ids.size(0) == 0:
                    # log.debug(f'New Reasoning Tokens Are Null, Current try is {n_of_retries + 1}')
                    n_of_retries += 1
                    continue
                elif llm_answer is None or llm_answer in ["", " "]:
                    # log.debug(f'New Reasoning Tokens Are None, Current try is {n_of_retries + 1}')
                    n_of_retries += 1
                    continue
                
                # reasoning tokens without final answer
                response_tokens = model.tokenizer.tokenize(response)
                # reasoning token ids without final answer
                response_token_ids = model.tokenizer.convert_tokens_to_ids(response_tokens)
                # full reasoning tokens
                original_tokens = model.tokenizer.convert_ids_to_tokens(generated_ids)

                probabilities = [
                    {i: p for i, p in enumerate(prob[0]) if p > 0}
                    for prob in [torch.softmax(score, dim=1).tolist() for score in outputs.scores]
                ]

                final_answer_probabilities = {}
                final_answer_token_ids = {}
                answer_start_indice, answer_token_ids = match_final_answer_token_ids(
                    model.tokenizer,
                    original_tokens,
                    response_tokens,
                    generated_ids,
                )
                if answer_start_indice == None:
                    # log.debug(f'Cannot locate the Final Answer, Current try is {n_of_retries + 1}')
                    n_of_retries += 1
                    continue
                answer_probs = []
                flag = False
                for j, token_id in enumerate(answer_token_ids):
                    idxx = j + answer_start_indice
                    if token_id not in probabilities[idxx].keys():
                        flag = True
                        break
                    answer_probs.append(probabilities[idxx][token_id])
                if flag:
                    # log.debug(f'Cannot locate the Final Answer Token Probability, Current try is {n_of_retries + 1}')
                    n_of_retries += 1
                    continue
                final_answer_probabilities[llm_answer] = answer_probs
                final_answer_token_ids[llm_answer] = answer_token_ids.tolist()
                
                # exacts_prompt = get_step_exact_tokens(args, q, response)
                keywords_extraction_prompt = keywords_extraction_instruction.replace('<QUESTION>', question).replace('<RESPONSE>', response)
                keywords_extraction_prompt_output = predict(keywords_extraction_prompt, model, model.tokenizer, self.max_length_cot, self.temperature)

                if "NO ANSWER" in keywords_extraction_prompt_output:
                    # log.debug(f'Exact Tokens Have NO ANSWER, Current try is {n_of_retries + 1}')
                    n_of_retries += 1
                    continue
                parsed_keywords_output = step_exacts_2_list(keywords_extraction_prompt_output)
                if not parsed_keywords_output:
                    # log.debug(f'Exact Tokens Have no contribution scores, Current try is {n_of_retries + 1}')
                    n_of_retries += 1
                    continue
                extracted_keywords, keywords_list, contributions_list = parsed_keywords_output
                if len(keywords_list) == 0:
                    # log.debug(f'Cannot Exract Effective Keywords, Current try is {n_of_retries + 1}')
                    n_of_retries += 1
                    continue

                if len(steps_dict) > len(keywords_list):
                    # log.debug(f'Len of keywords list doesn\'t match the len of step dict, Current try is {n_of_retries + 1}')
                    n_of_retries += 1
                    continue

                keywords_probabilities = {}
                keywords_contributions = {}
                keywords_token_ids = {}
                for step_idx, (step_name, step_text) in enumerate(steps_dict.items()):
                    # # Skip the Final Answer
                    keywords = keywords_list[step_idx]
                    contributions = contributions_list[step_idx]
                    if len(keywords) == 1 and keywords[0] == "NO ANSWER":
                        continue
                    step_tokens = model.tokenizer.tokenize(step_text)
                    space_token = model.tokenizer.tokenize(" ")
                    processed_step_tokens = [
                        (token[1:] if token.startswith(space_token) else token)
                        for token in step_tokens
                    ]
                    step_token_ids = model.tokenizer.convert_tokens_to_ids(step_tokens)
                    start_position = find_subsequence_position(step_token_ids[1:-2], generated_ids) - 1
                    step_token_ids = generated_ids[start_position : start_position + len(step_tokens)]
                    keywords_probabilities_dict = {}
                    keywords_contributions_dict = {}
                    keywords_token_ids_dict = {}
                    for keyword_idx, keyword in enumerate(keywords):

                        keyword_probs = []
                        keyword_token_ids = []
                        if is_word_in_sentence(step_text, keyword) is not True:
                            # log.debug(f"\n{step_name}-Keyword-{keyword_idx} Does not appear in the Step Text")
                            continue
                        keyword_token_start_idx, keyword_token_end_idx = find_token_indices(
                            processed_step_tokens, keyword
                        )
                        keyword_token_ids = generated_ids[
                            start_position + keyword_token_start_idx : start_position + keyword_token_end_idx + 1
                        ]
                        keyword_token_ids = keyword_token_ids.data.cpu().numpy()

                        for j, token_id in enumerate(keyword_token_ids):
                            idxx = start_position + keyword_token_start_idx + j
                            keyword_probs.append(probabilities[idxx][token_id])
                        keywords_probabilities_dict[keyword] = keyword_probs
                        keywords_contributions_dict[keyword] = int(contributions[keyword_idx])
                        keywords_token_ids_dict[keyword] = keyword_token_ids.tolist()

                    keywords_probabilities[step_name] = keywords_probabilities_dict
                    keywords_contributions[step_name] = keywords_contributions_dict
                    keywords_token_ids[step_name] = keywords_token_ids_dict

                if is_effectively_empty(keywords_probabilities):
                    # log.debug(f'Token Probability from All Steps are All None, Current try is {n_of_retries + 1}')
                    n_of_retries += 1
                    continue
                
                # Dict[str, np.ndarray]: dictionary with the following items:
                # - 'reasoning_output' (List[str]): model output for reasoning enhanced input,
                # - 'reasoning_answer' (List[str]): model answer for reasoning enhanced input,
                # - 'reasoning_answer_tokens_probs' (List[str]): token probabilities for `reasoning_answer`,
                # - 'reasoning_keywords' (List[str]): keywords from `reasoning_output`,
                # - 'reasoning_keywords_probabilities' (List[Dict[str, Dict[str, List[int]]]]): probabilities for `reasoning_keywords`,
                # - 'reasoning_keywords_contributions' (List[Dict[str, Dict[str, int]]]): contributions for `reasoning_keywords`,
                # - 'reasoning_keywords_token_ids' (List[Dict[str, Dict[str, List[int]]]]): step-wise token indices for `reasoning_keywords`,
                # - 'reasoning_answer_token_ids' (List[Dict[str, List[int]]]): token indices for `reasoning_keywords`.
                
                result_dict["reasoning_output"].append(response)
                result_dict["reasoning_answer"].append(llm_answer)
                result_dict["reasoning_answer_tokens_probs"].append(final_answer_probabilities)
                result_dict["reasoning_keywords"].append(extracted_keywords)
                result_dict["reasoning_keywords_probabilities"].append(keywords_probabilities)
                result_dict["reasoning_keywords_contributions"].append(keywords_contributions)
                result_dict["reasoning_keywords_token_ids"].append(keywords_token_ids)
                result_dict["reasoning_answer_token_ids"].append(final_answer_token_ids)
                break
                
            if n_of_retries >= self.max_retries:
                # log.debug(f'#####The Following Question:#####\n{q}\nHas no Meaningful Answer & Explanations, Record and Skip')
                result_dict["reasoning_output"].append(response)
                result_dict["reasoning_answer"].append(llm_answer)
                result_dict["reasoning_answer_tokens_probs"].append(None)
                result_dict["reasoning_keywords"].append(None)
                result_dict["reasoning_keywords_probabilities"].append(None)
                result_dict["reasoning_keywords_contributions"].append(None)
                result_dict["reasoning_keywords_token_ids"].append(None)
                result_dict["reasoning_answer_token_ids"].append(None)

        return result_dict
