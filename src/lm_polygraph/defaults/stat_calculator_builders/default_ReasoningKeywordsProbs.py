from lm_polygraph.stat_calculators.reasoning_keywords_probs import (
    ReasoningKeywordsProbs,
)


def load_stat_calculator(config, builder):
    return ReasoningKeywordsProbs(
        config.max_retries, config.max_length_cot, config.temperature
    )
