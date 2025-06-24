from lm_polygraph.stat_calculators.reasoning_probs import (
    ReasoningProbsCalculator,
)


def load_stat_calculator(config, builder):
    return ReasoningProbsCalculator(
        config.max_retries, config.max_length_cot, config.temperature
    )
