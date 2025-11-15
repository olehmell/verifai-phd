"""Prompt templates for VerifAI agents.

This module contains all prompt templates following Anthropic best practices:
- Clear, direct, and detailed instructions
- Examples (multishot prompting)
- Role-based system prompts
- Instructions in English, outputs in Ukrainian
"""

from .manipulation_classifier import (
    MANIPULATION_CLASSIFIER_SYSTEM_PROMPT,
    build_manipulation_classifier_prompt,
    MANIPULATION_TECHNIQUE_DESCRIPTIONS
)
from .narrative_extractor import (
    NARRATIVE_EXTRACTOR_SYSTEM_PROMPT,
    build_narrative_extractor_prompt
)
from .fact_checker import (
    FACT_CHECKER_SYSTEM_PROMPT,
    build_query_generation_prompt,
    build_fact_check_analysis_prompt
)
from .verifier import (
    VERIFIER_SYSTEM_PROMPT,
    build_verifier_prompt
)

__all__ = [
    "MANIPULATION_CLASSIFIER_SYSTEM_PROMPT",
    "build_manipulation_classifier_prompt",
    "MANIPULATION_TECHNIQUE_DESCRIPTIONS",
    "NARRATIVE_EXTRACTOR_SYSTEM_PROMPT",
    "build_narrative_extractor_prompt",
    "FACT_CHECKER_SYSTEM_PROMPT",
    "build_query_generation_prompt",
    "build_fact_check_analysis_prompt",
    "VERIFIER_SYSTEM_PROMPT",
    "build_verifier_prompt",
]

