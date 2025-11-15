from google import genai
from typing import Dict, Any, Optional
import os
import time
from verifai.utils.logging import get_logger
from verifai.utils.config import get_gemini_model
from verifai.utils.llm_client import LLMClient
from verifai.prompts.narrative_extractor import (
    build_narrative_extractor_prompt
)

# Initialize Gemini client (lazy loading)
_client = None

def get_gemini_client():
    """Get or create the Gemini client with proper error handling"""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        _client = genai.Client(api_key=api_key)
    return _client

def narrative_extractor(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract main narrative from content based on potential manipulation techniques.

    Args:
        state: Current graph state containing 'content' and 'manipulation_techniques'

    Returns:
        Updated state with 'narrative'
    """
    logger = get_logger()
    start_time = time.time()
    content = state.get("content", "")
    manipulation_techniques = state.get("manipulation_techniques", [])
    manipulation_probability = state.get("manipulation_probability", 0.0)
    content_id = state.get("content_id", "unknown")

    if not content.strip():
        logger.log_narrative_extraction(
            content_id=content_id,
            duration=time.time() - start_time,
            narrative="",
            techniques=manipulation_techniques
        )
        return {"narrative": ""}

    # Build prompt using prompt templates
    prompt = build_narrative_extractor_prompt(
        content=content,
        manipulation_techniques=manipulation_techniques,
        manipulation_probability=manipulation_probability
    )

    try:
        # Get LLM client from state or fall back to Gemini
        llm_client: Optional[LLMClient] = state.get("_llm_client")
        
        if llm_client:
            # Use provided LLM client
            narrative = llm_client.generate_content(prompt)
        else:
            # Fall back to Gemini client (backward compatibility)
            client = get_gemini_client()
            response = client.models.generate_content(
                model=get_gemini_model("narrative_extractor"),
                contents=prompt
            )
            narrative = response.candidates[0].content.parts[0].text.strip()

        duration = time.time() - start_time
        
        # Log metrics
        logger.log_narrative_extraction(
            content_id=content_id,
            duration=duration,
            narrative=narrative,
            techniques=manipulation_techniques
        )

        return {
            "narrative": narrative
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.log_step(
            step_name="narrative_extractor",
            content_id=content_id,
            duration=duration,
            error=str(e),
            input_data={"techniques": manipulation_techniques}
        )
        return {
            "narrative": f"Неможливо витягти наратив через помилку: {str(e)}"
        }