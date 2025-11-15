from google import genai
from typing import Dict, Any, Optional
import os
import json
import time
from verifai.utils.logging import get_logger
from verifai.utils.config import get_gemini_model
from verifai.utils.llm_client import LLMClient
from verifai.prompts.verifier import (
    build_verifier_prompt
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

def verifier(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formulate final response based on all previous analyses.

    Args:
        state: Complete graph state with all node outputs

    Returns:
        Updated state with final_result
    """
    logger = get_logger()
    start_time = time.time()
    content = state.get("content", "")
    manipulation_probability = state.get("manipulation_probability", 0.0)
    manipulation_techniques = state.get("manipulation_techniques", [])
    narrative = state.get("narrative", "")
    fact_check_results = state.get("fact_check_results", "")
    content_id = state.get("content_id", "unknown")

    if not content.strip():
        final_result = {
            "manipulation": False,
            "techniques": [],
            "disinfo": [],
            "explanation": "Не надано контенту для аналізу"
        }
        logger.log_verification(
            content_id=content_id,
            duration=time.time() - start_time,
            final_result=final_result
        )
        return {
            "final_result": final_result
        }

    # Build prompt using prompt templates
    prompt = build_verifier_prompt(
        content=content,
        manipulation_probability=manipulation_probability,
        manipulation_techniques=manipulation_techniques,
        narrative=narrative,
        fact_check_results=fact_check_results
    )

    try:
        # Get LLM client from state or fall back to Gemini
        llm_client: Optional[LLMClient] = state.get("_llm_client")
        
        if llm_client:
            # Use provided LLM client
            response_text = llm_client.generate_content(prompt)
        else:
            # Fall back to Gemini client (backward compatibility)
            client = get_gemini_client()
            response = client.models.generate_content(
                model=get_gemini_model("verifier"),
                contents=prompt
            )
            response_text = response.candidates[0].content.parts[0].text.strip()

        # Try to parse JSON response
        try:
            # Remove markdown code block markers if present
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]

            final_result = json.loads(response_text)

            # Validate structure
            required_keys = ["manipulation", "techniques", "disinfo", "explanation"]
            if not all(key in final_result for key in required_keys):
                raise ValueError("Missing required keys in response")

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: create structured response manually
            is_manipulation = manipulation_probability > 0.15 and len(manipulation_techniques) > 0

            final_result = {
                "manipulation": is_manipulation,
                "techniques": manipulation_techniques,
                "disinfo": [],
                "explanation": f"Аналіз завершено. Ймовірність маніпуляції: {manipulation_probability:.3f}. {narrative} {fact_check_results}"
            }

        duration = time.time() - start_time
        
        # Log metrics
        logger.log_verification(
            content_id=content_id,
            duration=duration,
            final_result=final_result
        )

        return {
            "final_result": final_result
        }

    except Exception as e:
        duration = time.time() - start_time
        # Fallback result
        is_manipulation = manipulation_probability > 0.15 and len(manipulation_techniques) > 0
        
        final_result = {
            "manipulation": is_manipulation,
            "techniques": manipulation_techniques,
            "disinfo": [],
            "explanation": f"Верифікація завершена з помилками. Ймовірність маніпуляції: {manipulation_probability:.3f}. Помилка: {str(e)}"
        }
        
        logger.log_step(
            step_name="verifier",
            content_id=content_id,
            duration=duration,
            error=str(e),
            output_data=final_result
        )

        return {
            "final_result": final_result
        }