from google import genai
from typing import Dict, Any, List, Optional
import os
import json
import time
from verifai.utils.logging import get_logger
from verifai.utils.config import get_gemini_model
from verifai.utils.llm_client import LLMClient
from verifai.prompts.manipulation_classifier import (
    build_manipulation_classifier_prompt,
    VALID_TECHNIQUES
)

# Initialize Gemini client (lazy loading)
_client = None

THRESHOLD = 0.5

def get_gemini_client():
    """Get or create the Gemini client with proper error handling"""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        _client = genai.Client(api_key=api_key)
    return _client

def manipulation_classifier(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify manipulation techniques in the given content using prompt-based approach.

    Args:
        state: Current graph state containing 'content'

    Returns:
        Updated state with manipulation_probability and manipulation_techniques
    """
    logger = get_logger()
    start_time = time.time()
    content = state.get("content", "")
    content_id = state.get("content_id", "unknown")

    if not content.strip():
        logger.log_classification(
            content_id=content_id,
            duration=time.time() - start_time,
            manipulation_probability=0.0,
            techniques=[],
            content_length=0
        )
        return {
            "manipulation_probability": 0.0,
            "manipulation_techniques": []
        }

    try:
        # Build prompt using prompt templates
        prompt = build_manipulation_classifier_prompt(content)

        # Get LLM client from state or fall back to Gemini
        llm_client: Optional[LLMClient] = state.get("_llm_client")
        
        if llm_client:
            # Use provided LLM client
            response_text = llm_client.generate_content(prompt)
        else:
            # Fall back to Gemini client (backward compatibility)
            client = get_gemini_client()
            response = client.models.generate_content(
                model=get_gemini_model("manipulation_classifier"),
                contents=prompt
            )
            response_text = response.candidates[0].content.parts[0].text.strip()

        # Parse JSON response
        try:
            # Remove markdown code block markers if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
            
            response_text = response_text.strip()
            classification_result = json.loads(response_text)

            # Extract and validate manipulation_probability
            manipulation_probability = float(classification_result.get("manipulation_probability", 0.0))
            manipulation_probability = max(0.0, min(1.0, manipulation_probability))  # Clamp to [0, 1]

            # Extract and validate manipulation_techniques
            raw_techniques = classification_result.get("manipulation_techniques", [])
            if not isinstance(raw_techniques, list):
                raw_techniques = []
            
            # Filter to only valid techniques
            manipulation_techniques = [
                tech for tech in raw_techniques 
                if tech in VALID_TECHNIQUES
            ]

            # If probability is below threshold, clear techniques
            if manipulation_probability < THRESHOLD:
                manipulation_techniques = []

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback: analyze manually based on content
            manipulation_probability = 0.0
            manipulation_techniques = []
            logger.log_step(
                step_name="manipulation_classifier",
                content_id=content_id,
                duration=time.time() - start_time,
                error=f"Failed to parse JSON response: {str(e)}",
                metrics={"content_length": len(content), "raw_response": response_text[:200]}
            )

        duration = time.time() - start_time
        
        # Log metrics
        logger.log_classification(
            content_id=content_id,
            duration=duration,
            manipulation_probability=manipulation_probability,
            techniques=manipulation_techniques,
            content_length=len(content)
        )
    
        return {
            "manipulation_probability": manipulation_probability,
            "manipulation_techniques": manipulation_techniques
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.log_step(
            step_name="manipulation_classifier",
            content_id=content_id,
            duration=duration,
            error=str(e),
            metrics={"content_length": len(content)}
        )
        return {
            "manipulation_probability": 0.0,
            "manipulation_techniques": []
        }
