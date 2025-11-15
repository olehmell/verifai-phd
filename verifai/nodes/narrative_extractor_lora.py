from google import genai
from typing import Dict, Any, Optional
import os
import time
from verifai.utils.logging import get_logger
from verifai.utils.config import get_gemini_model
from verifai.utils.llm_client import LLMClient
from verifai.utils.lora_client import LoRAClientManager
from verifai.prompts.narrative_extractor import (
    build_narrative_extractor_prompt
)

# Глобальний менеджер LoRA клієнтів
_lora_manager = None

def get_lora_manager():
    """Отримує або створює LoRA менеджер."""
    global _lora_manager
    if _lora_manager is None:
        _lora_manager = LoRAClientManager()
    return _lora_manager

def narrative_extractor_lora(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract narrative using LoRA adapter if available.
    Falls back to prompt-based approach if LoRA is not available.

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

    try:
        # Спробуємо використати LoRA адаптер
        lora_manager = get_lora_manager()

        # Перевіряємо доступні LoRA моделі
        available_adapters = lora_manager.list_available_adapters()
        lora_client = None

        for model_name in ["mamaylm", "lapa", "lapa_small"]:
            if model_name in available_adapters:
                if "narrative_extractor" in available_adapters[model_name]:
                    lora_client = lora_manager.get_client(
                        model_name=model_name,
                        task_type="narrative_extractor"
                    )
                    if lora_client:
                        logger.info(f"Використовую LoRA адаптер для narrative: {model_name}")
                        break

        if lora_client:
            # Використовуємо LoRA адаптер
            techniques_context = ""
            if manipulation_techniques and manipulation_probability > 0.15:
                techniques_list = ", ".join(manipulation_techniques)
                techniques_context = f"\nВиявлені техніки маніпуляції: {techniques_list}"

            prompt = f"""Витягніть головний наратив з контенту.{techniques_context}
Контент: {content}

Дайте стислий опис українською мовою в 2-3 реченнях:"""

            narrative = lora_client.generate_content(prompt)
        else:
            # Fallback на звичайний промпт-based підхід
            logger.info("LoRA адаптер для narrative не знайдено, використовую prompt-based підхід")
            prompt = build_narrative_extractor_prompt(
                content=content,
                manipulation_techniques=manipulation_techniques,
                manipulation_probability=manipulation_probability
            )

            llm_client: Optional[LLMClient] = state.get("_llm_client")

            if llm_client:
                narrative = llm_client.generate_content(prompt)
            else:
                from verifai.nodes.narrative_extractor import get_gemini_client
                client = get_gemini_client()
                response = client.models.generate_content(
                    model=get_gemini_model("narrative_extractor"),
                    contents=prompt
                )
                narrative = response.candidates[0].content.parts[0].text.strip()

        duration = time.time() - start_time

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
            step_name="narrative_extractor_lora",
            content_id=content_id,
            duration=duration,
            error=str(e),
            input_data={"techniques": manipulation_techniques}
        )
        return {
            "narrative": f"Неможливо витягти наратив через помилку: {str(e)}"
        }
