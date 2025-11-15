from google import genai
from typing import Dict, Any, List, Optional
import os
import json
import time
from verifai.utils.logging import get_logger
from verifai.utils.config import get_gemini_model
from verifai.utils.llm_client import LLMClient
from verifai.utils.lora_client import LoRAClientManager
from verifai.prompts.manipulation_classifier import (
    build_manipulation_classifier_prompt,
    VALID_TECHNIQUES
)

# Глобальний менеджер LoRA клієнтів
_lora_manager = None

THRESHOLD = 0.15

def get_lora_manager():
    """Отримує або створює LoRA менеджер."""
    global _lora_manager
    if _lora_manager is None:
        _lora_manager = LoRAClientManager()
    return _lora_manager

def manipulation_classifier_lora(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify manipulation techniques using LoRA adapter if available.
    Falls back to prompt-based approach if LoRA is not available.

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
        # Спробуємо використати LoRA адаптер
        lora_manager = get_lora_manager()

        # Перевіряємо доступні LoRA моделі (пріоритет: mamaylm > lapa)
        available_adapters = lora_manager.list_available_adapters()
        lora_client = None

        for model_name in ["mamaylm", "lapa", "lapa_small"]:
            if model_name in available_adapters:
                if "manipulation_classifier" in available_adapters[model_name]:
                    lora_client = lora_manager.get_client(
                        model_name=model_name,
                        task_type="manipulation_classifier"
                    )
                    if lora_client:
                        logger.info(f"Використовую LoRA адаптер: {model_name}")
                        break

        if lora_client:
            # Використовуємо LoRA адаптер
            prompt = f"""Проаналізуйте цей контент на предмет маніпулятивних технік.
Контент: {content}

Відповідайте у JSON форматі:"""

            response_text = lora_client.generate_content(prompt)
        else:
            # Fallback на звичайний промпт-based підхід
            logger.info("LoRA адаптер не знайдено, використовую prompt-based підхід")
            prompt = build_manipulation_classifier_prompt(content)

            # Використовуємо існуючу логіку
            llm_client: Optional[LLMClient] = state.get("_llm_client")

            if llm_client:
                response_text = llm_client.generate_content(prompt)
            else:
                from verifai.nodes.manipulation_classifier import get_gemini_client
                client = get_gemini_client()
                response = client.models.generate_content(
                    model=get_gemini_model("manipulation_classifier"),
                    contents=prompt
                )
                response_text = response.candidates[0].content.parts[0].text.strip()

        # Парсинг відповіді (однакова логіка для LoRA та prompt-based)
        try:
            # Очищення markdown
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

            # Валідація результатів
            manipulation_probability = float(classification_result.get("manipulation_probability", 0.0))
            manipulation_probability = max(0.0, min(1.0, manipulation_probability))

            raw_techniques = classification_result.get("manipulation_techniques", [])
            if not isinstance(raw_techniques, list):
                raw_techniques = []

            manipulation_techniques = [
                tech for tech in raw_techniques
                if tech in VALID_TECHNIQUES
            ]

            if manipulation_probability < THRESHOLD:
                manipulation_techniques = []

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            manipulation_probability = 0.0
            manipulation_techniques = []
            logger.log_step(
                step_name="manipulation_classifier_lora",
                content_id=content_id,
                duration=time.time() - start_time,
                error=f"Failed to parse JSON response: {str(e)}",
                metrics={"content_length": len(content), "raw_response": response_text[:200]}
            )

        duration = time.time() - start_time

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
            step_name="manipulation_classifier_lora",
            content_id=content_id,
            duration=duration,
            error=str(e),
            metrics={"content_length": len(content)}
        )
        return {
            "manipulation_probability": 0.0,
            "manipulation_techniques": []
        }
