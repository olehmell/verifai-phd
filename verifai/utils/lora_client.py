"""
LoRA Client для використання навчених адаптерів у системі VerifAI.

Забезпечує:
1. Завантаження LoRA адаптерів для різних завдань
2. Інтеграцію з існуючими LLM клієнтами
3. Кешування моделей для ефективності
4. Fallback на базові моделі при помилках
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from peft import PeftModel, PeftConfig
from verifai.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

class LoRAClient(LLMClient):
    """Клієнт для роботи з LoRA адаптерами."""

    def __init__(
        self,
        base_model_name: str,
        lora_adapter_path: str,
        task_type: str,
        max_length: int = 1024,
        temperature: float = 0.1,
        device: Optional[str] = None
    ):
        """
        Ініціалізація LoRA клієнта.

        Args:
            base_model_name: Назва базової моделі (lapa/mamaylm)
            lora_adapter_path: Шлях до LoRA адаптера
            task_type: Тип завдання (manipulation_classifier/narrative_extractor)
            max_length: Максимальна довжина генерації
            temperature: Температура для генерації
            device: Пристрій для обчислень (auto/cuda/cpu)
        """
        self.base_model_name = base_model_name
        self.lora_adapter_path = Path(lora_adapter_path)
        self.task_type = task_type
        self.max_length = max_length
        self.temperature = temperature

        # Автовизначення пристрою
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Мапінг базових моделей
        self.base_model_paths = {
            "lapa": "ukrainian-nlp/Lapa-7B",
            "mamaylm": "maywell/MamayLM-Gemma-2-9B-it",
            "lapa_small": "ukrainian-nlp/Lapa-1.5B",
        }

        # Ініціалізуємо модель та токенізатор
        self.model = None
        self.tokenizer = None
        self.generation_config = None

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Завантаження базової моделі та LoRA адаптера."""
        try:
            base_model_path = self.base_model_paths.get(self.base_model_name)
            if not base_model_path:
                raise ValueError(f"Невідома базова модель: {self.base_model_name}")

            if not self.lora_adapter_path.exists():
                raise FileNotFoundError(f"LoRA адаптер не знайдено: {self.lora_adapter_path}")

            logger.info(f"Завантаження базової моделі: {base_model_path}")

            # Завантажуємо токенізатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Завантажуємо базову модель
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                load_in_4bit=True if self.device == "cuda" else False,
            )

            # Завантажуємо LoRA адаптер
            logger.info(f"Завантаження LoRA адаптера: {self.lora_adapter_path}")
            self.model = PeftModel.from_pretrained(
                base_model,
                str(self.lora_adapter_path),
                torch_dtype=torch.float16,
            )

            # Налаштування генерації для завдання
            self._setup_generation_config()

            logger.info(f"✅ LoRA модель успішно завантажена для {self.task_type}")

        except Exception as e:
            logger.error(f"Помилка завантаження LoRA моделі: {e}")
            raise

    def _setup_generation_config(self):
        """Налаштування конфігурації генерації залежно від завдання."""
        if self.task_type == "manipulation_classifier":
            # Для класифікації потрібна детермінованість
            self.generation_config = GenerationConfig(
                max_new_tokens=256,
                temperature=0.01,  # Дуже низька для стабільності JSON
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.0,
            )
        else:  # narrative_extractor
            # Для витягування наративу трохи більше креативності
            self.generation_config = GenerationConfig(
                max_new_tokens=200,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

    def generate_content(self, prompt: str) -> str:
        """
        Генерує відповідь для заданого промпту.

        Args:
            prompt: Вхідний промпт

        Returns:
            Згенерована відповідь
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Модель не завантажена")

        try:
            # Токенізація
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )

            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Генерація
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    use_cache=True,
                )

            # Декодування відповіді (лише нова частина)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            logger.error(f"Помилка генерації: {e}")
            raise

    def chat_completion(self, messages: list, **kwargs) -> str:
        """
        Альтернативний метод для сумісності з OpenAI API style.

        Args:
            messages: Список повідомлень (format: [{"role": "user", "content": "..."}])

        Returns:
            Відповідь моделі
        """
        # Конвертуємо messages у простий промпт
        if len(messages) == 1 and messages[0]["role"] == "user":
            prompt = messages[0]["content"]
        else:
            # Комбінуємо всі повідомлення
            prompt = "\n".join([msg["content"] for msg in messages])

        return self.generate_content(prompt)

    def __del__(self):
        """Очищення ресурсів при видаленні об'єкта."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LoRAClientManager:
    """Менеджер для управління множинними LoRA клієнтами."""

    def __init__(self, models_dir: str = "lora_models"):
        """
        Ініціалізація менеджера.

        Args:
            models_dir: Директорія з LoRA моделями
        """
        self.models_dir = Path(models_dir)
        self.clients: Dict[str, LoRAClient] = {}

    def get_client(
        self,
        model_name: str,
        task_type: str,
        **kwargs
    ) -> Optional[LoRAClient]:
        """
        Отримує або створює LoRA клієнта для конкретної задачі.

        Args:
            model_name: Назва моделі (lapa/mamaylm)
            task_type: Тип завдання
            **kwargs: Додаткові параметри для LoRAClient

        Returns:
            LoRA клієнт або None якщо адаптер не знайдено
        """
        client_key = f"{model_name}_{task_type}"

        # Повертаємо існуючий клієнт якщо є
        if client_key in self.clients:
            return self.clients[client_key]

        # Шукаємо адаптер
        adapter_path = self.models_dir / f"{model_name}_{task_type}_lora"

        if not adapter_path.exists():
            logger.warning(f"LoRA адаптер не знайдено: {adapter_path}")
            return None

        try:
            # Створюємо новий клієнт
            client = LoRAClient(
                base_model_name=model_name,
                lora_adapter_path=str(adapter_path),
                task_type=task_type,
                **kwargs
            )

            # Кешуємо клієнт
            self.clients[client_key] = client
            logger.info(f"Створено LoRA клієнт: {client_key}")

            return client

        except Exception as e:
            logger.error(f"Помилка створення LoRA клієнта {client_key}: {e}")
            return None

    def list_available_adapters(self) -> Dict[str, list]:
        """
        Повертає список доступних LoRA адаптерів.

        Returns:
            Словник з адаптерами по моделях та завданнях
        """
        adapters = {}

        if not self.models_dir.exists():
            return adapters

        for adapter_dir in self.models_dir.iterdir():
            if adapter_dir.is_dir() and "_lora" in adapter_dir.name:
                # Парсимо назву: model_task_lora
                name_parts = adapter_dir.name.replace("_lora", "").split("_")
                if len(name_parts) >= 2:
                    model = name_parts[0]
                    task = "_".join(name_parts[1:])

                    if model not in adapters:
                        adapters[model] = []
                    adapters[model].append(task)

        return adapters

    def clear_cache(self):
        """Очищає кеш завантажених клієнтів."""
        for client in self.clients.values():
            del client

        self.clients.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Кеш LoRA клієнтів очищено")