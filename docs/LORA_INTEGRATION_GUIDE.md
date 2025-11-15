# Інструкції з інтеграції LoRA адаптерів

## Файли створено:
1. verifai/nodes/manipulation_classifier_lora.py
2. verifai/nodes/narrative_extractor_lora.py
3. verifai/utils/lora_client.py

## Для активації LoRA адаптерів:

### 1. Встановіть необхідні залежності:
```bash
pip install torch transformers peft trl datasets bitsandbytes accelerate
# Опціонально для прискорення:
pip install unsloth
```

### 2. Модифікуйте verifai/graph.py:
Додайте цей код до verifai/graph.py:

# Додайте це до verifai/graph.py для інтеграції LoRA адаптерів

# В імпортах:
try:
    from verifai.nodes.manipulation_classifier_lora import manipulation_classifier_lora
    from verifai.nodes.narrative_extractor_lora import narrative_extractor_lora
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False

def create_graph_with_lora():
    """Створює граф з LoRA адаптерами якщо доступні."""
    from langgraph.graph import StateGraph
    from verifai.state import VerifaiState
    from verifai.nodes.fact_checker import fact_checker
    from verifai.nodes.verifier import verifier

    if LORA_AVAILABLE:
        from verifai.nodes.manipulation_classifier_lora import manipulation_classifier_lora
        from verifai.nodes.narrative_extractor_lora import narrative_extractor_lora
        print("🚀 Використовую LoRA адаптери")
        manipulation_node = manipulation_classifier_lora
        narrative_node = narrative_extractor_lora
    else:
        from verifai.nodes.manipulation_classifier import manipulation_classifier
        from verifai.nodes.narrative_extractor import narrative_extractor
        print("📝 Використовую prompt-based підхід")
        manipulation_node = manipulation_classifier
        narrative_node = narrative_extractor

    # Створюємо граф
    graph = StateGraph(VerifaiState)

    # Додаємо вузли
    graph.add_node("manipulation_classifier", manipulation_node)
    graph.add_node("narrative_extractor", narrative_node)
    graph.add_node("fact_checker", fact_checker)
    graph.add_node("verifier", verifier)

    # Додаємо ребра (та сама логіка)
    from langgraph.graph import START, END
    graph.add_edge(START, "manipulation_classifier")
    graph.add_edge(START, "fact_checker")
    graph.add_edge("manipulation_classifier", "narrative_extractor")
    graph.add_edge("narrative_extractor", "verifier")
    graph.add_edge("manipulation_classifier", "verifier")
    graph.add_edge("fact_checker", "verifier")
    graph.add_edge("verifier", END)

    return graph.compile()

# Модифікуйте існуючу функцію create_graph():
def create_graph():
    """Створює оригінальний граф або з LoRA адаптерами."""
    try:
        return create_graph_with_lora()
    except Exception as e:
        print(f"Помилка створення LoRA графу: {e}")
        print("Використовую стандартний граф...")
        # Fallback на оригінальну логіку
        # [існуючий код create_graph()]
        pass


### 3. Тренування адаптерів:
```bash
# Для manipulation classifier
python scripts/train_lora_adapters.py --model mamaylm --task manipulation_classifier --epochs 3

# Для narrative extractor
python scripts/train_lora_adapters.py --model mamaylm --task narrative_extractor --epochs 3
```

### 4. Використання:
Система автоматично визначить наявність LoRA адаптерів та використає їх.
Якщо адаптери недоступні, система fallback на prompt-based підхід.

### 5. Моніторинг:
- Логи покажуть чи використовуються LoRA адаптери
- Перевірте directory lora_models/ для збережених адаптерів

## Переваги LoRA адаптерів:
- Швидша генерація (менше токенів у промпті)
- Краща консистентність результатів
- Спеціалізація на українському контенті
- Менше використання API токенів для локальних моделей
