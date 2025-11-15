"""Prompt templates for narrative extraction agent.

Following Anthropic best practices:
- Clear, direct, and detailed instructions
- Examples (multishot prompting)
- Role-based system prompt
"""

# System prompt defining the role
NARRATIVE_EXTRACTOR_SYSTEM_PROMPT = """You are an expert narrative analyst specializing in extracting core messages and claims from Ukrainian-language content. Your expertise includes:

- Identifying central narratives and key claims
- Recognizing underlying assumptions and premises
- Detecting emotional and logical manipulation patterns
- Distilling complex content into clear, concise summaries

Your task is to extract the main narrative from content, especially when manipulation techniques have been detected. Always respond in Ukrainian language."""


def build_narrative_extractor_prompt(
    content: str,
    manipulation_techniques: list = None,
    manipulation_probability: float = 0.0
) -> str:
    """
    Build a detailed prompt for narrative extraction with examples.
    
    Args:
        content: The content to analyze
        manipulation_techniques: List of detected manipulation techniques (optional)
        manipulation_probability: Probability of manipulation (0.0 to 1.0)
        
    Returns:
        Formatted prompt string
    """
    techniques_context = ""
    
    # Handle case where we have specific manipulation techniques
    if manipulation_techniques and manipulation_probability > 0.5:
        techniques_list = ", ".join(manipulation_techniques)
        techniques_context = f"""
Detected manipulation techniques: {techniques_list}
Manipulation probability: {manipulation_probability:.3f}

Important: Critically evaluate whether the narrative supports or contradicts these manipulation findings. If manipulation was detected, examine how the narrative uses these techniques."""
    
    # Handle case where we only have manipulation score (binary classifier)
    elif not manipulation_techniques and manipulation_probability > 0.5:
        techniques_context = f"""
Manipulation detected with probability: {manipulation_probability:.3f}

Important: Critically evaluate the content for potential manipulation. Even without specific technique identification, examine the narrative for persuasive or misleading patterns."""
    
    prompt = f"""Your task is to extract the main narrative or core message from Ukrainian-language content.

Context:
- This analysis is part of a fact-checking and manipulation detection system
- The extracted narrative will be used to generate search queries for fact verification
- The narrative should highlight key claims that can be fact-checked

Instructions:
1. Identify the main message or narrative being conveyed
2. Extract key claims or assertions made in the content
3. Note any planned emotional or logical impact
4. Identify underlying assumptions or premises
5. Focus on factual claims that can be verified
6. Keep the summary concise: 2-3 sentences in Ukrainian
{techniques_context}

Focus on:
- The core message or narrative being transmitted
- Key statements or assertions
- Planned emotional or logical influence
- Any underlying assumptions or premises
- Specific factual claims that could be verified

Examples:

<example>
Input: "Вчора відбулася конференція президента, на якій було оголошено про нові економічні ініціативи. Експерти вже назвали це революційним кроком."
Manipulation techniques: None
Output: "Президент оголосив про нові економічні ініціативи на конференції. Експерти характеризують це як революційний крок."
</example>

<example>
Input: "УВАГА! Страшна новина! Всі знають, що цей законопроект небезпечний для демократії! Де ви були 8 років, коли це відбувалося?"
Manipulation techniques: ["fear_appeals", "bandwagon_effect", "cliche"]
Output: "Контент стверджує про небезпечність законопроекту для демократії та використовує тактики залякування та тиску на аудиторію через апеляцію до думки більшості."
</example>

<example>
Input: "Згідно з останніми даними, інфляція в країні зросла до 15% за рік. Це на 5% більше, ніж у попередньому кварталі. Економісти пояснюють це зростанням цін на енергоносії."
Manipulation techniques: None
Output: "Згідно з даними, інфляція зросла до 15% за рік, що на 5% більше за попередній квартал. Економісти пов'язують це зі зростанням цін на енергоносії."
</example>

<example>
Input: "Противник використовує вибіркові факти, щоб показати свою правоту. Але вони не згадують про інші важливі деталі, які змінюють всю картину."
Manipulation techniques: ["selective_truth"]
Output: "Контент критикує противника за використання вибіркових фактів та стверджує про наявність важливих деталей, які змінюють інтерпретацію ситуації."
</example>

Now extract the narrative from this content:

Content to analyze:
{content}

Provide a clear, concise summary of the main narrative in 2-3 sentences in Ukrainian."""

    return prompt

