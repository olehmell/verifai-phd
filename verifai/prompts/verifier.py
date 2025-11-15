"""Prompt templates for final verification agent.

Following Anthropic best practices:
- Clear, direct, and detailed instructions
- Examples (multishot prompting)
- Role-based system prompt
"""

# System prompt defining the role
VERIFIER_SYSTEM_PROMPT = """You are an expert content verification analyst specializing in synthesizing multi-source analysis results. Your expertise includes:

- Integrating manipulation detection, narrative analysis, and fact-checking results
- Making final determinations about content credibility
- Identifying disinformation patterns
- Providing clear, structured explanations of verification findings

Your task is to synthesize all analysis results and provide a final structured assessment. Always respond in Ukrainian language for outputs."""


def build_verifier_prompt(
    content: str,
    manipulation_probability: float,
    manipulation_techniques: list,
    narrative: str,
    fact_check_results: str
) -> str:
    """
    Build a detailed prompt for final verification with examples.
    
    Args:
        content: The original content
        manipulation_probability: Probability of manipulation (0.0 to 1.0)
        manipulation_techniques: List of detected manipulation techniques
        narrative: Extracted narrative summary
        fact_check_results: Results from fact-checking analysis
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Your task is to synthesize all analysis results from a multi-agent verification system and provide a final structured assessment of Ukrainian-language content.

Context:
- You are the final step in a multi-agent fact-checking and manipulation detection pipeline
- Previous agents have analyzed manipulation techniques, extracted narratives, and fact-checked claims
- Your output will be used to inform users about content credibility
- The manipulation threshold is 0.15 - content with probability >= 0.15 is considered manipulative

Input data:
- Original content
- Manipulation analysis results
- Extracted narrative
- Fact-checking results

Instructions:
1. Review all analysis results comprehensively
2. Determine if manipulation is present (probability >= 0.15 AND techniques found)
3. Identify specific disinformation items (false claims, contradictions, etc.)
4. Ensure consistency between manipulation detection, narrative, and fact-check results
5. Provide a clear explanation of the overall assessment
6. Format your response as JSON only

Output format:
Provide your assessment as a JSON object with exactly this structure:
{{
    "manipulation": boolean (true if probability >= 0.15 AND techniques found),
    "techniques": [array of detected manipulation techniques],
    "disinfo": [array of disinformation items, format: "claim: explanation"],
    "explanation": "Overall summary in Ukrainian explaining the complete analysis"
}}

Consider:
- Manipulation threshold: 0.15
- Consistency between detected techniques and narrative
- Fact-check results and any contradictions found
- Overall credibility and accuracy of content

Examples:

<example>
Content: "Вчора відбулася конференція президента, на якій було оголошено про нові економічні ініціативи."
Manipulation probability: 0.05
Manipulation techniques: []
Narrative: "Президент оголосив про нові економічні ініціативи на конференції."
Fact-check results: "Твердження підтверджуються офіційними джерелами."
Output:
{{
    "manipulation": false,
    "techniques": [],
    "disinfo": [],
    "explanation": "Контент не містить ознак маніпуляції. Ймовірність маніпуляції становить 0.05, що нижче порогу 0.15. Твердження про конференцію та економічні ініціативи підтверджуються фактами. Контент можна вважати достовірним."
}}
</example>

<example>
Content: "УВАГА! Страшна новина! Всі знають, що цей законопроект небезпечний для демократії! Де ви були 8 років?"
Manipulation probability: 0.85
Manipulation techniques: ["fear_appeals", "bandwagon_effect", "cliche"]
Narrative: "Контент стверджує про небезпечність законопроекту для демократії та використовує тактики залякування."
Fact-check results: "Твердження про небезпеку не підтверджуються конкретними фактами. Пошукові результати показують різні думки експертів."
Output:
{{
    "manipulation": true,
    "techniques": ["fear_appeals", "bandwagon_effect", "cliche"],
    "disinfo": [
        "Небезпечність законопроекту для демократії: Твердження не підтверджуються конкретними фактами та експертними оцінками",
        "Апеляція до думки 'всіх': Використовується техніка ефекту натовпу без підтвердження"
    ],
    "explanation": "Контент містить значні ознаки маніпуляції (ймовірність 0.85). Виявлено використання технік залякування, ефекту натовпу та думко-припиняючих кліше. Фактична перевірка не підтверджує заявлену небезпеку законопроекту для демократії. Контент містить дезінформаційні елементи та використовує маніпулятивні техніки для впливу на аудиторію."
}}
</example>

<example>
Content: "Інфляція в Україні зросла до 20% у 2024 році."
Manipulation probability: 0.10
Manipulation techniques: []
Narrative: "Згідно з даними, інфляція зросла до 20% у 2024 році."
Fact-check results: "Твердження не підтверджується. Офіційні дані показують інфляцію приблизно 5.1% у 2024 році."
Output:
{{
    "manipulation": false,
    "techniques": [],
    "disinfo": [
        "Інфляція 20% у 2024 році: Офіційні статистичні дані показують інфляцію приблизно 5.1%, що значно нижче заявлених 20%"
    ],
    "explanation": "Контент не містить маніпулятивних технік, але містить фактичну помилку. Хоча ймовірність маніпуляції низька (0.10), твердження про інфляцію не відповідає офіційним даним. Інфляція у 2024 році становила приблизно 5.1%, а не 20%."
}}
</example>

<example>
Content: "Противник використовує вибіркові факти, щоб показати свою правоту. Але вони не згадують про інші важливі деталі."
Manipulation probability: 0.25
Manipulation techniques: ["selective_truth"]
Narrative: "Контент критикує противника за використання вибіркових фактів."
Fact-check results: "Твердження про вибіркові факти важко перевірити без конкретних прикладів."
Output:
{{
    "manipulation": true,
    "techniques": ["selective_truth"],
    "disinfo": [],
    "explanation": "Контент містить ознаки маніпуляції (ймовірність 0.25). Виявлено використання техніки селективної правди. Однак конкретні факти, які були пропущені, не вказані, що ускладнює перевірку. Контент використовує техніку критики противника через невизначені заяви про пропущені деталі."
}}
</example>

Now synthesize the analysis for this content:

ORIGINAL CONTENT:
{content}

MANIPULATION ANALYSIS:
- Probability: {manipulation_probability:.3f}
- Detected techniques: {manipulation_techniques}

NARRATIVE EXTRACTION:
{narrative}

FACT-CHECK RESULTS:
{fact_check_results}

Provide your final assessment as a JSON object only, without any additional text or explanation."""

    return prompt

