"""Prompt templates for manipulation classification agent.

Following Anthropic best practices:
- Clear, direct, and detailed instructions
- Examples (multishot prompting)
- Role-based system prompt
"""

# Human-readable descriptions for each technique
MANIPULATION_TECHNIQUE_DESCRIPTIONS = {
    "emotional_manipulation": "Емоційна маніпуляція - Використовує експресивну мову з сильним емоційним забарвленням або ейфорійний тон для підняття бойового духу та впливу на думку",
    "fear_appeals": "Апеляції до страху - Грає на страхах, стереотипах чи упередженнях. Включає тактики страху, невизначеності та сумнівів (FUD)",
    "bandwagon_effect": "Ефект натовпу - Використовує загальні позитивні концепції або заклики до мас ('всі так думають') для заохочення згоди",
    "selective_truth": "Селективна правда - Використовує логічні помилки, такі як вибірковий відбір фактів, whataboutism для відвернення критики, або створення опудальних аргументів",
    "cliche": "Думко-припиняючі кліше - Використовує формульні фрази, розроблені для припинення критичного мислення та завершення дискусії. Приклади: 'Все не так однозначно', 'Де ви були 8 років?'"
}

VALID_TECHNIQUES = list(MANIPULATION_TECHNIQUE_DESCRIPTIONS.keys())

# System prompt defining the role
MANIPULATION_CLASSIFIER_SYSTEM_PROMPT = """You are an expert content analyst specializing in detecting manipulation techniques and disinformation in Ukrainian-language content. Your expertise includes:

- Identifying psychological manipulation tactics
- Recognizing disinformation patterns
- Analyzing emotional and logical manipulation techniques
- Maintaining objectivity and critical thinking in analysis

Your task is to analyze content for manipulation techniques and provide structured, accurate assessments. Always respond in Ukrainian language."""


def build_manipulation_classifier_prompt(content: str) -> str:
    """
    Build a detailed prompt for manipulation classification with examples.
    
    Args:
        content: The content to analyze
        
    Returns:
        Formatted prompt string
    """
    techniques_descriptions_text = "\n".join([
        f"- {name}: {desc}" 
        for name, desc in MANIPULATION_TECHNIQUE_DESCRIPTIONS.items()
    ])
    
    prompt = f"""Your task is to analyze Ukrainian-language content for manipulation techniques and disinformation patterns.

Context:
- You are analyzing content that may contain psychological manipulation or disinformation
- The results will be used by a fact-checking system to identify potentially misleading content
- This analysis is part of a multi-agent verification pipeline

Available manipulation techniques:
{techniques_descriptions_text}

Instructions:
1. Carefully read the entire content
2. Identify any manipulation techniques from the list above
3. Assess the overall manipulation probability (0.0 = no manipulation, 1.0 = strong manipulation)
4. Only include techniques that are clearly present in the content
5. Be objective and critical - do not overclassify or underclassify
6. If no manipulation is detected, set manipulation_probability = 0.0 and manipulation_techniques = []

Output format:
Provide your analysis as a JSON object with exactly this structure:
{{
    "manipulation_probability": float (0.0 to 1.0),
    "manipulation_techniques": [array of technique names from the list]
}}

Examples:

<example>
Input: "Всі нормальні люди підтримують цю ідею. Якщо ви не згодні, ви просто не розумієте ситуацію."
Analysis:
- Uses bandwagon effect ("всі нормальні люди")
- Uses emotional manipulation through pressure and exclusion
- Low manipulation probability as it's relatively mild

Output:
{{
    "manipulation_probability": 0.4,
    "manipulation_techniques": ["bandwagon_effect", "emotional_manipulation"]
}}
</example>

<example>
Input: "Вчора відбулася конференція президента. Було обговорено нові ініціативи щодо економіки."
Analysis:
- Factual, neutral statement
- No manipulation techniques detected

Output:
{{
    "manipulation_probability": 0.0,
    "manipulation_techniques": []
}}
</example>

<example>
Input: "УВАГА! Страшна новина! Всі знають, що це небезпечно! Де ви були 8 років, коли це відбувалося? Все не так однозначно..."
Analysis:
- Uses fear appeals ("страшна новина", "небезпечно")
- Uses bandwagon effect ("всі знають")
- Uses cliche phrases ("Де ви були 8 років?", "Все не так однозначно")
- High manipulation probability

Output:
{{
    "manipulation_probability": 0.85,
    "manipulation_techniques": ["fear_appeals", "bandwagon_effect", "cliche"]
}}
</example>

<example>
Input: "Противник використовує вибіркові факти, щоб показати свою правоту. Але вони не згадують про інші важливі деталі. Це класичний whataboutism."
Analysis:
- Mentions selective truth and whataboutism as concepts
- The content itself describes manipulation but doesn't necessarily use it
- Low manipulation probability

Output:
{{
    "manipulation_probability": 0.2,
    "manipulation_techniques": ["selective_truth"]
}}
</example>

Now analyze this content:

Content to analyze:
{content}

Provide your analysis as a JSON object only, without any additional text or explanation."""

    return prompt

