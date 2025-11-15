"""Prompt templates for fact-checking agent.

Following Anthropic best practices:
- Clear, direct, and detailed instructions
- Examples (multishot prompting)
- Role-based system prompt
"""

# System prompt defining the role
FACT_CHECKER_SYSTEM_PROMPT = """You are an expert fact-checker and search query analyst specializing in verifying claims in Ukrainian-language content. Your expertise includes:

- Generating precise, factual search queries
- Analyzing search results for factual accuracy
- Identifying verifiable claims vs. opinions
- Cross-referencing multiple sources for verification

Your task is to create effective search queries and analyze fact-checking results. Always respond in Ukrainian language for outputs."""


def build_query_generation_prompt(content: str, narrative: str = "") -> str:
    """
    Build a detailed prompt for query generation with examples.
    
    Args:
        content: The original content
        narrative: Extracted narrative summary
        
    Returns:
        Formatted prompt string
    """
    narrative_section = f"""
Extracted narrative (if available): {narrative}""" if narrative else ""
    
    prompt = f"""Your task is to generate 2-3 specific search queries to verify factual claims and statements in Ukrainian-language content.

Context:
- These queries will be used to search the web for fact-checking purposes
- The queries should focus on verifiable facts, statistics, events, or claims
- Results will be used to validate or refute statements in the content
- Limit to 2-3 queries maximum per request

Instructions:
1. Identify factual claims, statistics, events, or key assertions in the content
2. Create specific, searchable queries that can verify these claims
3. Focus on queries that can return factual, verifiable information
4. Avoid queries about opinions or subjective statements
5. Make queries specific enough to find relevant information
6. Return only the search queries, one per line, without numbers or markers

Focus on verifying:
- Factual statements and statistics
- Mentioned events or incidents
- Key claims or assertions
- Specific dates, numbers, or facts

Examples:

<example>
Content: "Вчора відбулася конференція президента, на якій було оголошено про нові економічні ініціативи. Експерти вже назвали це революційним кроком."
Narrative: "Президент оголосив про нові економічні ініціативи на конференції."
Output:
конференція президента нові економічні ініціативи
президент економічні ініціативи оголошення
</example>

<example>
Content: "Згідно з останніми даними, інфляція в країні зросла до 15% за рік."
Narrative: "Інфляція зросла до 15% за рік."
Output:
інфляція Україна 15 відсотків 2024
інфляція статистика Україна рік
</example>

<example>
Content: "УВАГА! Страшна новина! Всі знають, що цей законопроект небезпечний для демократії!"
Narrative: "Контент стверджує про небезпечність законопроекту для демократії."
Output:
законопроект небезпечний демократія Україна
законопроект демократія критика Україна
</example>

<example>
Content: "Експерти вважають, що це найкраще рішення для економіки."
Narrative: "Експерти характеризують рішення як найкраще для економіки."
Output:
експерти економіка рішення Україна
економісти рекомендації Україна
</example>

Now generate search queries for this content:

Content: {content}{narrative_section}

Provide 2-3 search queries, one per line, without numbers or markers."""

    return prompt


def build_fact_check_analysis_prompt(
    content: str,
    search_queries: list,
    search_results: list
) -> str:
    """
    Build a detailed prompt for fact-check analysis with examples.
    
    Args:
        content: The original content
        search_queries: List of search queries used
        search_results: List of search results (dicts with 'url' and 'snippet')
        
    Returns:
        Formatted prompt string
    """
    # Format search results
    formatted_results = []
    for i, result in enumerate(search_results[:5], 1):  # Limit to 5 results
        if isinstance(result, dict):
            formatted_results.append(f"Result {i}:\nURL: {result.get('url', 'N/A')}\nSnippet: {result.get('snippet', 'N/A')}")
        else:
            formatted_results.append(f"Result {i}:\n{str(result)}")
    
    results_text = "\n\n".join(formatted_results) if formatted_results else "No search results available"
    
    prompt = f"""Your task is to analyze search results and verify the factual accuracy of statements in Ukrainian-language content.

Context:
- You are part of a fact-checking system that analyzes content for manipulation and disinformation
- The search results were obtained using specific queries to verify claims
- Your analysis will be used to determine if the content contains factual inaccuracies or disinformation
- Be objective and evidence-based in your assessment

Instructions:
1. Compare statements in the original content with information found in search results
2. Identify which claims can be verified or refuted
3. Note any factual evidence found or missing
4. Provide an overall assessment of factual accuracy
5. Cite sources used from search results
6. Keep the analysis concise and focused on facts
7. Write your response in Ukrainian

Focus on:
- Which statements can be verified or refuted
- What factual evidence was found or is missing
- Overall assessment of factual accuracy
- References to sources used

Search queries used: {', '.join(search_queries)}
Number of search results: {len(search_results)}

Search results:
{results_text}

Examples:

<example>
Original content: "Інфляція в Україні зросла до 20% у 2024 році."
Search results: Contain multiple sources showing inflation was 5.1% in 2024.
Analysis: "Твердження про інфляцію 20% не підтверджуються даними. За офіційними статистичними джерелами, інфляція в Україні у 2024 році становила приблизно 5.1%, що значно нижче заявлених 20%. Джерела: [URLs from search results]"
</example>

<example>
Original content: "Президент оголосив про нові економічні ініціативи на конференції."
Search results: Confirm the president held a conference and announced economic initiatives.
Analysis: "Твердження про оголошення економічних ініціатив підтверджується офіційними джерелами. Конференція президента дійсно відбулася, і були оголошені нові економічні ініціативи. Джерела: [URLs from search results]"
</example>

<example>
Original content: "Експерти вважають, що це найкраще рішення."
Search results: Mixed opinions - some experts support, others criticize.
Analysis: "Твердження про консенсус експертів не підтверджується. Пошукові результати показують різні думки експертів - одні підтримують рішення, інші критикують його. Не можна стверджувати про однозначну підтримку. Джерела: [URLs from search results]"
</example>

Now analyze the factual accuracy of this content:

Original content: {content}

Provide a concise analysis in Ukrainian, focusing on verified facts and evidence."""

    return prompt

