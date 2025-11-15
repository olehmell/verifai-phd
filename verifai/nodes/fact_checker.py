from google import genai
from typing import Dict, Any, List, Optional
import os
from perplexity import Perplexity
import time
from verifai.utils.logging import get_logger
from verifai.utils.config import get_gemini_model
from verifai.utils.llm_client import LLMClient
from verifai.prompts.fact_checker import (
    build_query_generation_prompt,
    build_fact_check_analysis_prompt
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

def perform_web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Perform web search using Perplexity API.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        List of search results with URL and snippet
    """
    try:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable is required")
        
        # Create search domain filter - negative filters for unwanted sites
        search_domain_filter = [
            "-pinterest.com",
            "-reddit.com",
            "-quora.com"
        ]
        
        # Initialize Perplexity client
        client = Perplexity(api_key=api_key)
        
        # Create completion
        completion = client.chat.completions.create(
            model="sonar",
            messages=[
                {
                    "role": "system",
                    "content": "You are a search assistant. Provide factual search results with source URLs."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            extra_body={
                "search_domain_filter": search_domain_filter
            }
        )
        
        # Extract content and citations
        content = completion.choices[0].message.content
        citations = getattr(completion, 'citations', [])
        
        # Create results
        results = []
        if citations:
            for citation in citations[:num_results]:
                results.append({
                    "url": citation,
                    "snippet": content[:200]
                })
        else:
            # Fallback: extract from content
            if content:
                results.append({
                    "url": "perplexity_search",
                    "snippet": content[:500]
                })
        
        return results
        
    except Exception as e:
        print(f"Search error for query '{query}': {e}")
        return []

def fact_checker(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create search queries and validate statements from the content.
    
    Supports caching of Perplexity search results for experiments.

    Args:
        state: Current graph state containing 'content' and potentially 'narrative'
            - _use_search_cache: Optional bool flag to use cached search results
            - _cached_search_queries: Optional list of cached search queries
            - _cached_search_results: Optional list of cached search results

    Returns:
        Updated state with search queries and fact-check results
    """
    logger = get_logger()
    start_time = time.time()
    content = state.get("content", "")
    narrative = state.get("narrative", "")
    content_id = state.get("content_id", "unknown")
    
    # Check for cached search data
    use_cache = state.get("_use_search_cache", False)
    cached_queries = state.get("_cached_search_queries")
    cached_results = state.get("_cached_search_results")

    if not content.strip():
        logger.log_fact_checking(
            content_id=content_id,
            duration=time.time() - start_time,
            queries=[],
            results_count=0,
            fact_check_results="Немає контенту для перевірки фактів"
        )
        return {
            "search_queries": [],
            "search_results": [],
            "fact_check_results": "Немає контенту для перевірки фактів"
        }

    # Generate search queries using prompt templates
    query_prompt = build_query_generation_prompt(content=content, narrative=narrative)

    try:
        # Get LLM client from state or fall back to Gemini
        llm_client: Optional[LLMClient] = state.get("_llm_client")
        
        # Check if we should use cached search results
        if use_cache and cached_queries is not None and cached_results is not None:
            print(f"    [CACHE] Using cached search results for {content_id[:8]}...")
            search_queries = cached_queries
            all_search_results = cached_results
        else:
            # Generate search queries
            if llm_client:
                # Use provided LLM client
                query_text = llm_client.generate_content(query_prompt)
            else:
                # Fall back to Gemini client (backward compatibility)
                client = get_gemini_client()
                response = client.models.generate_content(
                    model=get_gemini_model("fact_checker"),
                    contents=query_prompt
                )
                query_text = response.candidates[0].content.parts[0].text.strip()
            search_queries = [q.strip() for q in query_text.split('\n') if q.strip()]

            # Perform searches (only if not using cache)
            all_search_results = []
            for query in search_queries[:3]:  # Limit to 3 queries
                results = perform_web_search(query, num_results=3)
                all_search_results.extend(results)

        # Generate fact-check analysis using prompt templates
        fact_check_prompt = build_fact_check_analysis_prompt(
            content=content,
            search_queries=search_queries,
            search_results=all_search_results
        )

        if llm_client:
            # Use provided LLM client
            fact_check_results = llm_client.generate_content(fact_check_prompt)
        else:
            # Fall back to Gemini client (backward compatibility)
            client = get_gemini_client()
            fact_check_response = client.models.generate_content(
                model=get_gemini_model("fact_checker"),
                contents=fact_check_prompt
            )
            fact_check_results = fact_check_response.candidates[0].content.parts[0].text.strip()

        duration = time.time() - start_time
        
        # Log metrics
        logger.log_fact_checking(
            content_id=content_id,
            duration=duration,
            queries=search_queries,
            results_count=len(all_search_results),
            fact_check_results=fact_check_results
        )

        return {
            "search_queries": search_queries,
            "search_results": all_search_results,
            "fact_check_results": fact_check_results
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.log_step(
            step_name="fact_checker",
            content_id=content_id,
            duration=duration,
            error=str(e)
        )
        return {
            "search_queries": [],
            "search_results": [],
            "fact_check_results": f"Перевірка фактів не вдалася через помилку: {str(e)}"
        }