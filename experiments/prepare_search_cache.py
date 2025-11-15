#!/usr/bin/env python3
"""
Prepare search cache for VerifAI experiments.

Runs Perplexity searches once for all test messages and saves results to a cache file.
This cache can then be reused by run_experiment.py for all model comparisons.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from verifai.utils.logging import get_logger
from verifai.utils.config import get_gemini_model
from google import genai

# Load environment variables
load_dotenv()


def load_test_messages(csv_path: Path, limit: int = 40) -> List[Dict[str, Any]]:
    """
    Load test messages from CSV file.
    
    Args:
        csv_path: Path to test CSV file
        limit: Maximum number of messages to load
        
    Returns:
        List of message dictionaries with 'id' and 'content' keys
    """
    df = pd.read_csv(csv_path)
    
    messages = []
    for idx, row in df.head(limit).iterrows():
        messages.append({
            "message_id": row.get("id", f"msg_{idx}"),
            "content": str(row.get("content", "")).strip()
        })
    
    return messages


def prepare_search_cache(
    test_messages: List[Dict[str, Any]],
    logger: Any
) -> Dict[str, Dict[str, Any]]:
    """
    Run Perplexity searches once and cache results for all messages.
    
    Args:
        test_messages: List of test messages to process
        logger: Logger instance
        
    Returns:
        Dictionary mapping message_id to cached search data (queries and results)
    """
    print("\n" + "="*60)
    print("Preparing Perplexity search cache...")
    print("="*60 + "\n")
    
    # Create Gemini client for search query generation
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    gemini_client = genai.Client(api_key=api_key)
    model_id = get_gemini_model("fact_checker")
    
    # Import here to avoid circular imports
    from verifai.nodes.fact_checker import perform_web_search
    from verifai.prompts.fact_checker import build_query_generation_prompt
    
    cache = {}
    
    for i, message in enumerate(test_messages, 1):
        message_id = message["message_id"]
        content = message["content"]
        
        if not content:
            continue
        
        print(f"  [{i}/{len(test_messages)}] Processing search for: {message_id[:8]}...")
        
        try:
            # Generate search queries using Gemini
            query_prompt = build_query_generation_prompt(content=content, narrative="")
            response = gemini_client.models.generate_content(
                model=model_id,
                contents=query_prompt
            )
            query_text = response.candidates[0].content.parts[0].text.strip()
            search_queries = [q.strip() for q in query_text.split('\n') if q.strip()]
            
            # Perform Perplexity searches
            all_search_results = []
            for query in search_queries[:3]:  # Limit to 3 queries
                results = perform_web_search(query, num_results=3)
                all_search_results.extend(results)
            
            cache[message_id] = {
                "search_queries": search_queries,
                "search_results": all_search_results
            }
            
            print(f"    ✓ Cached {len(search_queries)} queries, {len(all_search_results)} results")
            
        except Exception as e:
            print(f"    ✗ Error processing search for {message_id}: {str(e)}")
            logger.log_step(
                step_name="cache_search_error",
                content_id=message_id,
                error=str(e)
            )
            # Store empty cache entry on error
            cache[message_id] = {
                "search_queries": [],
                "search_results": []
            }
    
    print(f"\n✓ Prepared search cache for {len(cache)} messages\n")
    return cache


def main():
    """Main function to prepare search cache"""
    project_root = Path(__file__).parent.parent
    test_csv_path = project_root / "data" / "test.csv"
    cache_dir = project_root / "experiments" / "cache"
    cache_file = cache_dir / "search_cache.json"
    
    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = get_logger()
    
    print("="*60)
    print("VerifAI Search Cache Preparation")
    print("="*60)
    
    # Load test messages
    try:
        test_messages = load_test_messages(test_csv_path, limit=40)
        print(f"\nLoaded {len(test_messages)} test messages")
    except Exception as e:
        print(f"Error loading test messages: {e}")
        sys.exit(1)
    
    # Prepare search cache
    try:
        search_cache = prepare_search_cache(test_messages, logger)
        
        # Save cache to file
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "total_messages": len(test_messages),
            "cached_messages": len(search_cache),
            "cache": search_cache
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print("="*60)
        print("Search cache preparation completed!")
        print("="*60)
        print(f"\n✓ Cache saved to: {cache_file}")
        print(f"  Cached {len(search_cache)} messages")
        print(f"\nYou can now run: uv run scripts/run_experiment.py")
        
    except Exception as e:
        print(f"\n✗ Failed to prepare search cache: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()





