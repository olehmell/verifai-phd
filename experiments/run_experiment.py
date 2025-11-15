#!/usr/bin/env python3
"""
Experiment runner for comparing different LLM models on VerifAI test dataset.

Runs 40 test messages through the VerifAI pipeline using different models
and saves results to separate JSON files for analysis.

Requires search cache prepared by prepare_search_cache.py first.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from verifai.utils.llm_client import create_llm_client
from verifai import create_graph
from verifai.utils.logging import get_logger

# Load environment variables
load_dotenv()


def load_model_config(config_path: Path) -> List[Dict[str, Any]]:
    """
    Load model configurations from JSON file.
    
    Args:
        config_path: Path to model configuration JSON file
        
    Returns:
        List of model configurations
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config.get("models", [])


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


def load_search_cache(cache_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load search cache from JSON file.
    
    Args:
        cache_file: Path to cache JSON file
        
    Returns:
        Dictionary mapping message_id to cached search data
    """
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Search cache file not found: {cache_file}\n"
            f"Please run 'uv run scripts/prepare_search_cache.py' first."
        )
    
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    cache = cache_data.get("cache", {})
    timestamp = cache_data.get("timestamp", "unknown")
    cached_messages = cache_data.get("cached_messages", len(cache))
    
    print(f"  Loaded cache from {timestamp}")
    print(f"  Cache contains {cached_messages} messages")
    
    return cache


def run_experiment_for_model(
    model_config: Dict[str, Any],
    test_messages: List[Dict[str, Any]],
    logger: Any,
    search_cache: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Run experiment for a single model.
    
    Args:
        model_config: Model configuration dictionary
        test_messages: List of test messages to process
        logger: Logger instance
        
    Returns:
        Dictionary containing experiment results
    """
    model_name = model_config["name"]
    model_type = model_config["type"]
    model_id = model_config["model_id"]
    base_url = model_config.get("base_url")
    
    print(f"\n{'='*60}")
    print(f"Running experiment for model: {model_name}")
    print(f"Type: {model_type}, Model ID: {model_id}")
    print(f"{'='*60}\n")
    
    try:
        # Create LLM client
        llm_client = create_llm_client(model_type, model_id, base_url)
        
        # Create graph with LLM client
        graph = create_graph(llm_client=llm_client, use_binary_classifier=True)
        
        results = []
        
        for i, message in enumerate(test_messages, 1):
            message_id = message["message_id"]
            content = message["content"]
            
            if not content:
                print(f"  [{i}/{len(test_messages)}] Skipping empty message: {message_id}")
                continue
            
            print(f"  [{i}/{len(test_messages)}] Processing message: {message_id[:8]}...")
            
            try:
                # Prepare state with cache if available
                state_input = {
                    "content": content,
                    "content_id": message_id
                }
                
                # Add cached search data if available
                if search_cache and message_id in search_cache:
                    cached_data = search_cache[message_id]
                    state_input["_use_search_cache"] = True
                    state_input["_cached_search_queries"] = cached_data["search_queries"]
                    state_input["_cached_search_results"] = cached_data["search_results"]
                
                # Run analysis
                result = graph.invoke(state_input)
                
                # Extract final result
                final_result = result.get("final_result", {})
                
                results.append({
                    "message_id": message_id,
                    "content": content,
                    "final_result": final_result
                })
                
                print(f"    ✓ Completed - Manipulation: {final_result.get('manipulation', False)}")
                
            except Exception as e:
                print(f"    ✗ Error processing message {message_id}: {str(e)}")
                logger.log_step(
                    step_name="experiment_error",
                    content_id=message_id,
                    error=str(e),
                    input_data={"model": model_name}
                )
                
                # Add error result
                results.append({
                    "message_id": message_id,
                    "content": content,
                    "final_result": {
                        "manipulation": False,
                        "techniques": [],
                        "disinfo": [],
                        "explanation": f"Error during processing: {str(e)}"
                    },
                    "error": str(e)
                })
        
        return {
            "model": model_name,
            "model_type": model_type,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "total_messages": len(test_messages),
            "processed_messages": len(results),
            "results": results
        }
        
    except Exception as e:
        print(f"  ✗ Failed to initialize model {model_name}: {str(e)}")
        logger.log_step(
            step_name="experiment_init_error",
            content_id="experiment",
            error=str(e),
            input_data={"model": model_name}
        )
        raise


def main():
    """Main experiment runner function"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "experiments" / "model_config.json"
    test_csv_path = project_root / "data" / "test.csv"
    cache_file = project_root / "experiments" / "cache" / "search_cache.json"
    results_dir = project_root / "experiments" / "results"
    
    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = get_logger()
    
    print("="*60)
    print("VerifAI LLM Comparison Experiment")
    print("="*60)
    
    # Load configurations
    try:
        model_configs = load_model_config(config_path)
        print(f"\nLoaded {len(model_configs)} model configurations")
    except Exception as e:
        print(f"Error loading model configuration: {e}")
        sys.exit(1)
    
    # Load test messages
    try:
        test_messages = load_test_messages(test_csv_path, limit=40)
        print(f"Loaded {len(test_messages)} test messages")
    except Exception as e:
        print(f"Error loading test messages: {e}")
        sys.exit(1)
    
    # Load search cache
    try:
        search_cache = load_search_cache(cache_file)
        print(f"✓ Search cache loaded with {len(search_cache)} entries")
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"⚠ Warning: Failed to load search cache: {e}")
        print("  Continuing without cache (will run searches for each model)...")
        search_cache = None
    
    # Run experiments for each model
    for model_config in model_configs:
        model_name = model_config["name"]
        results_file = results_dir / f"{model_name}_results.json"
        
        try:
            # Run experiment
            experiment_results = run_experiment_for_model(
                model_config,
                test_messages,
                logger,
                search_cache=search_cache
            )
            
            # Save results
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(experiment_results, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ Results saved to: {results_file}")
            print(f"  Processed {experiment_results['processed_messages']}/{experiment_results['total_messages']} messages")
            
        except Exception as e:
            print(f"\n✗ Failed to run experiment for {model_name}: {str(e)}")
            print(f"  Continuing with next model...")
            continue
    
    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()

