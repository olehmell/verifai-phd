"""Configuration for VerifAI models"""

# Gemini model configurations
GEMINI_MODELS = {
    "manipulation_classifier": "gemini-2.5-flash",
    "narrative_extractor": "gemini-2.5-flash",
    "fact_checker": "gemini-2.5-flash",
    "verifier": "gemini-2.5-flash",
}


def get_gemini_model(node_name: str) -> str:
    """
    Get the Gemini model name for a specific node.
    
    Args:
        node_name: Name of the node (e.g., 'fact_checker', 'verifier')
    
    Returns:
        Model name string (e.g., 'gemini-2.5-flash')
    
    Raises:
        ValueError: If node_name is not found in configuration
    """
    if node_name not in GEMINI_MODELS:
        raise ValueError(
            f"Unknown node name: {node_name}. "
            f"Available nodes: {list(GEMINI_MODELS.keys())}"
        )
    return GEMINI_MODELS[node_name]

