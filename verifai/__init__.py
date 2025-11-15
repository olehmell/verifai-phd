# Lazy import to avoid dependency issues
def create_graph(llm_client=None, use_binary_classifier=True):
    """Create and return the VerifAI graph
    
    Args:
        llm_client: Optional LLM client to use for all nodes. If None, nodes will use Gemini client.
        use_binary_classifier: Whether to use binary classifier (True) or prompt-based (False). Default: True.
    
    Returns:
        Compiled StateGraph instance
    """
    from .graph import create_graph as _create_graph
    return _create_graph(llm_client=llm_client, use_binary_classifier=use_binary_classifier)

__all__ = ["create_graph"]