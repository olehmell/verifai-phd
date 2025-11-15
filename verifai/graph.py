from typing import TypedDict, List, Optional, Any
from langgraph.graph import StateGraph, START, END
import time
import uuid
from verifai.utils.logging import get_logger
from verifai.utils.llm_client import LLMClient

class VerifaiState(TypedDict):
    """State for the VerifAI multi-agent system"""
    # Input
    content: str
    content_id: Optional[str]  # ID for tracking metrics

    # Internal: LLM client for experiment mode
    _llm_client: Optional[LLMClient]  # Internal field for LLM client injection
    
    # Internal: Search cache for experiment mode
    _use_search_cache: Optional[bool]  # Flag to use cached search results
    _cached_search_queries: Optional[List[str]]  # Cached search queries
    _cached_search_results: Optional[List[Any]]  # Cached search results

    # Internal: Manipulation classifier configuration
    _manipulation_model: Optional[str]  # Model to use ("lapa-llm" or "modern-bert")
    _manipulation_threshold: Optional[float]  # Threshold for binary classification
    _use_binary_classifier: Optional[bool]  # Whether to use binary classifier (default: True)

    # Manipulation Classifier outputs
    manipulation_probability: Optional[float]  # For backward compatibility
    manipulation_techniques: Optional[List[str]]
    is_manipulation: Optional[bool]  # Binary classification result
    manipulation_score: Optional[float]  # Raw score from classifier

    # Narrative Extractor outputs
    narrative: Optional[str]

    # Fact-checker outputs
    search_queries: Optional[List[str]]
    search_results: Optional[List[str]]
    fact_check_results: Optional[str]

    # Final Verifier output
    final_result: Optional[dict]


def create_graph(
    llm_client: Optional[LLMClient] = None,
    use_binary_classifier: bool = True
) -> StateGraph:
    """
    Create the VerifAI computational graph
    
    Args:
        llm_client: Optional LLM client to use for all nodes. If None, nodes will use Gemini client.
        use_binary_classifier: Whether to use binary classifier (True) or prompt-based (False). Default: True.
    
    Returns:
        Compiled StateGraph instance
    """

    # Initialize the graph with our state
    graph = StateGraph(VerifaiState)

    # Import nodes based on classifier choice
    if use_binary_classifier:
        from .nodes.manipulation_classifier_binary import manipulation_classifier_binary as manipulation_classifier
    else:
        from .nodes.manipulation_classifier import manipulation_classifier
    
    from .nodes.narrative_extractor import narrative_extractor
    from .nodes.fact_checker import fact_checker
    from .nodes.verifier import verifier

    # Add nodes to the graph
    graph.add_node("manipulation_classifier", manipulation_classifier)
    graph.add_node("narrative_extractor", narrative_extractor)
    graph.add_node("fact_checker", fact_checker)
    graph.add_node("verifier", verifier)

    # Define the flow according to the PRD diagram
    graph.add_edge(START, "manipulation_classifier")
    graph.add_edge(START, "fact_checker")
    graph.add_edge("manipulation_classifier", "narrative_extractor")
    graph.add_edge("narrative_extractor", "verifier")
    graph.add_edge("manipulation_classifier", "verifier")
    graph.add_edge("fact_checker", "verifier")
    graph.add_edge("verifier", END)

    compiled_graph = graph.compile()
    
    # Wrap the invoke method to add logging
    original_invoke = compiled_graph.invoke
    
    def invoke_with_logging(state_or_input: dict, config: Optional[dict] = None):
        """Invoke graph with logging"""
        logger = get_logger()
        start_time = time.time()
        
        # Ensure content_id exists
        if "content_id" not in state_or_input or not state_or_input.get("content_id"):
            state_or_input["content_id"] = str(uuid.uuid4())
        
        # Inject LLM client if provided
        if llm_client is not None:
            state_or_input["_llm_client"] = llm_client
        
        content_id = state_or_input["content_id"]
        
        try:
            # Run the graph
            result = original_invoke(state_or_input, config)
            
            # Log pipeline completion
            total_duration = time.time() - start_time
            final_result = result.get("final_result", {})
            
            logger.log_pipeline_complete(
                content_id=content_id,
                total_duration=total_duration,
                final_result=final_result
            )
            
            return result
            
        except Exception as e:
            total_duration = time.time() - start_time
            logger.log_step(
                step_name="pipeline_complete",
                content_id=content_id,
                duration=total_duration,
                error=str(e)
            )
            raise
    
    compiled_graph.invoke = invoke_with_logging
    return compiled_graph