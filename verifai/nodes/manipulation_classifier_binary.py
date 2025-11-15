from transformers import pipeline
from typing import Dict, Any, List, Optional
import torch
import time
from verifai.utils.logging import get_logger

# Model configurations
MODELS = {
    "lapa-llm": {
        "name": "lapa-llm/manipulative-score-model",
        "type": "binary",
        "default_threshold": 0.5
    },
    "modern-bert": {
        "name": "olehmell/ukr-manipulation-detector-modern-bert-2048-checkpoint-2870",
        "type": "multilabel",
        "default_threshold": 0.15
    }
}

# Label mapping for multilabel model
LABEL_MAPPING = {
    "LABEL_0": "emotional_manipulation",
    "LABEL_1": "fear_appeals",
    "LABEL_2": "bandwagon_effect",
    "LABEL_3": "selective_truth",
    "LABEL_4": "cliche"
}

# Cache pipelines to avoid reloading
_pipelines = {}

def get_classifier_pipeline(model_key: str):
    """
    Get or create the classifier pipeline for the specified model.
    
    Args:
        model_key: Key for the model (e.g., "lapa-llm" or "modern-bert")
    
    Returns:
        Classifier pipeline
    """
    if model_key not in _pipelines:
        if model_key not in MODELS:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODELS.keys())}")
        
        model_config = MODELS[model_key]
        _pipelines[model_key] = pipeline(
            "text-classification",
            model=model_config["name"],
            device=0 if torch.cuda.is_available() else -1,
            trust_remote_code=True
        )
    
    return _pipelines[model_key]

def manipulation_classifier_binary(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Binary manipulation classifier with configurable model selection.
    
    Supports two models:
    - lapa-llm/manipulative-score-model: Binary classification
    - olehmell/ukr-manipulation-detector-modern-bert-2048-checkpoint-2870: Multilabel classification
    
    Args:
        state: Current graph state containing:
            - content: str - text to classify
            - _manipulation_model: Optional[str] - model to use (default: "lapa-llm")
            - _manipulation_threshold: Optional[float] - threshold for classification
    
    Returns:
        Updated state with:
            - is_manipulation: bool - binary classification result
            - manipulation_score: float - raw score from model
            - manipulation_techniques: List[str] - detected techniques (only for multilabel model)
            - manipulation_probability: float - same as manipulation_score (for backward compatibility)
    """
    logger = get_logger()
    start_time = time.time()
    
    content = state.get("content", "")
    content_id = state.get("content_id", "unknown")
    
    # Get configuration from state
    model_key = state.get("_manipulation_model", "lapa-llm")
    threshold = state.get("_manipulation_threshold")
    
    # Use model-specific default threshold if not provided
    if threshold is None:
        threshold = MODELS.get(model_key, {}).get("default_threshold", 0.5)
    
    if not content.strip():
        logger.log_classification(
            content_id=content_id,
            duration=time.time() - start_time,
            manipulation_probability=0.0,
            techniques=[],
            content_length=0
        )
        return {
            "is_manipulation": False,
            "manipulation_score": 0.0,
            "manipulation_probability": 0.0,
            "manipulation_techniques": []
        }
    
    try:
        # Get the classifier pipeline
        classifier = get_classifier_pipeline(model_key)
        model_config = MODELS[model_key]
        
        # Run classification
        results = classifier(content)
        
        # Initialize output variables
        manipulation_score = 0.0
        manipulation_techniques = []
        
        # Process results based on model type
        if model_config["type"] == "binary":
            # Binary classifier - single label with score
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
                manipulation_score = result.get("score", 0.0)
                
                # Some binary classifiers might return label like "manipulative" / "not_manipulative"
                # Adjust score based on label if needed
                label = result.get("label", "").lower()
                if "not" in label or "non" in label or label == "0":
                    manipulation_score = 1.0 - manipulation_score
        
        elif model_config["type"] == "multilabel":
            # Multilabel classifier - multiple labels with scores
            if isinstance(results, list) and len(results) > 0:
                # Check if results is list of dicts (multilabel format)
                if isinstance(results[0], dict):
                    for result in results:
                        score = result.get("score", 0.0)
                        if score > threshold:
                            raw_label = result.get("label", "unknown")
                            readable_label = LABEL_MAPPING.get(raw_label, raw_label)
                            manipulation_techniques.append(readable_label)
                        # Track maximum score across all labels
                        manipulation_score = max(manipulation_score, score)
                else:
                    # Single result case
                    result = results[0]
                    score = result.get("score", 0.0)
                    manipulation_score = score
                    if score > threshold:
                        raw_label = result.get("label", "unknown")
                        readable_label = LABEL_MAPPING.get(raw_label, raw_label)
                        manipulation_techniques.append(readable_label)
        
        # Determine binary classification
        is_manipulation = manipulation_score >= threshold
        
        duration = time.time() - start_time
        
        # Log metrics
        logger.log_classification(
            content_id=content_id,
            duration=duration,
            manipulation_probability=manipulation_score,
            techniques=manipulation_techniques,
            content_length=len(content)
        )
        
        return {
            "is_manipulation": is_manipulation,
            "manipulation_score": manipulation_score,
            "manipulation_probability": manipulation_score,  # For backward compatibility
            "manipulation_techniques": manipulation_techniques
        }
    
    except Exception as e:
        duration = time.time() - start_time
        logger.log_step(
            step_name="manipulation_classifier_binary",
            content_id=content_id,
            duration=duration,
            error=str(e),
            metrics={"content_length": len(content), "model": model_key}
        )
        
        # Return safe defaults on error
        return {
            "is_manipulation": False,
            "manipulation_score": 0.0,
            "manipulation_probability": 0.0,
            "manipulation_techniques": []
        }

