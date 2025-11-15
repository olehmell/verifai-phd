# Binary Manipulation Classifier Implementation

## Overview

This document describes the implementation of a binary manipulation classifier with configurable model selection for the VerifAI system.

## Implementation Summary

### 1. New Binary Classifier Node

Created `/Users/olehmell/projects/agents/verifai/verifai/nodes/manipulation_classifier_binary.py` with the following features:

- **Two Model Support:**
  - `lapa-llm/manipulative-score-model` - Binary classification model
  - `olehmell/ukr-manipulation-detector-modern-bert-2048-checkpoint-2870` - Multilabel classification model

- **Configurable Parameters (via state):**
  - `_manipulation_model: str` - Model selection ("lapa-llm" or "modern-bert")
  - `_manipulation_threshold: float` - Classification threshold (model-specific defaults)

- **Output Fields:**
  - `is_manipulation: bool` - Binary classification result
  - `manipulation_score: float` - Raw classification score
  - `manipulation_probability: float` - Same as score (backward compatibility)
  - `manipulation_techniques: List[str]` - Detected techniques (multilabel model only)

### 2. Updated State Definition

Extended `VerifaiState` in `/Users/olehmell/projects/agents/verifai/verifai/graph.py`:

```python
# Internal: Manipulation classifier configuration
_manipulation_model: Optional[str]
_manipulation_threshold: Optional[float]
_use_binary_classifier: Optional[bool]

# Manipulation Classifier outputs
is_manipulation: Optional[bool]
manipulation_score: Optional[float]
manipulation_probability: Optional[float]  # Backward compatibility
manipulation_techniques: Optional[List[str]]
```

### 3. Enhanced Narrative Extractor

Updated `/Users/olehmell/projects/agents/verifai/verifai/prompts/narrative_extractor.py`:

- Handles empty `manipulation_techniques` list (binary classifier output)
- Handles populated `manipulation_techniques` list (multilabel classifier output)
- Adjusts prompt context based on available information

### 4. Flexible Graph Creation

Modified `create_graph()` function to support both classifiers:

```python
# Use binary classifier (default)
graph = create_graph(use_binary_classifier=True)

# Use prompt-based classifier (legacy)
graph = create_graph(use_binary_classifier=False)
```

### 5. Comprehensive Testing

Created `/Users/olehmell/projects/agents/verifai/tests/test_binary_classifier.py` with:

- Binary classifier functionality tests
- Model-specific tests (lapa-llm and modern-bert)
- Threshold configuration tests
- Error handling tests
- Graph integration tests
- Narrative extractor compatibility tests

## Usage Examples

### Basic Usage (Default Binary Classifier)

```python
from verifai import create_graph

# Create graph with binary classifier (default)
graph = create_graph()

# Process content
result = graph.invoke({
    "content": "Your Ukrainian text here"
})

# Access binary classification result
is_manipulative = result["is_manipulation"]
score = result["manipulation_score"]
```

### Using Specific Model

```python
from verifai import create_graph

# Create graph
graph = create_graph(use_binary_classifier=True)

# Configure model via state
result = graph.invoke({
    "content": "Your Ukrainian text here",
    "_manipulation_model": "modern-bert",  # or "lapa-llm"
    "_manipulation_threshold": 0.15
})

# For multilabel model, techniques are also available
techniques = result.get("manipulation_techniques", [])
```

### Using Legacy Prompt-Based Classifier

```python
from verifai import create_graph

# Use prompt-based classifier
graph = create_graph(use_binary_classifier=False)

result = graph.invoke({
    "content": "Your Ukrainian text here"
})

# Access legacy outputs
probability = result["manipulation_probability"]
techniques = result["manipulation_techniques"]
```

## Model Details

### lapa-llm/manipulative-score-model
- Type: Binary classifier
- Default threshold: 0.5
- Output: Single manipulation score
- Techniques: Not provided

### olehmell/ukr-manipulation-detector-modern-bert-2048-checkpoint-2870
- Type: Multilabel classifier
- Default threshold: 0.15
- Output: Scores for 5 manipulation techniques
- Techniques:
  - `emotional_manipulation` - Emotional manipulation
  - `fear_appeals` - Fear appeals
  - `bandwagon_effect` - Bandwagon effect
  - `selective_truth` - Selective truth
  - `cliche` - Thought-terminating clichés

## Backward Compatibility

All changes maintain full backward compatibility:

- Original `manipulation_classifier.py` is preserved
- Legacy prompt-based approach available via `use_binary_classifier=False`
- State fields include `manipulation_probability` for backward compatibility
- Existing tests continue to pass
- Narrative extractor handles both output formats

## Test Results

All tests pass successfully:

```
tests/test_binary_classifier.py:
- 9 passed, 2 skipped (requires GEMINI_API_KEY)
- All model-specific tests passed
- Graph integration tests passed

tests/test_basic.py:
- 7 passed, 1 skipped
- Full backward compatibility confirmed
```

## Configuration Best Practices

1. **For Binary Detection Only:**
   ```python
   graph = create_graph(use_binary_classifier=True)
   result = graph.invoke({
       "content": text,
       "_manipulation_model": "lapa-llm",
       "_manipulation_threshold": 0.5
   })
   ```

2. **For Detailed Technique Analysis:**
   ```python
   graph = create_graph(use_binary_classifier=True)
   result = graph.invoke({
       "content": text,
       "_manipulation_model": "modern-bert",
       "_manipulation_threshold": 0.15
   })
   ```

3. **For Prompt-Based Analysis (Legacy):**
   ```python
   graph = create_graph(use_binary_classifier=False)
   result = graph.invoke({"content": text})
   ```

## Performance Considerations

- Models are cached globally to avoid reloading
- GPU support automatic if CUDA is available
- Binary classifier is generally faster than prompt-based
- Multilabel model provides more detailed analysis but slightly slower

## Future Enhancements

Possible improvements:
- Ensemble approach combining both models
- Dynamic threshold adjustment based on content type
- Additional model support
- Performance benchmarking utilities

