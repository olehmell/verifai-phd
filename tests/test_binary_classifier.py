"""
Tests for binary manipulation classifier.
Tests both lapa-llm and modern-bert models with configurable thresholds.
"""
import pytest
import sys
import os
from pathlib import Path

# Add project root to path
test_dir = Path(__file__).parent.absolute()
project_root = test_dir.parent
sys.path.insert(0, str(project_root))


class TestBinaryClassifier:
    """Test binary manipulation classifier functionality"""
    
    def test_binary_classifier_can_be_imported(self):
        """Test that binary classifier can be imported"""
        from verifai.nodes.manipulation_classifier_binary import manipulation_classifier_binary
        assert manipulation_classifier_binary is not None
    
    def test_binary_classifier_handles_empty_content(self):
        """Test that binary classifier handles empty content"""
        pytest.importorskip("transformers")
        
        from verifai.nodes.manipulation_classifier_binary import manipulation_classifier_binary
        
        state = {
            "content": "",
            "content_id": "test-empty"
        }
        
        result = manipulation_classifier_binary(state)
        
        assert "is_manipulation" in result
        assert "manipulation_score" in result
        assert "manipulation_probability" in result
        assert "manipulation_techniques" in result
        assert result["is_manipulation"] is False
        assert result["manipulation_score"] == 0.0
        assert result["manipulation_probability"] == 0.0
        assert result["manipulation_techniques"] == []
    
    def test_binary_classifier_with_lapa_model(self):
        """Test binary classifier with lapa-llm model"""
        pytest.importorskip("transformers")
        pytest.importorskip("torch")
        
        from verifai.nodes.manipulation_classifier_binary import manipulation_classifier_binary
        
        # Use a neutral test text
        state = {
            "content": "Сьогодні була гарна погода",
            "content_id": "test-lapa",
            "_manipulation_model": "lapa-llm",
            "_manipulation_threshold": 0.5
        }
        
        result = manipulation_classifier_binary(state)
        
        assert "is_manipulation" in result
        assert "manipulation_score" in result
        assert isinstance(result["is_manipulation"], bool)
        assert 0.0 <= result["manipulation_score"] <= 1.0
        assert "manipulation_probability" in result
        assert result["manipulation_probability"] == result["manipulation_score"]
    
    def test_binary_classifier_with_modern_bert_model(self):
        """Test binary classifier with modern-bert model"""
        pytest.importorskip("transformers")
        pytest.importorskip("torch")
        
        from verifai.nodes.manipulation_classifier_binary import manipulation_classifier_binary
        
        # Use a test text with potential manipulation
        state = {
            "content": "УВАГА! Страшна новина! Всі знають про цю небезпеку!",
            "content_id": "test-modern-bert",
            "_manipulation_model": "modern-bert",
            "_manipulation_threshold": 0.15
        }
        
        result = manipulation_classifier_binary(state)
        
        assert "is_manipulation" in result
        assert "manipulation_score" in result
        assert "manipulation_techniques" in result
        assert isinstance(result["is_manipulation"], bool)
        assert 0.0 <= result["manipulation_score"] <= 1.0
        assert isinstance(result["manipulation_techniques"], list)
        
        # If manipulation detected, techniques should be populated
        if result["is_manipulation"]:
            assert len(result["manipulation_techniques"]) > 0
    
    def test_binary_classifier_respects_threshold(self):
        """Test that binary classifier respects custom threshold"""
        pytest.importorskip("transformers")
        pytest.importorskip("torch")
        
        from verifai.nodes.manipulation_classifier_binary import manipulation_classifier_binary
        
        content = "Це тестовий текст"
        
        # Test with low threshold
        state_low = {
            "content": content,
            "content_id": "test-threshold-low",
            "_manipulation_model": "lapa-llm",
            "_manipulation_threshold": 0.1
        }
        
        result_low = manipulation_classifier_binary(state_low)
        
        # Test with high threshold
        state_high = {
            "content": content,
            "content_id": "test-threshold-high",
            "_manipulation_model": "lapa-llm",
            "_manipulation_threshold": 0.9
        }
        
        result_high = manipulation_classifier_binary(state_high)
        
        # Both should have same score but potentially different binary result
        assert "manipulation_score" in result_low
        assert "manipulation_score" in result_high
        
        # High threshold is less likely to classify as manipulation
        if result_low["is_manipulation"] and not result_high["is_manipulation"]:
            assert result_low["manipulation_score"] < 0.9
    
    def test_binary_classifier_error_handling(self):
        """Test that binary classifier handles errors gracefully"""
        from verifai.nodes.manipulation_classifier_binary import manipulation_classifier_binary
        
        # Invalid model key
        state = {
            "content": "Test content",
            "content_id": "test-error",
            "_manipulation_model": "invalid-model-key"
        }
        
        result = manipulation_classifier_binary(state)
        
        # Should return safe defaults on error
        assert result["is_manipulation"] is False
        assert result["manipulation_score"] == 0.0
        assert result["manipulation_techniques"] == []


class TestGraphWithBinaryClassifier:
    """Test graph integration with binary classifier"""
    
    def test_graph_can_be_created_with_binary_classifier(self):
        """Test that graph can be created with binary classifier"""
        pytest.importorskip("langgraph")
        
        from verifai import create_graph
        
        graph = create_graph(use_binary_classifier=True)
        assert graph is not None
    
    def test_graph_can_be_created_with_prompt_classifier(self):
        """Test that graph can still use prompt-based classifier"""
        pytest.importorskip("langgraph")
        
        from verifai import create_graph
        
        graph = create_graph(use_binary_classifier=False)
        assert graph is not None
    
    def test_graph_defaults_to_binary_classifier(self):
        """Test that graph defaults to binary classifier"""
        pytest.importorskip("langgraph")
        
        from verifai import create_graph
        
        # Default should be binary classifier
        graph = create_graph()
        assert graph is not None


class TestNarrativeExtractorWithBinaryClassifier:
    """Test narrative extractor with binary classifier outputs"""
    
    def test_narrative_extractor_handles_no_techniques(self):
        """Test that narrative extractor handles empty techniques list"""
        pytest.importorskip("google.genai")
        
        # Skip if API key is not set
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        
        from verifai.nodes.narrative_extractor import narrative_extractor
        
        state = {
            "content": "Це тестовий контент для аналізу",
            "content_id": "test-narrative",
            "manipulation_probability": 0.7,
            "manipulation_techniques": [],  # Empty techniques from binary classifier
            "manipulation_score": 0.7
        }
        
        result = narrative_extractor(state)
        
        assert "narrative" in result
        assert isinstance(result["narrative"], str)
        assert len(result["narrative"]) > 0
    
    def test_narrative_extractor_handles_with_techniques(self):
        """Test that narrative extractor handles populated techniques list"""
        pytest.importorskip("google.genai")
        
        # Skip if API key is not set
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        
        from verifai.nodes.narrative_extractor import narrative_extractor
        
        state = {
            "content": "Це тестовий контент з маніпуляцією",
            "content_id": "test-narrative-with-tech",
            "manipulation_probability": 0.8,
            "manipulation_techniques": ["fear_appeals", "emotional_manipulation"],
            "manipulation_score": 0.8
        }
        
        result = narrative_extractor(state)
        
        assert "narrative" in result
        assert isinstance(result["narrative"], str)
        assert len(result["narrative"]) > 0

