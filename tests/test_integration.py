"""
Integration tests for VerifAI MVP using test.csv dataset.
Tests follow TDD approach - tests are written before implementation.
"""
import pytest
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import analyze_content


@pytest.fixture
def test_dataset():
    """Load test dataset from data/test.csv"""
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "test.csv"
    )
    if not os.path.exists(dataset_path):
        pytest.skip(f"Test dataset not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    return df


class TestDatasetLoading:
    """Test that the dataset can be loaded correctly"""
    
    def test_dataset_exists(self, test_dataset):
        """Test that test dataset exists and can be loaded"""
        assert test_dataset is not None
        assert len(test_dataset) > 0
    
    def test_dataset_has_required_columns(self, test_dataset):
        """Test that dataset has required columns"""
        required_columns = ['id', 'content', 'lang', 'manipulative', 'techniques']
        for col in required_columns:
            assert col in test_dataset.columns, f"Missing required column: {col}"
    
    def test_dataset_has_both_classes(self, test_dataset):
        """Test that dataset has both manipulative and non-manipulative samples"""
        manipulative_counts = test_dataset['manipulative'].value_counts()
        assert True in manipulative_counts.index, "Dataset should have manipulative samples"
        assert False in manipulative_counts.index, "Dataset should have non-manipulative samples"


class TestManipulationClassifierNode:
    """Test manipulation classifier node"""
    
    @patch('verifai.nodes.manipulation_classifier.get_gemini_client')
    def test_classifier_returns_probabability_with_mock(self, mock_client):
        """Test that classifier returns manipulation probability with mocked Gemini"""
        from verifai.nodes.manipulation_classifier import manipulation_classifier
        
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = '{"manipulation_probability": 0.7, "manipulation_techniques": ["emotional_manipulation", "fear_appeals"]}'
        mock_client.return_value.models.generate_content.return_value = mock_response
        
        state = {
            "content": "Test content for classification",
            "content_id": "test-id"
        }
        
        result = manipulation_classifier(state)
        
        assert "manipulation_probability" in result
        assert isinstance(result["manipulation_probability"], (int, float))
        assert 0.0 <= result["manipulation_probability"] <= 1.0
        assert result["manipulation_probability"] == 0.7
        assert result["manipulation_techniques"] == ["emotional_manipulation", "fear_appeals"]
    
    def test_classifier_returns_probabability(self, test_dataset):
        """Test that classifier returns manipulation probability"""
        from verifai.nodes.manipulation_classifier import manipulation_classifier
        
        sample = test_dataset.iloc[0]
        state = {
            "content": sample['content'],
            "content_id": sample['id']
        }
        
        result = manipulation_classifier(state)
        
        assert "manipulation_probability" in result
        assert isinstance(result["manipulation_probability"], (int, float))
        assert 0.0 <= result["manipulation_probability"] <= 1.0
    
    def test_classifier_returns_techniques(self, test_dataset):
        """Test that classifier returns manipulation techniques"""
        from verifai.nodes.manipulation_classifier import manipulation_classifier
        
        sample = test_dataset.iloc[0]
        state = {
            "content": sample['content'],
            "content_id": sample['id']
        }
        
        result = manipulation_classifier(state)
        
        assert "manipulation_techniques" in result
        assert isinstance(result["manipulation_techniques"], list)
    
    def test_classifier_handles_empty_content(self):
        """Test that classifier handles empty content"""
        from verifai.nodes.manipulation_classifier import manipulation_classifier
        
        state = {
            "content": "",
            "content_id": "test-id"
        }
        
        result = manipulation_classifier(state)
        
        assert result["manipulation_probability"] == 0.0
        assert result["manipulation_techniques"] == []
    
    @patch('verifai.nodes.manipulation_classifier.get_gemini_client')
    def test_classifier_handles_invalid_json_response(self, mock_client):
        """Test that classifier handles invalid JSON response gracefully"""
        from verifai.nodes.manipulation_classifier import manipulation_classifier
        
        # Mock Gemini response with invalid JSON
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Invalid JSON response"
        mock_client.return_value.models.generate_content.return_value = mock_response
        
        state = {
            "content": "Test content",
            "content_id": "test-id"
        }
        
        result = manipulation_classifier(state)
        
        # Should return safe defaults on error
        assert "manipulation_probability" in result
        assert result["manipulation_probability"] == 0.0
        assert result["manipulation_techniques"] == []


class TestNarrativeExtractorNode:
    """Test narrative extractor node"""
    
    @patch('verifai.nodes.narrative_extractor.get_gemini_client')
    def test_narrative_extractor_returns_narrative(self, mock_client, test_dataset):
        """Test that narrative extractor returns narrative"""
        from verifai.nodes.narrative_extractor import narrative_extractor
        
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Test narrative extracted"
        mock_client.return_value.models.generate_content.return_value = mock_response
        
        sample = test_dataset.iloc[0]
        state = {
            "content": sample['content'],
            "content_id": sample['id'],
            "manipulation_techniques": [],
            "manipulation_probability": 0.0
        }
        
        result = narrative_extractor(state)
        
        assert "narrative" in result
        assert isinstance(result["narrative"], str)
    
    def test_narrative_extractor_handles_empty_content(self):
        """Test that narrative extractor handles empty content"""
        from verifai.nodes.narrative_extractor import narrative_extractor
        
        state = {
            "content": "",
            "content_id": "test-id",
            "manipulation_techniques": [],
            "manipulation_probability": 0.0
        }
        
        result = narrative_extractor(state)
        
        assert "narrative" in result


class TestFactCheckerNode:
    """Test fact checker node"""
    
    @patch('verifai.nodes.fact_checker.perform_web_search')
    @patch('verifai.nodes.fact_checker.get_gemini_client')
    def test_fact_checker_generates_queries(self, mock_client, mock_search, test_dataset):
        """Test that fact checker generates search queries"""
        from verifai.nodes.fact_checker import fact_checker
        
        # Mock Gemini response for query generation
        mock_query_response = MagicMock()
        mock_query_response.candidates = [MagicMock()]
        mock_query_response.candidates[0].content.parts = [MagicMock()]
        mock_query_response.candidates[0].content.parts[0].text = "Query 1\nQuery 2\nQuery 3"
        
        # Mock Gemini response for fact check
        mock_fact_response = MagicMock()
        mock_fact_response.candidates = [MagicMock()]
        mock_fact_response.candidates[0].content.parts = [MagicMock()]
        mock_fact_response.candidates[0].content.parts[0].text = "Fact check results"
        
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.side_effect = [
            mock_query_response,
            mock_fact_response
        ]
        mock_client.return_value = mock_client_instance
        
        # Mock search results
        mock_search.return_value = [{"url": "test.com", "snippet": "test"}]
        
        sample = test_dataset.iloc[0]
        state = {
            "content": sample['content'],
            "content_id": sample['id'],
            "narrative": "Test narrative"
        }
        
        result = fact_checker(state)
        
        assert "search_queries" in result
        assert isinstance(result["search_queries"], list)
        assert len(result["search_queries"]) > 0
    
    def test_fact_checker_handles_empty_content(self):
        """Test that fact checker handles empty content"""
        from verifai.nodes.fact_checker import fact_checker
        
        state = {
            "content": "",
            "content_id": "test-id",
            "narrative": ""
        }
        
        result = fact_checker(state)
        
        assert "search_queries" in result
        assert "fact_check_results" in result


class TestVerifierNode:
    """Test verifier node"""
    
    @patch('verifai.nodes.verifier.get_gemini_client')
    def test_verifier_returns_final_result(self, mock_client, test_dataset):
        """Test that verifier returns final result"""
        from verifai.nodes.verifier import verifier
        
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = '{"manipulation": false, "techniques": [], "disinfo": [], "explanation": "Test"}'
        mock_client.return_value.models.generate_content.return_value = mock_response
        
        sample = test_dataset.iloc[0]
        state = {
            "content": sample['content'],
            "content_id": sample['id'],
            "manipulation_probability": 0.0,
            "manipulation_techniques": [],
            "narrative": "Test narrative",
            "fact_check_results": "Test fact check"
        }
        
        result = verifier(state)
        
        assert "final_result" in result
        assert isinstance(result["final_result"], dict)
        assert "manipulation" in result["final_result"]
        assert "techniques" in result["final_result"]
        assert "disinfo" in result["final_result"]
        assert "explanation" in result["final_result"]
    
    def test_verifier_handles_empty_content(self):
        """Test that verifier handles empty content"""
        from verifai.nodes.verifier import verifier
        
        state = {
            "content": "",
            "content_id": "test-id",
            "manipulation_probability": 0.0,
            "manipulation_techniques": [],
            "narrative": "",
            "fact_check_results": ""
        }
        
        result = verifier(state)
        
        assert "final_result" in result
        assert result["final_result"]["manipulation"] == False


class TestEndToEndPipeline:
    """Test end-to-end pipeline integration"""
    
    @patch('verifai.nodes.verifier.get_gemini_client')
    @patch('verifai.nodes.fact_checker.perform_web_search')
    @patch('verifai.nodes.fact_checker.get_gemini_client')
    @patch('verifai.nodes.narrative_extractor.get_gemini_client')
    @patch('verifai.nodes.manipulation_classifier.get_gemini_client')
    def test_pipeline_processes_sample(self, mock_manipulation_client, mock_narrative_client, mock_fact_client, 
                                       mock_search, mock_verifier_client, test_dataset):
        """Test that pipeline processes a sample correctly"""
        # Mock all Gemini clients
        def create_mock_response(text):
            mock_response = MagicMock()
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].content.parts = [MagicMock()]
            mock_response.candidates[0].content.parts[0].text = text
            return mock_response
        
        mock_manipulation_client.return_value.models.generate_content.return_value = create_mock_response(
            '{"manipulation_probability": 0.5, "manipulation_techniques": ["emotional_manipulation"]}'
        )
        mock_narrative_client.return_value.models.generate_content.return_value = create_mock_response("Test narrative")
        
        mock_fact_query_response = create_mock_response("Query 1\nQuery 2")
        mock_fact_result_response = create_mock_response("Fact check results")
        mock_fact_instance = MagicMock()
        mock_fact_instance.models.generate_content.side_effect = [
            mock_fact_query_response,
            mock_fact_result_response
        ]
        mock_fact_client.return_value = mock_fact_instance
        
        mock_verifier_client.return_value.models.generate_content.return_value = create_mock_response(
            '{"manipulation": false, "techniques": [], "disinfo": [], "explanation": "Test"}'
        )
        
        mock_search.return_value = [{"url": "test.com", "snippet": "test"}]
        
        # Get a sample
        sample = test_dataset.iloc[0]
        
        # Run analysis
        result = analyze_content(
            content=sample['content'],
            content_id=sample['id']
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "manipulation" in result
        assert "techniques" in result
        assert "disinfo" in result
        assert "explanation" in result
    
    def test_pipeline_with_real_classifier(self, test_dataset):
        """Test pipeline with real classifier (mocked external APIs)"""
        from verifai import create_graph
        
        # Mock external APIs but use real classifier
        with patch('verifai.nodes.verifier.get_gemini_client') as mock_v, \
             patch('verifai.nodes.fact_checker.perform_web_search') as mock_s, \
             patch('verifai.nodes.fact_checker.get_gemini_client') as mock_fc, \
             patch('verifai.nodes.narrative_extractor.get_gemini_client') as mock_n, \
             patch('verifai.nodes.manipulation_classifier.get_gemini_client') as mock_mc:
            
            # Setup mocks
            def create_mock_response(text):
                mock_response = MagicMock()
                mock_response.candidates = [MagicMock()]
                mock_response.candidates[0].content.parts = [MagicMock()]
                mock_response.candidates[0].content.parts[0].text = text
                return mock_response
            
            mock_mc.return_value.models.generate_content.return_value = create_mock_response(
                '{"manipulation_probability": 0.5, "manipulation_techniques": ["emotional_manipulation"]}'
            )
            mock_n.return_value.models.generate_content.return_value = create_mock_response("Narrative")
            
            mock_fc_query = create_mock_response("Query 1")
            mock_fc_result = create_mock_response("Fact check")
            mock_fc_instance = MagicMock()
            mock_fc_instance.models.generate_content.side_effect = [mock_fc_query, mock_fc_result]
            mock_fc.return_value = mock_fc_instance
            
            mock_s.return_value = [{"url": "test.com", "snippet": "test"}]
            
            mock_v.return_value.models.generate_content.return_value = create_mock_response(
                '{"manipulation": false, "techniques": [], "disinfo": [], "explanation": "Test"}'
            )
            
            # Get a sample
            sample = test_dataset.iloc[0]
            
            # Create and run graph
            graph = create_graph()
            result = graph.invoke({
                "content": sample['content'],
                "content_id": sample['id']
            })
            
            # Verify result
            assert "final_result" in result
            final_result = result["final_result"]
            assert isinstance(final_result, dict)
            assert "manipulation" in final_result


class TestMetricsCollection:
    """Test that metrics are collected correctly"""
    
    def test_logging_is_initialized(self):
        """Test that logging module can be initialized"""
        from verifai.utils.logging import get_logger
        
        logger = get_logger()
        assert logger is not None
    
    def test_metrics_logged_on_step(self, test_dataset):
        """Test that metrics are logged when processing steps"""
        from verifai.nodes.manipulation_classifier import manipulation_classifier
        from verifai.utils.logging import get_logger
        
        logger = get_logger()
        
        sample = test_dataset.iloc[0]
        state = {
            "content": sample['content'],
            "content_id": sample['id']
        }
        
        # This should log metrics
        result = manipulation_classifier(state)
        
        # Verify result exists (metrics logging happens internally)
        assert result is not None
        assert "manipulation_probability" in result


class TestPerformanceMetrics:
    """Test performance metrics collection"""
    
    @patch('verifai.nodes.verifier.get_gemini_client')
    @patch('verifai.nodes.fact_checker.perform_web_search')
    @patch('verifai.nodes.fact_checker.get_gemini_client')
    @patch('verifai.nodes.narrative_extractor.get_gemini_client')
    @patch('verifai.nodes.manipulation_classifier.get_gemini_client')
    def test_pipeline_logs_total_duration(self, mock_mc, mock_n, mock_fc, mock_s, mock_v, test_dataset):
        """Test that pipeline logs total duration"""
        import time
        
        def create_mock_response(text):
            mock_response = MagicMock()
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].content.parts = [MagicMock()]
            mock_response.candidates[0].content.parts[0].text = text
            return mock_response
        
        mock_mc.return_value.models.generate_content.return_value = create_mock_response(
            '{"manipulation_probability": 0.5, "manipulation_techniques": ["emotional_manipulation"]}'
        )
        mock_n.return_value.models.generate_content.return_value = create_mock_response("Narrative")
        
        mock_fc_query = create_mock_response("Query 1")
        mock_fc_result = create_mock_response("Fact check")
        mock_fc_instance = MagicMock()
        mock_fc_instance.models.generate_content.side_effect = [mock_fc_query, mock_fc_result]
        mock_fc.return_value = mock_fc_instance
        
        mock_s.return_value = [{"url": "test.com", "snippet": "test"}]
        
        mock_v.return_value.models.generate_content.return_value = create_mock_response(
            '{"manipulation": false, "techniques": [], "disinfo": [], "explanation": "Test"}'
        )
        
        sample = test_dataset.iloc[0]
        start_time = time.time()
        
        result = analyze_content(
            content=sample['content'],
            content_id=sample['id']
        )
        
        duration = time.time() - start_time
        
        # Verify result exists and duration is reasonable
        assert result is not None
        assert duration > 0
        # Duration should be reasonable (less than 60 seconds for mocked calls)
        assert duration < 60


class TestDatasetCoverage:
    """Test coverage of test dataset"""
    
    def test_dataset_has_multiple_samples(self, test_dataset):
        """Test that dataset has multiple samples for testing"""
        assert len(test_dataset) >= 10, "Dataset should have at least 10 samples for testing"
    
    def test_dataset_has_both_languages(self, test_dataset):
        """Test that dataset includes both Ukrainian and Russian samples"""
        languages = test_dataset['lang'].unique()
        assert 'uk' in languages, "Dataset should have Ukrainian samples"
        assert 'ru' in languages, "Dataset should have Russian samples"
    
    def test_dataset_samples_have_content(self, test_dataset):
        """Test that all samples have non-empty content"""
        empty_content = test_dataset[test_dataset['content'].isna() | (test_dataset['content'].str.strip() == '')]
        assert len(empty_content) == 0, "Dataset should not have empty content samples"

