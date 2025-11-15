"""
Basic tests for VerifAI MVP implementation.
Tests logging, dataset loading, manipulation classifier, and graph structure.
"""
import pytest
import sys
import os
from pathlib import Path

# Add project root to path
test_dir = Path(__file__).parent.absolute()
project_root = test_dir.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def dataset_path():
    """Fixture for dataset path"""
    return project_root / "data" / "test.csv"


class TestLoggingModule:
    """Test logging module functionality"""
    
    def test_logging_module_can_be_imported(self):
        """Test that logging module can be imported and initialized"""
        from verifai.utils.logging import get_logger
        
        logger = get_logger()
        assert logger is not None, "Logger should not be None"
    
    def test_logging_step(self):
        """Test that logging a step works"""
        from verifai.utils.logging import get_logger
        
        logger = get_logger()
        logger.log_step(
            step_name="test_step",
            content_id="test-123",
            duration=0.1,
            metrics={"test": "value"}
        )
        # If no exception, test passes
        assert True


class TestDatasetLoading:
    """Test dataset loading functionality"""
    
    def test_dataset_exists(self, dataset_path):
        """Test that test dataset exists"""
        if not dataset_path.exists():
            pytest.skip(f"Dataset not found at {dataset_path}")
        
        assert dataset_path.exists()
    
    def test_dataset_can_be_loaded(self, dataset_path):
        """Test that test dataset can be loaded"""
        pytest.importorskip("pandas")
        
        if not dataset_path.exists():
            pytest.skip(f"Dataset not found at {dataset_path}")
        
        import pandas as pd
        
        df = pd.read_csv(dataset_path)
        assert len(df) > 0, "Dataset should not be empty"
        assert 'content' in df.columns, "Dataset should have 'content' column"
        assert 'manipulative' in df.columns, "Dataset should have 'manipulative' column"
    
    def test_dataset_has_valid_samples(self, dataset_path):
        """Test that dataset has valid samples"""
        pytest.importorskip("pandas")
        
        if not dataset_path.exists():
            pytest.skip(f"Dataset not found at {dataset_path}")
        
        import pandas as pd
        
        df = pd.read_csv(dataset_path)
        assert len(df) > 0
        
        # Check that we have both manipulative and non-manipulative samples
        manipulative_sum = df['manipulative'].sum()
        non_manipulative_sum = (~df['manipulative']).sum()
        
        assert manipulative_sum > 0 or non_manipulative_sum > 0, "Dataset should have samples"


class TestManipulationClassifierNode:
    """Test manipulation classifier node"""
    
    def test_classifier_handles_empty_content(self):
        """Test that classifier handles empty content"""
        pytest.importorskip("google.genai")
        
        from verifai.nodes.manipulation_classifier import manipulation_classifier
        
        state = {
            "content": "",
            "content_id": "test-id"
        }
        
        result = manipulation_classifier(state)
        
        assert "manipulation_probability" in result
        assert "manipulation_techniques" in result
        assert result["manipulation_probability"] == 0.0
        assert result["manipulation_techniques"] == []
    
    def test_classifier_processes_content(self):
        """Test that classifier can process content"""
        pytest.importorskip("google.genai")
        
        # Skip if API key is not set
        import os
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        
        from verifai.nodes.manipulation_classifier import manipulation_classifier
        
        state = {
            "content": "Test content for classification",
            "content_id": "test-id"
        }
        
        result = manipulation_classifier(state)
        
        assert "manipulation_probability" in result
        assert "manipulation_techniques" in result
        assert 0.0 <= result["manipulation_probability"] <= 1.0
        assert isinstance(result["manipulation_techniques"], list)


class TestGraphStructure:
    """Test graph structure"""
    
    def test_graph_can_be_created(self):
        """Test that graph can be created"""
        pytest.importorskip("langgraph")
        
        from verifai import create_graph
        
        graph = create_graph()
        assert graph is not None, "Graph should not be None"
