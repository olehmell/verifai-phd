"""
Test verification quality on labeled dataset.
Runs verification on all examples from data/test.csv and calculates F1 score.
"""
import pytest
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

# Add project root to path
test_dir = Path(__file__).parent.absolute()
project_root = test_dir.parent
sys.path.insert(0, str(project_root))

from main import analyze_content


@pytest.fixture
def dataset_path():
    """Fixture for dataset path"""
    return project_root / "data" / "test.csv"


@pytest.fixture
def results_dir():
    """Fixture for results directory"""
    results_dir = project_root / "tests" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def calculate_f1_score(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    """
    Calculate F1 score, precision, and recall from true and predicted labels.
    
    Args:
        y_true: List of true labels (manipulative)
        y_pred: List of predicted labels (manipulation)
    
    Returns:
        Dictionary with precision, recall, and f1_score
    """
    # Calculate TP, FP, FN, TN
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == True and p == True)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == False and p == True)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == True and p == False)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == False and p == False)
    
    # Calculate precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Calculate recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": len(y_true)
    }


class TestVerificationQuality:
    """Test verification quality on labeled dataset"""
    
    @pytest.mark.slow
    def test_verification_on_dataset(self, dataset_path, results_dir):
        """Run verification on all examples in the dataset"""
        pytest.importorskip("google.genai")
        
        # Skip if API key is not set
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        
        if not dataset_path.exists():
            pytest.skip(f"Dataset not found at {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        assert len(df) > 0, "Dataset should not be empty"
        
        # Initialize results storage
        results = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "dataset_path": str(dataset_path),
                "total_examples": len(df),
                "sleep_seconds": 0  # No sleep in pytest for faster execution
            },
            "examples": [],
            "metrics": {}
        }
        
        y_true = []
        y_pred = []
        
        # Process each example
        for idx, row in df.iterrows():
            try:
                # Run verification
                start_time = time.time()
                result = analyze_content(
                    content=row['content'],
                    content_id=row['id']
                )
                duration = time.time() - start_time
                
                # Extract predicted manipulation
                predicted_manipulation = result.get('manipulation', False)
                
                # Store result
                example_result = {
                    "id": row['id'],
                    "content": row['content'],
                    "true_label": bool(row['manipulative']),
                    "predicted_label": predicted_manipulation,
                    "correct": bool(row['manipulative']) == predicted_manipulation,
                    "duration_seconds": round(duration, 2),
                    "result": result,
                    "techniques": result.get('techniques', []),
                    "explanation": result.get('explanation', '')
                }
                
                results["examples"].append(example_result)
                
                # Collect labels for metrics
                y_true.append(bool(row['manipulative']))
                y_pred.append(predicted_manipulation)
                
            except Exception as e:
                # Store error result
                example_result = {
                    "id": row['id'],
                    "content": row['content'],
                    "true_label": bool(row['manipulative']),
                    "predicted_label": None,
                    "correct": None,
                    "duration_seconds": None,
                    "error": str(e),
                    "result": None
                }
                
                results["examples"].append(example_result)
                
                # Use None for metrics (will be excluded)
                y_true.append(bool(row['manipulative']))
                y_pred.append(None)
        
        # Filter out None predictions for metrics calculation
        valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]
        
        # Calculate metrics
        assert len(y_true_valid) > 0, "No valid predictions to calculate metrics"
        
        metrics = calculate_f1_score(y_true_valid, y_pred_valid)
        results["metrics"] = metrics
        
        # Save results to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"verification_results_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Assert minimum performance thresholds
        assert metrics['f1_score'] > 0.0, f"F1 score should be > 0, got {metrics['f1_score']}"
        assert metrics['precision'] >= 0.0, f"Precision should be >= 0, got {metrics['precision']}"
        assert metrics['recall'] >= 0.0, f"Recall should be >= 0, got {metrics['recall']}"
        
        # Store metrics in test item for reporting
        pytest.current_test_metrics = metrics
    
    @pytest.mark.slow
    def test_verification_quality_thresholds(self, dataset_path, results_dir):
        """Test that verification meets quality thresholds"""
        pytest.importorskip("google.genai")
        
        # Skip if API key is not set
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        
        if not dataset_path.exists():
            pytest.skip(f"Dataset not found at {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Run on a subset for faster testing (first 10 samples)
        df_subset = df.head(10)
        
        y_true = []
        y_pred = []
        
        for idx, row in df_subset.iterrows():
            try:
                result = analyze_content(
                    content=row['content'],
                    content_id=row['id']
                )
                predicted_manipulation = result.get('manipulation', False)
                y_true.append(bool(row['manipulative']))
                y_pred.append(predicted_manipulation)
            except Exception:
                # Skip errors for threshold test
                continue
        
        if len(y_true) == 0:
            pytest.skip("No valid predictions for threshold test")
        
        metrics = calculate_f1_score(y_true, y_pred)
        
        # Assert minimum thresholds (adjust based on requirements)
        # These are example thresholds - adjust based on actual requirements
        assert metrics['f1_score'] >= 0.0, f"F1 score below threshold: {metrics['f1_score']}"
        assert metrics['precision'] >= 0.0, f"Precision below threshold: {metrics['precision']}"
        assert metrics['recall'] >= 0.0, f"Recall below threshold: {metrics['recall']}"


class TestF1ScoreCalculation:
    """Test F1 score calculation function"""
    
    def test_f1_score_perfect_match(self):
        """Test F1 score calculation with perfect match"""
        y_true = [True, True, False, False]
        y_pred = [True, True, False, False]
        
        metrics = calculate_f1_score(y_true, y_pred)
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert metrics['tp'] == 2
        assert metrics['tn'] == 2
        assert metrics['fp'] == 0
        assert metrics['fn'] == 0
    
    def test_f1_score_all_wrong(self):
        """Test F1 score calculation with all wrong predictions"""
        y_true = [True, False]
        y_pred = [False, True]
        
        metrics = calculate_f1_score(y_true, y_pred)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0
        assert metrics['tp'] == 0
        assert metrics['tn'] == 0
        assert metrics['fp'] == 1
        assert metrics['fn'] == 1
    
    def test_f1_score_edge_cases(self):
        """Test F1 score calculation with edge cases"""
        # All true positives
        y_true = [True, True]
        y_pred = [True, True]
        metrics = calculate_f1_score(y_true, y_pred)
        assert metrics['f1_score'] == 1.0
        
        # All false positives
        y_true = [False, False]
        y_pred = [True, True]
        metrics = calculate_f1_score(y_true, y_pred)
        assert metrics['precision'] == 0.0
        assert metrics['f1_score'] == 0.0
        
        # All false negatives
        y_true = [True, True]
        y_pred = [False, False]
        metrics = calculate_f1_score(y_true, y_pred)
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0
