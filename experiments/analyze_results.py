#!/usr/bin/env python3
"""
Script to analyze experiment results by comparing with test dataset.
Only checks manipulative True/False classification.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict


def load_test_dataset(test_path: str) -> pd.DataFrame:
    """Load test dataset from CSV."""
    df = pd.read_csv(test_path)
    # Convert manipulative column to boolean if it's string
    if df['manipulative'].dtype == 'object':
        df['manipulative'] = df['manipulative'].map({'True': True, 'False': False, True: True, False: False})
    return df


def load_experiment_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all experiment result JSON files."""
    results = {}
    
    for result_file in results_dir.glob("*_results.json"):
        model_name = result_file.stem.replace("_results", "")
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results[model_name] = data
    
    return results


def calculate_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Calculate classification metrics."""
    # Calculate confusion matrix manually
    tp = ((y_true == True) & (y_pred == True)).sum()
    tn = ((y_true == False) & (y_pred == False)).sum()
    fp = ((y_true == False) & (y_pred == True)).sum()
    fn = ((y_true == True) & (y_pred == False)).sum()
    
    # Calculate metrics
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
    }


def analyze_model_results(test_df: pd.DataFrame, model_results: Dict, model_name: str) -> Dict:
    """Analyze results for a single model - only checks manipulative True/False."""
    # Create a mapping from message_id to ground truth
    test_dict = test_df.set_index('id').to_dict('index')
    
    # Prepare predictions
    predictions = []
    
    for result in model_results.get('results', []):
        msg_id = result['message_id']
        if msg_id not in test_dict:
            continue
        
        pred_manipulation = result['final_result']['manipulation']
        true_manipulation = test_dict[msg_id]['manipulative']
        
        predictions.append({
            'message_id': msg_id,
            'predicted_manipulation': pred_manipulation,
            'true_manipulation': true_manipulation,
        })
    
    pred_df = pd.DataFrame(predictions)
    
    if len(pred_df) == 0:
        return {'error': 'No matching messages found'}
    
    # Calculate manipulation detection metrics
    manipulation_metrics = calculate_classification_metrics(
        pred_df['true_manipulation'],
        pred_df['predicted_manipulation']
    )
    
    # Error analysis
    false_positives = pred_df[
        (~pred_df['true_manipulation']) & (pred_df['predicted_manipulation'])
    ]
    false_negatives = pred_df[
        (pred_df['true_manipulation']) & (~pred_df['predicted_manipulation'])
    ]
    
    return {
        'model_name': model_name,
        'total_messages': len(pred_df),
        'manipulation_metrics': manipulation_metrics,
        'false_positives_count': len(false_positives),
        'false_negatives_count': len(false_negatives),
        'predictions': pred_df.to_dict('records'),
    }


def print_analysis_report(analysis_results: Dict[str, Dict]):
    """Print a formatted analysis report."""
    print("=" * 80)
    print("EXPERIMENT RESULTS ANALYSIS")
    print("=" * 80)
    print()
    
    for model_name, results in analysis_results.items():
        if 'error' in results:
            print(f"Model: {model_name}")
            print(f"Error: {results['error']}")
            print()
            continue
        
        print(f"Model: {results['model_name']}")
        print(f"Total Messages Analyzed: {results['total_messages']}")
        print()
        
        # Manipulation detection metrics
        mm = results['manipulation_metrics']
        print("Manipulation Detection Metrics:")
        print(f"  Accuracy:  {mm['accuracy']:.4f}")
        print(f"  Precision: {mm['precision']:.4f}")
        print(f"  Recall:    {mm['recall']:.4f}")
        print(f"  F1-Score:  {mm['f1_score']:.4f}")
        print(f"  TP: {mm['true_positives']}, TN: {mm['true_negatives']}, "
              f"FP: {mm['false_positives']}, FN: {mm['false_negatives']}")
        print()
        
        # Error summary
        print(f"Errors:")
        print(f"  False Positives: {results['false_positives_count']}")
        print(f"  False Negatives: {results['false_negatives_count']}")
        print()
        print("-" * 80)
        print()


def save_detailed_results(analysis_results: Dict[str, Dict], output_dir: Path):
    """Save detailed results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary metrics
    summary_data = []
    for model_name, results in analysis_results.items():
        if 'error' in results:
            continue
        
        mm = results['manipulation_metrics']
        summary_data.append({
            'model': model_name,
            'total_messages': results['total_messages'],
            'accuracy': mm['accuracy'],
            'precision': mm['precision'],
            'recall': mm['recall'],
            'f1_score': mm['f1_score'],
            'true_positives': mm['true_positives'],
            'true_negatives': mm['true_negatives'],
            'false_positives': mm['false_positives'],
            'false_negatives': mm['false_negatives'],
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'summary_metrics.csv', index=False)
    print(f"Saved summary metrics to {output_dir / 'summary_metrics.csv'}")
    
    # Detailed predictions per model
    for model_name, results in analysis_results.items():
        if 'error' in results:
            continue
        
        pred_df = pd.DataFrame(results['predictions'])
        pred_df.to_csv(output_dir / f'{model_name}_detailed_predictions.csv', index=False)
        print(f"Saved detailed predictions for {model_name} to {output_dir / f'{model_name}_detailed_predictions.csv'}")


def main():
    """Main analysis function."""
    # Paths
    script_dir = Path(__file__).parent
    experiments_dir = script_dir
    results_dir = experiments_dir / 'results'
    test_csv_path = script_dir.parent / 'data' / 'test.csv'
    output_dir = experiments_dir / 'analysis_output'
    
    try:
        # Load data
        print("Loading test dataset...")
        test_df = load_test_dataset(str(test_csv_path))
        print(f"Loaded {len(test_df)} test samples")
        
        print("\nLoading experiment results...")
        experiment_results = load_experiment_results(results_dir)
        print(f"Found {len(experiment_results)} experiment result files")
        
        # Analyze each model
        print("\nAnalyzing results...")
        analysis_results = {}
        for model_name, model_data in experiment_results.items():
            print(f"  Analyzing {model_name}...")
            analysis_results[model_name] = analyze_model_results(
                test_df, model_data, model_name
            )
        
        # Print report
        print("\n" + "=" * 80)
        print_analysis_report(analysis_results)
        
        # Save detailed results
        print("Saving detailed results...")
        save_detailed_results(analysis_results, output_dir)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

