"""
Logging utility module for VerifAI metrics collection.
Provides structured logging for all steps in the analysis pipeline.
"""
import logging
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class MetricsLogger:
    """Logger for collecting metrics at each step of the analysis pipeline"""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the metrics logger
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup structured logger
        self.logger = logging.getLogger("verifai.metrics")
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # File handler for metrics
            metrics_file = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
            file_handler = logging.FileHandler(metrics_file)
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_step(self, 
                 step_name: str,
                 content_id: Optional[str] = None,
                 duration: Optional[float] = None,
                 input_data: Optional[Dict[str, Any]] = None,
                 output_data: Optional[Dict[str, Any]] = None,
                 metrics: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None):
        """
        Log a pipeline step with metrics
        
        Args:
            step_name: Name of the step (e.g., 'manipulation_classifier')
            content_id: Optional identifier for the content being analyzed
            duration: Time taken for this step in seconds
            input_data: Input data for this step
            output_data: Output data from this step
            metrics: Additional metrics dictionary
            error: Error message if step failed
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step_name,
            "content_id": content_id,
            "duration_seconds": duration,
            "input": input_data,
            "output": output_data,
            "metrics": metrics or {},
            "error": error,
            "success": error is None
        }
        
        # Log as JSON for structured logging
        self.logger.info(json.dumps(log_entry, ensure_ascii=False, default=str))
    
    def log_classification(self,
                          content_id: str,
                          duration: float,
                          manipulation_probability: float,
                          techniques: list,
                          content_length: int):
        """Log manipulation classification step"""
        self.log_step(
            step_name="manipulation_classifier",
            content_id=content_id,
            duration=duration,
            output_data={
                "manipulation_probability": manipulation_probability,
                "techniques": techniques
            },
            metrics={
                "content_length": content_length,
                "techniques_count": len(techniques)
            }
        )
    
    def log_narrative_extraction(self,
                                content_id: str,
                                duration: float,
                                narrative: str,
                                techniques: list):
        """Log narrative extraction step"""
        self.log_step(
            step_name="narrative_extractor",
            content_id=content_id,
            duration=duration,
            input_data={"techniques": techniques},
            output_data={"narrative": narrative},
            metrics={
                "narrative_length": len(narrative) if narrative else 0
            }
        )
    
    def log_fact_checking(self,
                         content_id: str,
                         duration: float,
                         queries: list,
                         results_count: int,
                         fact_check_results: str):
        """Log fact checking step"""
        self.log_step(
            step_name="fact_checker",
            content_id=content_id,
            duration=duration,
            output_data={
                "queries": queries,
                "results_count": results_count,
                "fact_check_results": fact_check_results
            },
            metrics={
                "queries_count": len(queries)
            }
        )
    
    def log_verification(self,
                        content_id: str,
                        duration: float,
                        final_result: Dict[str, Any]):
        """Log verification step"""
        self.log_step(
            step_name="verifier",
            content_id=content_id,
            duration=duration,
            output_data=final_result,
            metrics={
                "disinfo_count": len(final_result.get("disinfo", []))
            }
        )
    
    def log_pipeline_complete(self,
                             content_id: str,
                             total_duration: float,
                             final_result: Dict[str, Any]):
        """Log completion of entire pipeline"""
        self.log_step(
            step_name="pipeline_complete",
            content_id=content_id,
            duration=total_duration,
            output_data=final_result,
            metrics={
                "total_duration": total_duration,
                "manipulation_detected": final_result.get("manipulation", False),
                "techniques_count": len(final_result.get("techniques", [])),
                "disinfo_count": len(final_result.get("disinfo", []))
            }
        )


# Global logger instance
_metrics_logger = None


def get_logger() -> MetricsLogger:
    """Get or create the global metrics logger instance"""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = MetricsLogger()
    return _metrics_logger

