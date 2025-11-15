import os
import uuid
from typing import Optional
from dotenv import load_dotenv
from verifai import create_graph
from verifai.prompts.manipulation_classifier import MANIPULATION_TECHNIQUE_DESCRIPTIONS
from verifai.utils.logging import get_logger

# Load environment variables from .env file
load_dotenv()

def analyze_content(content: str, content_id: Optional[str] = None):
    """
    Analyze content using the VerifAI multi-agent system.

    Args:
        content: The content to analyze
        content_id: Optional identifier for tracking metrics

    Returns:
        Final analysis result
    """
    logger = get_logger()
    
    # Generate content_id if not provided
    if content_id is None:
        content_id = str(uuid.uuid4())
    
    # Create the graph
    graph = create_graph()

    # Run the analysis with content_id
    result = graph.invoke({
        "content": content,
        "content_id": content_id
    })

    return result.get("final_result", {})

def main():
    """Main entry point for VerifAI"""
    # Check for required environment variables
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is required")
        print("Please set it in your .env file or environment")
        return

    # Example usage
    sample_content = """
   ❗️По материалам СБУ мэру Одессы Геннадию Труханову прекращено гражданство Украины

Такое решение, в частности, основано на доказательной базе Службы безопасности и утверждено Указом Президента Украины. Как установила СБУ, по состоянию на данный момент действующий городской голова Одессы Геннадий Труханов является гражданином РФ и имеет действующий заграничный паспорт страны-агрессора.
    """

    print("VerifAI - Multi-Agent Misinformation Detection System")
    print("=" * 55)
    print(f"Analyzing sample content: {sample_content[:100]}...")
    print()

    try:
        # Run analysis
        result = analyze_content(sample_content)

        # Display results
        print("ANALYSIS RESULTS:")
        print("-" * 20)
        print(f"Manipulation detected: {result.get('manipulation', False)}")

        techniques = result.get('techniques', [])
        print(f"Techniques found: {techniques}")

        # Show detailed descriptions for each technique
        if techniques:
            print("\nTECHNIQUE DETAILS:")
            for technique in techniques:
                description = MANIPULATION_TECHNIQUE_DESCRIPTIONS.get(technique, f"Unknown technique: {technique}")
                print(f"- {description}")

        print(f"Disinformation flags: {len(result.get('disinfo', []))}")
        print()
        print("EXPLANATION:")
        print(result.get('explanation', 'No explanation available'))
        print()

        if result.get('disinfo'):
            print("DISINFORMATION DETAILS:")
            for item in result.get('disinfo', []):
                print(f"- {item}")

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please check your environment variables and configuration")
    except Exception as e:
        print(f"Unexpected error running analysis: {e}")
        print("This might be due to network issues, API limits, or unexpected content")
        print("Please try again or check your internet connection")

if __name__ == "__main__":
    main()
