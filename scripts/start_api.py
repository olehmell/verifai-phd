#!/usr/bin/env python3
"""
Simple script to start the VerifAI API server.
This script starts the uvicorn server for the FastAPI application.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Ensure we're in the project root (go up from scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # Check if we're in the right directory
    if not Path("api/main.py").exists():
        print("Error: api/main.py not found. Please run this script from the project root directory.")
        sys.exit(1)

    # Check if .env file exists
    if not Path(".env").exists():
        print("Warning: .env file not found. Please make sure you have configured your environment variables.")
        print("You can copy .env.example to .env and fill in the required values.")

    # Start the API server
    print("Starting VerifAI API server...")
    print("API will be available at: http://localhost:8000")
    print("Health check: http://localhost:8000/health")
    print("API documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")

    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\nAPI server stopped.")
    except FileNotFoundError:
        print("Error: uvicorn not found. Please install dependencies with: uv sync")
        sys.exit(1)

if __name__ == "__main__":
    main()