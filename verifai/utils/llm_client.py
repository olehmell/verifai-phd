"""LLM Client abstraction for supporting multiple LLM backends"""

from typing import Protocol, Optional
from google import genai
import os
import httpx
import json


class LLMClient(Protocol):
    """Protocol for LLM client interface"""
    
    def generate_content(self, prompt: str) -> str:
        """Generate content from a prompt"""
        ...


class GeminiClient:
    """Client for Google Gemini API"""
    
    def __init__(self, model_id: str):
        """
        Initialize Gemini client.
        
        Args:
            model_id: Model identifier (e.g., 'gemini-2.5-flash')
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
    
    def generate_content(self, prompt: str) -> str:
        """
        Generate content using Gemini API.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            Generated text response
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            return response.candidates[0].content.parts[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")


class LocalLLMClient:
    """Client for OpenAI-compatible local LLM APIs"""
    
    def __init__(self, model_id: str, base_url: str = "http://127.0.0.1:1234"):
        """
        Initialize local LLM client.
        
        Args:
            model_id: Model identifier (e.g., 'mamaylm-gemma-3-4b-it-v1.0')
            base_url: Base URL for the local LLM API (default: http://127.0.0.1:1234)
        """
        self.model_id = model_id
        self.base_url = base_url.rstrip('/')
        self.client = httpx.Client(timeout=300.0)  # 5 minute timeout for local models
    
    def generate_content(self, prompt: str) -> str:
        """
        Generate content using OpenAI-compatible API.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            Generated text response
        """
        try:
            url = f"{self.base_url}/v1/chat/completions"
            
            response = self.client.post(
                url,
                json={
                    "model": self.model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4096
                }
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Local LLM API HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise RuntimeError(f"Local LLM API request error: {str(e)}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Local LLM API response parsing error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Local LLM API error: {str(e)}")
    
    def __del__(self):
        """Close HTTP client on cleanup"""
        if hasattr(self, 'client'):
            self.client.close()


def create_llm_client(model_type: str, model_id: str, base_url: Optional[str] = None) -> LLMClient:
    """
    Factory function to create an LLM client based on type.
    
    Args:
        model_type: Type of model ('gemini' or 'local')
        model_id: Model identifier
        base_url: Base URL for local models (required if model_type is 'local')
        
    Returns:
        LLMClient instance
        
    Raises:
        ValueError: If model_type is invalid or base_url is missing for local models
    """
    if model_type == "gemini":
        return GeminiClient(model_id)
    elif model_type == "local":
        if base_url is None:
            raise ValueError("base_url is required for local models")
        return LocalLLMClient(model_id, base_url)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'gemini' or 'local'")

