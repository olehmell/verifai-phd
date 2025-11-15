"""Tests for fact checker node with Perplexity integration"""
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verifai.nodes.fact_checker import perform_web_search, load_trusted_domains


class TestLoadTrustedDomains:
    def test_load_trusted_domains_returns_list(self):
        """Test that load_trusted_domains returns a list"""
        domains = load_trusted_domains()
        assert isinstance(domains, list)
        assert len(domains) > 0


class TestPerformWebSearch:
    @patch('verifai.nodes.fact_checker.Perplexity')
    @patch('verifai.nodes.fact_checker.os.getenv')
    def test_perform_web_search_uses_perplexity_api(self, mock_getenv, mock_perplexity):
        """Test that perform_web_search calls Perplexity API"""
        # Setup mocks
        mock_getenv.return_value = "test_api_key"
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Test search result content"
        mock_completion.citations = ["https://stopfake.org/test"]
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_perplexity.return_value = mock_client
        
        # Execute
        results = perform_web_search("test query")
        
        # Verify
        assert isinstance(results, list)
        assert len(results) > 0
        
    @patch('verifai.nodes.fact_checker.os.getenv')
    def test_perform_web_search_requires_api_key(self, mock_getenv):
        """Test that perform_web_search requires PERPLEXITY_API_KEY"""
        mock_getenv.return_value = None
        
        results = perform_web_search("test query")
        
        # Should return empty list on error
        assert results == []
    
    @patch('verifai.nodes.fact_checker.Perplexity')
    @patch('verifai.nodes.fact_checker.os.getenv')
    def test_perform_web_search_returns_dict_with_url_and_snippet(self, mock_getenv, mock_perplexity):
        """Test that search results contain url and snippet"""
        mock_getenv.return_value = "test_api_key"
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Search result with information"
        mock_completion.citations = ["https://stopfake.org/article1"]
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_perplexity.return_value = mock_client
        
        results = perform_web_search("test query", num_results=1)
        
        assert len(results) > 0
        for result in results:
            assert isinstance(result, dict)
            assert "url" in result
            assert "snippet" in result
    
    @patch('verifai.nodes.fact_checker.Perplexity')
    @patch('verifai.nodes.fact_checker.os.getenv')
    def test_perform_web_search_handles_api_errors(self, mock_getenv, mock_perplexity):
        """Test that perform_web_search handles API errors gracefully"""
        mock_getenv.return_value = "test_api_key"
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_perplexity.return_value = mock_client
        
        results = perform_web_search("test query")
        
        # Should return empty list on error
        assert results == []

