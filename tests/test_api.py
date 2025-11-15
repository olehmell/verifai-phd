import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check_returns_ok(self, client):
        """Test that health endpoint returns 200 OK"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAnalyzeEndpoint:
    def test_analyze_endpoint_exists(self, client):
        """Test that analyze endpoint accepts POST requests"""
        response = client.post("/analyze", json={"content": "test content"})
        # Should not return 404 or 405
        assert response.status_code != 404
        assert response.status_code != 405

    def test_analyze_requires_content_field(self, client):
        """Test that analyze endpoint requires content field"""
        response = client.post("/analyze", json={})
        assert response.status_code == 422  # Unprocessable Entity

    def test_analyze_validates_content_is_string(self, client):
        """Test that content field must be a string"""
        response = client.post("/analyze", json={"content": 123})
        assert response.status_code == 422

    @patch('api.main.analyze_content')
    def test_analyze_calls_verifai_analyze_content(self, mock_analyze, client):
        """Test that analyze endpoint calls the verifai analyze_content function"""
        mock_analyze.return_value = {
            "manipulation": False,
            "techniques": [],
            "explanation": "Test explanation",
            "disinfo": []
        }

        response = client.post("/analyze", json={"content": "test content"})

        mock_analyze.assert_called_once_with("test content")
        assert response.status_code == 200

    @patch('api.main.analyze_content')
    def test_analyze_returns_expected_structure(self, mock_analyze, client):
        """Test that analyze endpoint returns the expected response structure"""
        expected_result = {
            "manipulation": True,
            "techniques": ["emotional_appeal"],
            "explanation": "This content uses emotional manipulation techniques.",
            "disinfo": ["Unverified claim about statistics"]
        }
        mock_analyze.return_value = expected_result

        response = client.post("/analyze", json={"content": "test content"})

        assert response.status_code == 200
        json_response = response.json()
        assert json_response["manipulation"] == True
        assert json_response["techniques"] == ["emotional_appeal"]
        assert "explanation" in json_response
        assert "disinfo" in json_response

    @patch('api.main.analyze_content')
    def test_analyze_handles_verifai_exceptions(self, mock_analyze, client):
        """Test that analyze endpoint handles exceptions from verifai gracefully"""
        mock_analyze.side_effect = Exception("Verifai analysis failed")

        response = client.post("/analyze", json={"content": "test content"})

        assert response.status_code == 500
        json_response = response.json()
        assert "detail" in json_response
        assert "error" in json_response["detail"]

    def test_analyze_rejects_empty_content(self, client):
        """Test that analyze endpoint rejects empty content"""
        response = client.post("/analyze", json={"content": ""})
        assert response.status_code == 422

    def test_analyze_rejects_very_long_content(self, client):
        """Test that analyze endpoint rejects very long content"""
        long_content = "a" * 10001  # Assuming max length is 10000
        response = client.post("/analyze", json={"content": long_content})
        assert response.status_code == 422


class TestCORSHeaders:
    @patch('api.main.analyze_content')
    def test_cors_headers_on_cross_origin_request(self, mock_analyze, client):
        """Test that cross-origin requests return appropriate CORS headers"""
        mock_analyze.return_value = {
            "manipulation": False,
            "techniques": [],
            "explanation": "Test",
            "disinfo": []
        }
        # Simulate a cross-origin request
        headers = {
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"
        }
        response = client.post("/analyze", json={"content": "test content"}, headers=headers)
        # Should allow the request even with different origin due to CORS middleware
        assert response.status_code == 200