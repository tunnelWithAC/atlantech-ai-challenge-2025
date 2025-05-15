import pytest
from app import app
import json
from unittest.mock import patch
import requests

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_get_score_success(client):
    """Test successful score retrieval for a valid office"""
    response = client.get('/score?office_name=platform94')
    assert response.status_code == 200
    data = response.get_json()
    assert data == {
        "scores": {
            "barna": 6,
            "knocknacarra": 8,
            "oranmore": 7
        }
    }

def test_get_score_missing_parameter(client):
    """Test error response when office_name parameter is missing"""
    response = client.get('/score')
    assert response.status_code == 400
    data = response.get_json()
    assert data == {"error": "office_name parameter is required"}

def test_get_score_invalid_office(client):
    """Test error response for non-existent office"""
    response = client.get('/score?office_name=nonexistent')
    assert response.status_code == 404
    data = response.get_json()
    assert data == {"error": "No data found for office: nonexistent"}

def test_get_score_case_insensitive(client):
    """Test that office name matching is case insensitive"""
    response = client.get('/score?office_name=PLATFORM94')
    assert response.status_code == 200
    data = response.get_json()
    assert data == {
        "scores": {
            "barna": 6,
            "knocknacarra": 8,
            "oranmore": 7
        }
    }

def test_ollama_request_success(client):
    """Test successful Ollama request"""
    mock_response = {
        "response": "This is a test response",
        "model": "llama2"
    }
    
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": "This is a test response"}
        
        response = client.post('/ollama', 
                             json={"prompt": "Test prompt"},
                             content_type='application/json')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["response"] == "This is a test response"
        assert data["model"] == "llama2"

def test_ollama_request_missing_prompt(client):
    """Test error when prompt is missing"""
    response = client.post('/ollama', 
                         json={},
                         content_type='application/json')
    
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "prompt is required"

def test_ollama_request_connection_error(client):
    """Test error handling when Ollama is not available"""
    with patch('requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")
        
        response = client.post('/ollama', 
                             json={"prompt": "Test prompt"},
                             content_type='application/json')
        
        assert response.status_code == 500
        data = response.get_json()
        assert "Failed to connect to Ollama" in data["error"] 