import pytest
from app import app

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
        "barna": 6,
        "knocknacarra": 8,
        "oranmore": 7
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
        "barna": 6,
        "knocknacarra": 8,
        "oranmore": 7
    } 