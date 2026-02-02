import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web_app import app

@pytest.fixture
def client():
    """Simulate a client to test Flask app"""
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client
