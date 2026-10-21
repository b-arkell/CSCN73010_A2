"""Shared pytest fixtures."""
import pytest

from app import app as flask_app


@pytest.fixture(name="client")
def client_fixture():
    """Provide a Flask test client for sending fake requests to the app."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client
