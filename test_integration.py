import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_actions_integration():
    response = client.get("/actions", params={"sentence": "Play a sound when a key is pressed and stop it when the key is released."})
    assert response.status_code == 200
    assert response.json() == {"actions": "OnKeyPress, PlaySound, OnKeyRelease, StopSound"}
