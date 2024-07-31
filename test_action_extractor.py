import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from main import app, chain, HuggingFaceTextGenInference

client = TestClient(app)

def test_get_actions():
    response = client.get("/actions", params={"sentence": "Fetch data when a button is clicked, cache the data, and then display it on the screen."})
    assert response.status_code == 200
    assert response.json() == {"actions": "OnClick, FetchData, CacheData, Show"}

def test_huggingface_text_gen_inference_init():
    llm = chain.llm.llm
    assert llm.inference_server_url == "http://host.docker.internal:8080/"
    assert llm.max_new_tokens == 50
    assert llm.top_k == 10
    assert llm.temperature == 0.01
    assert llm.repetition_penalty == 1.0

def test_chain_run():
    mock_llm = MagicMock()
    mock_llm.run.return_value = "OnClick, FetchData, CacheData, Show"
    chain.llm.llm = mock_llm
    result = chain.run(text="Fetch data when a button is clicked, cache the data, and then display it on the screen.")
    assert result == "OnClick, FetchData, CacheData, Show"
