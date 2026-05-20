import pytest
from unittest.mock import MagicMock, patch
from rag import is_valid_groq_key, get_groq_client

def test_is_valid_groq_key():
    assert not is_valid_groq_key(None)
    assert not is_valid_groq_key("")
    assert not is_valid_groq_key("   ")
    assert not is_valid_groq_key("None")
    assert not is_valid_groq_key("none")
    assert not is_valid_groq_key("undefined")
    assert is_valid_groq_key("gsk_some_valid_looking_key_12345")

@patch("rag.generator.Groq")
def test_get_groq_client_caching(mock_groq):
    # Clear cache singleton to avoid side-effects between runs
    get_groq_client.clear()
    
    # 1. Test invalid keys return None
    assert get_groq_client(None) is None
    assert get_groq_client("none") is None
    assert get_groq_client("   ") is None
    
    # 2. Test valid key returns a client
    mock_instance = MagicMock()
    mock_groq.return_value = mock_instance
    
    client1 = get_groq_client("gsk_test_key_abc")
    assert client1 is mock_instance
    
    # 3. Test caching: calling again with same key shouldn't re-instantiate Groq
    client2 = get_groq_client("gsk_test_key_abc")
    assert client2 is client1
    assert mock_groq.call_count == 1
