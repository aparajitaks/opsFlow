import pytest
from unittest.mock import MagicMock, patch
from core.config import settings
from rag.generator import generate_llm_completion

def test_conversational_memory_injection():
    """Verifies that conversational history turns are correctly prepended in messages sequence."""
    query = "What causes power failure?"
    retrieved_chunks = [
        {"doc_name": "manual.txt", "chunk_index": 1, "text": "PWF is caused by high torque and high rotational speed."}
    ]
    
    # Define past turns
    history = [
        {"role": "user", "content": "How do I check equipment status?"},
        {"role": "assistant", "content": "Refer to section 2.1 in the maintenance manual."}
    ]
    
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Power failure is caused by high mechanical strain (torque * speed)."
    mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=20, total_tokens=120)
    mock_client.chat.completions.create.return_value = mock_response
    
    # Call completion
    answer = generate_llm_completion(query, retrieved_chunks, mock_client, conversation_history=history)
    
    assert answer == "Power failure is caused by high mechanical strain (torque * speed)."
    
    # Assert on messages structured sequence passed to Groq
    mock_client.chat.completions.create.assert_called_once()
    kwargs = mock_client.chat.completions.create.call_args[1]
    messages = kwargs["messages"]
    
    # Messages should be:
    # 0: system instructions
    # 1: historical user query
    # 2: historical assistant response
    # 3: current user query with context reference
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "How do I check equipment status?"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Refer to section 2.1 in the maintenance manual."
    assert messages[3]["role"] == "user"
    assert "What causes power failure?" in messages[3]["content"]
    assert "manual.txt" in messages[3]["content"]
