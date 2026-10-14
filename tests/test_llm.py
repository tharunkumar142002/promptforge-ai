from types import SimpleNamespace

import pytest

from promptforge.llm import ModelCallError, generate


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary":"hello"}'))],
            model="llama-3.3-70b-versatile",
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7, total_tokens=19),
        )


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_offline_mock_call():
    mock = FakeClient()
    result = generate(api_key="dummy_key_only", client=mock,
                      messages=[{"role": "user", "content": "Hello"}], json_mode=True)
    assert result.total_tokens == 19
    assert result.text.startswith("{")
    assert mock.completions.kwargs["response_format"] == {"type": "json_object"}
    assert mock.completions.kwargs["max_completion_tokens"] == 700


def test_missing_key():
    with pytest.raises(ModelCallError, match="API key"):
        generate(api_key="", messages=[{"role": "user", "content": "x"}], client=FakeClient())


def test_invalid_temperature():
    with pytest.raises(ValueError):
        generate(api_key="dummy", messages=[{"role": "user", "content": "x"}],
                 temperature=1.4, client=FakeClient())


def test_unsupported_model():
    with pytest.raises(ModelCallError, match="supported"):
        generate(api_key="dummy", messages=[{"role": "user", "content": "x"}],
                 model="invalid-model", client=FakeClient())
