import pytest

from promptforge.engine import (
    MAX_CONTEXT_CHARS, MAX_TASK_CHARS, TECHNIQUES, PromptConfig,
    build_messages, make_optimization_messages, messages_to_text, parse_examples,
)


def test_zero_shot_has_system_and_user():
    result = build_messages(PromptConfig("Zero-shot", "Explain RAG"))
    assert [m["role"] for m in result] == ["system", "user"]
    assert "Explain RAG" in result[-1]["content"]


def test_all_techniques_build():
    for technique in TECHNIQUES:
        cfg = PromptConfig(technique, "Classify customer sentiment",
                           context="Review is positive", examples="Great => positive")
        assert build_messages(cfg)[-1]["role"] == "user"


def test_role_is_on_system_instruction():
    messages = build_messages(PromptConfig("Role-based", "Hello", role="a teacher"))
    assert "You are a teacher" in messages[0]["content"]


def test_few_shot_examples_as_separate_messages():
    messages = build_messages(PromptConfig("Few-shot", "OK", examples="A => B\nC => D"))
    assert [x["role"] for x in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[2]["content"] == "B"


def test_structured_json_mentions_schema():
    messages = build_messages(PromptConfig("Structured JSON", "Summarize"))
    assert "key_points" in messages[0]["content"]


def test_grounded_rejects_missing_context():
    with pytest.raises(ValueError, match="requires reference context"):
        build_messages(PromptConfig("Context-grounded", "Question"))


def test_grounded_context_marked_untrusted():
    msgs = build_messages(PromptConfig("Context-grounded", "Question", context="Some data"))
    assert "untrusted" in msgs[0]["content"]
    assert "<reference_material>" in msgs[-1]["content"]


def test_empty_task_rejected():
    with pytest.raises(ValueError, match="Enter a task"):
        build_messages(PromptConfig("Zero-shot", "   "))


def test_length_limits():
    with pytest.raises(ValueError, match="4000"):
        build_messages(PromptConfig("Zero-shot", "x" * (MAX_TASK_CHARS + 1)))
    with pytest.raises(ValueError, match="Context"):
        build_messages(PromptConfig("Zero-shot", "OK", context="x" * (MAX_CONTEXT_CHARS + 1)))


def test_invalid_examples_rejected():
    with pytest.raises(ValueError, match="=>"):
        parse_examples("bad format")
    with pytest.raises(ValueError, match="nonempty"):
        parse_examples("A =>")


def test_few_shot_must_have_examples():
    with pytest.raises(ValueError, match="at least one"):
        build_messages(PromptConfig("Few-shot", "Classify"))


def test_unsupported_technique():
    with pytest.raises(ValueError, match="Unsupported"):
        build_messages(PromptConfig("Do-anything", "Test"))


def test_preview_has_roles():
    text = messages_to_text(build_messages(PromptConfig("Zero-shot", "Hello")))
    assert "[SYSTEM]" in text and "[USER]" in text


def test_optimizer_treats_input_as_data():
    messages = make_optimization_messages("Write email")
    assert "data to improve" in messages[0]["content"]
    assert "Write email" in messages[-1]["content"]


def test_optimizer_rejects_blank():
    with pytest.raises(ValueError):
        make_optimization_messages(" ")
