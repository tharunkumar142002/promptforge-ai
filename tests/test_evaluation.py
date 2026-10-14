import pytest

from promptforge.evaluation import MAX_EVAL_ROWS, check_keywords, load_cases


def test_load_cases():
    rows = load_cases('input,expected_keywords\n"What is RAG?","retrieval;generation"\n')
    assert rows[0].task == "What is RAG?"
    assert rows[0].expected_keywords == ("retrieval", "generation")


def test_case_insensitive_checks():
    passed, missing = check_keywords("RETRIEVAL augmented GENERATION", ("retrieval", "generation"))
    assert passed and missing == []


def test_missing_keywords():
    passed, missing = check_keywords("This is a retrieval demo", ("retrieval", "generation"))
    assert not passed and missing == ["generation"]


def test_bad_header():
    with pytest.raises(ValueError, match="columns"):
        load_cases("question,expected\nhello,world")


def test_missing_fields():
    with pytest.raises(ValueError, match="row 2"):
        load_cases("input,expected_keywords\nhello,\n")


def test_too_many_rows():
    text = "input,expected_keywords\n" + "hi,hey\n" * (MAX_EVAL_ROWS + 1)
    with pytest.raises(ValueError, match="at most"):
        load_cases(text)


def test_empty_csv():
    with pytest.raises(ValueError):
        load_cases("input,expected_keywords\n")
