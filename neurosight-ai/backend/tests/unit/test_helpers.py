"""Tests for utility helper functions."""
import pytest
from app.utils.helpers import (
    clamp, normalize, safe_divide, slugify, truncate,
    flatten_dict, chunk_list, moving_average, percentage_change, is_valid_uuid,
)


def test_clamp_within():
    assert clamp(5.0, 0.0, 10.0) == 5.0

def test_clamp_below():
    assert clamp(-1.0, 0.0, 10.0) == 0.0

def test_clamp_above():
    assert clamp(15.0, 0.0, 10.0) == 10.0

def test_normalize_midpoint():
    assert normalize(50.0, 0.0, 100.0) == pytest.approx(0.5)

def test_normalize_zero_span():
    assert normalize(5.0, 5.0, 5.0) == 0.0

def test_safe_divide_normal():
    assert safe_divide(10.0, 2.0) == 5.0

def test_safe_divide_by_zero():
    assert safe_divide(10.0, 0.0, fallback=-1.0) == -1.0

def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"

def test_slugify_special_chars():
    assert slugify("AI & ML: Fundamentals!") == "ai-ml-fundamentals"

def test_truncate_short():
    assert truncate("hello", 10) == "hello"

def test_truncate_long():
    result = truncate("hello world", 8)
    assert len(result) == 8
    assert result.endswith("...")

def test_flatten_dict():
    nested = {"a": {"b": {"c": 1}, "d": 2}, "e": 3}
    flat = flatten_dict(nested)
    assert flat["a.b.c"] == 1
    assert flat["a.d"] == 2
    assert flat["e"] == 3

def test_chunk_list():
    chunks = chunk_list([1, 2, 3, 4, 5], 2)
    assert chunks == [[1, 2], [3, 4], [5]]

def test_chunk_list_exact():
    chunks = chunk_list([1, 2, 3, 4], 2)
    assert chunks == [[1, 2], [3, 4]]

def test_moving_average_basic():
    result = moving_average([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
    assert len(result) == 5
    assert result[2] == pytest.approx(2.0)

def test_percentage_change_up():
    assert percentage_change(100.0, 150.0) == pytest.approx(50.0)

def test_percentage_change_down():
    assert percentage_change(100.0, 80.0) == pytest.approx(-20.0)

def test_percentage_change_zero_base():
    assert percentage_change(0.0, 50.0) == 0.0

def test_is_valid_uuid_true():
    import uuid
    assert is_valid_uuid(str(uuid.uuid4())) is True

def test_is_valid_uuid_false():
    assert is_valid_uuid("not-a-uuid") is False

def test_is_valid_uuid_empty():
    assert is_valid_uuid("") is False
