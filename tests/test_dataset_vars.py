"""Tests for dataset-level vars: $var_name substitution in pipeline node args.

Dataset entries can declare a ``vars:`` block:

    my_dataset:
      name: MyData
      file_pattern: "data/**/*.vhdr"
      vars:
        condition_name: EO_baseline
        epoch_duration: 2.0

Pipeline node args reference them with ``$var_name``:

    args:
      condition: $condition_name   # → "EO_baseline"
      duration:  $epoch_duration   # → 2.0

neurodags substitutes the value from the active dataset entry at runtime.
Only whole-string values matching the ``$identifier`` pattern are substituted;
embedded ``$`` characters in paths or other strings are left untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurodags.dag import _resolve_vars, run_derivative
from neurodags.definitions import DatasetConfig, NodeResult

# ---------------------------------------------------------------------------
# Unit tests: _resolve_vars
# ---------------------------------------------------------------------------


class TestResolveVars:
    def test_simple_string_substitution(self):
        assert _resolve_vars("$foo", {"foo": "bar"}) == "bar"

    def test_substitution_returns_non_string_type(self):
        result = _resolve_vars("$duration", {"duration": 2.0})
        assert result == 2.0
        assert isinstance(result, float)

    def test_integer_var(self):
        assert _resolve_vars("$n", {"n": 42}) == 42

    def test_bool_var(self):
        assert _resolve_vars("$flag", {"flag": True}) is True

    def test_list_var(self):
        assert _resolve_vars("$bands", {"bands": [1, 4]}) == [1, 4]

    def test_non_matching_string_untouched(self):
        assert _resolve_vars("hello", {"hello": "nope"}) == "hello"

    def test_partial_dollar_untouched(self):
        # Embedded $ in a path should not be substituted
        assert _resolve_vars("/path/$HOME/file", {}) == "/path/$HOME/file"

    def test_dict_values_substituted(self):
        result = _resolve_vars({"cond": "$c", "dur": "$d"}, {"c": "EO", "d": 2.0})
        assert result == {"cond": "EO", "dur": 2.0}

    def test_nested_dict(self):
        result = _resolve_vars({"outer": {"inner": "$x"}}, {"x": 99})
        assert result["outer"]["inner"] == 99

    def test_list_values_substituted(self):
        result = _resolve_vars(["$a", "$b", "literal"], {"a": 1, "b": 2})
        assert result == [1, 2, "literal"]

    def test_tuple_values_substituted(self):
        result = _resolve_vars(("$a",), {"a": "hi"})
        assert result == ("hi",)

    def test_non_string_scalar_untouched(self):
        assert _resolve_vars(3.14, {}) == 3.14
        assert _resolve_vars(None, {}) is None

    def test_empty_vars_no_refs_ok(self):
        assert _resolve_vars({"k": "v"}, {}) == {"k": "v"}

    def test_missing_var_raises_key_error(self):
        with pytest.raises(KeyError, match="Dataset var '\\$missing'"):
            _resolve_vars("$missing", {"other": 1})

    def test_missing_var_error_lists_available(self):
        with pytest.raises(KeyError, match="available_key"):
            _resolve_vars("$missing", {"available_key": 1})


# ---------------------------------------------------------------------------
# Integration: vars flow through run_derivative
# ---------------------------------------------------------------------------


def _ref_base(tmp_path: Path) -> Path:
    ref = tmp_path / "subject" / "sample"
    ref.parent.mkdir(parents=True, exist_ok=True)
    return ref


def _make_dataset_config(vars_dict: dict) -> DatasetConfig:
    return DatasetConfig(
        name="test_ds",
        file_pattern="data/**/*.vhdr",
        vars=vars_dict,
    )


def test_vars_substituted_in_node_args(tmp_path):
    """$param_value in node args is replaced by the dataset var at runtime."""
    derivative_def = {
        "overwrite": False,
        "nodes": [
            {
                "id": 0,
                "node": "dummy",
                "args": {"param1": "$my_param"},
            }
        ],
    }
    ds = _make_dataset_config({"my_param": "hello_from_var"})
    result = run_derivative(
        derivative_def,
        derivative_name="TestVars",
        file_path="input.vhdr",
        reference_base=_ref_base(tmp_path),
        dataset_config=ds,
    )
    assert isinstance(result, NodeResult)
    assert "hello_from_var" in result.artifacts[".message.txt"].item


def test_vars_numeric_value(tmp_path):
    """Numeric vars are passed as their native type, not as strings."""
    derivative_def = {
        "overwrite": False,
        "nodes": [
            {
                "id": 0,
                "node": "dummy",
                "args": {"param1": "$duration"},
            }
        ],
    }
    ds = _make_dataset_config({"duration": 2.0})
    result = run_derivative(
        derivative_def,
        derivative_name="TestNumericVar",
        file_path="input.vhdr",
        reference_base=_ref_base(tmp_path),
        dataset_config=ds,
    )
    assert isinstance(result, NodeResult)


def test_no_vars_no_substitution(tmp_path):
    """Pipeline with no $refs and dataset with no vars runs without error."""
    derivative_def = {
        "overwrite": False,
        "nodes": [
            {"id": 0, "node": "dummy", "args": {"param1": "literal_value"}},
        ],
    }
    ds = _make_dataset_config({})
    result = run_derivative(
        derivative_def,
        derivative_name="TestNoVars",
        file_path="input.vhdr",
        reference_base=_ref_base(tmp_path),
        dataset_config=ds,
    )
    assert isinstance(result, NodeResult)


def test_missing_var_raises_at_runtime(tmp_path):
    """Referencing an undefined var raises KeyError with the var name."""
    derivative_def = {
        "overwrite": False,
        "nodes": [
            {"id": 0, "node": "dummy", "args": {"param1": "$undefined_var"}},
        ],
    }
    ds = _make_dataset_config({})
    with pytest.raises(KeyError, match="undefined_var"):
        run_derivative(
            derivative_def,
            derivative_name="TestMissingVar",
            file_path="input.vhdr",
            reference_base=_ref_base(tmp_path),
            dataset_config=ds,
        )


def test_vars_none_dataset_config_still_works(tmp_path):
    """run_derivative with dataset_config=None (no vars) runs normally."""
    derivative_def = {
        "overwrite": False,
        "nodes": [
            {"id": 0, "node": "dummy", "args": {"param1": "static"}},
        ],
    }
    result = run_derivative(
        derivative_def,
        derivative_name="TestNullDataset",
        file_path="input.vhdr",
        reference_base=_ref_base(tmp_path),
        dataset_config=None,
    )
    assert isinstance(result, NodeResult)


def test_vars_and_id_refs_coexist(tmp_path):
    """$var_name and id.N refs can both appear in the same derivative."""
    derivative_def = {
        "overwrite": False,
        "nodes": [
            {"id": 0, "node": "dummy", "args": {"param1": "first"}},
            {
                "id": 1,
                "node": "dummy",
                "args": {
                    "param1": "id.0",  # id ref
                    "param2": "$extra_param",  # var ref
                },
            },
        ],
    }
    ds = _make_dataset_config({"extra_param": "from_var"})
    result = run_derivative(
        derivative_def,
        derivative_name="TestMixed",
        file_path="input.vhdr",
        reference_base=_ref_base(tmp_path),
        dataset_config=ds,
    )
    assert isinstance(result, NodeResult)


# ---------------------------------------------------------------------------
# DatasetConfig model
# ---------------------------------------------------------------------------


class TestDatasetConfigVars:
    def test_vars_field_accepted(self):
        ds = DatasetConfig(
            name="ds",
            file_pattern="data/**/*.vhdr",
            vars={"condition_name": "EO_baseline", "epoch_duration": 2.0},
        )
        assert ds.vars == {"condition_name": "EO_baseline", "epoch_duration": 2.0}

    def test_vars_defaults_to_none(self):
        ds = DatasetConfig(name="ds", file_pattern="data/**/*.vhdr")
        assert ds.vars is None

    def test_vars_survives_serialization_roundtrip(self):
        ds = DatasetConfig(
            name="ds",
            file_pattern="data/**/*.vhdr",
            vars={"k": "v", "n": 42},
        )
        if hasattr(ds, "model_dump"):
            payload = ds.model_dump(mode="json")
        else:
            payload = ds.dict()
        ds2 = DatasetConfig(**payload)
        assert ds2.vars == {"k": "v", "n": 42}
