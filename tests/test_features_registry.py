import pytest

import cocofeats.features as features
from cocofeats.features.preprocessing import basic_preprocessing
from cocofeats.features.spectral import spectrum


def test_known_features_registered():
    registered = features.list_features()
    assert "basic_preprocessing" in registered
    assert "spectrum" in registered
    assert features.get_feature("basic_preprocessing") is basic_preprocessing
    assert features.get_feature("spectrum") is spectrum


def test_register_feature_duplicate_guard():
    @features.register_feature(name="temporary_feature")
    def temporary_feature():
        return "ok"

    try:
        with pytest.raises(ValueError):

            @features.register_feature(name="temporary_feature")
            def duplicate_feature():
                return "duplicate"

    finally:
        features.unregister_feature("temporary_feature")


def test_get_feature_unknown_raises_key_error():
    with pytest.raises(KeyError) as excinfo:
        features.get_feature("does_not_exist")

    assert "Unknown feature" in str(excinfo.value)
    assert "does_not_exist" in str(excinfo.value)
