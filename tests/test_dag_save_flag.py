from pathlib import Path

from cocofeats.dag import run_feature
from cocofeats.definitions import NodeResult


def _build_feature_def(*, save: bool | None = None) -> dict:
    definition: dict[str, object] = {
        "nodes": [
            {
                "id": 0,
                "node": "dummy",
                "args": {"param1": "foo"},
            }
        ]
    }
    if save is not None:
        definition["save"] = save
    return definition


def _run_dummy_feature(tmp_path: Path, feature_def: dict) -> tuple[NodeResult, Path]:
    reference_base = tmp_path / "subject" / "sample"
    reference_base.parent.mkdir(parents=True, exist_ok=True)
    result = run_feature(
        feature_def,
        feature_name="dummy_feature",
        file_path="input-file.vhdr",
        reference_base=reference_base,
    )
    output_path = Path(f"{reference_base.as_posix()}@DummyFeature.message.txt")
    return result, output_path


def test_run_feature_saves_artifacts_by_default(tmp_path):
    feature_def = _build_feature_def()
    result, output_path = _run_dummy_feature(tmp_path, feature_def)

    assert isinstance(result, NodeResult)
    assert output_path.exists()
    assert output_path.read_text() == "Dummy feature extraction completed with param1=foo and param2=None"


def test_run_feature_skips_saving_when_flag_false(tmp_path):
    feature_def = _build_feature_def(save=False)
    result, output_path = _run_dummy_feature(tmp_path, feature_def)

    assert isinstance(result, NodeResult)
    assert not output_path.exists()


def test_save_false_does_not_trigger_cached_shortcut(tmp_path):
    save_feature = _build_feature_def()
    _, output_path = _run_dummy_feature(tmp_path, save_feature)
    assert output_path.exists()

    nosave_feature = _build_feature_def(save=False)
    result, _ = _run_dummy_feature(tmp_path, nosave_feature)

    assert isinstance(result, NodeResult)
