from pathlib import Path

from brainage.data.hcp_mmse import HCPMMSEExample
from brainage.experiments.run_hcp_mmse import load_split_assignments, write_split_assignments


def test_hcp_split_roundtrip(tmp_path: Path):
    examples = [
        HCPMMSEExample(subject_id="s1", image_path=tmp_path / "s1.nii.gz", mmse=29.0),
        HCPMMSEExample(subject_id="s2", image_path=tmp_path / "s2.nii.gz", mmse=28.0),
        HCPMMSEExample(subject_id="s3", image_path=tmp_path / "s3.nii.gz", mmse=27.0),
    ]
    split_sets = {
        "train": [examples[0]],
        "val": [examples[1]],
        "test": [examples[2]],
    }
    split_file = tmp_path / "hcp_split.csv"

    write_split_assignments(split_file, split_sets)
    loaded = load_split_assignments(split_file, examples)

    assert [item.subject_id for item in loaded["train"]] == ["s1"]
    assert [item.subject_id for item in loaded["val"]] == ["s2"]
    assert [item.subject_id for item in loaded["test"]] == ["s3"]
