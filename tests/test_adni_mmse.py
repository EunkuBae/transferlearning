from pathlib import Path

from brainage.data.adni_mmse import discover_adni_mmse_examples, normalize_adni_diagnosis


def test_normalize_adni_diagnosis_maps_expected_labels():
    assert normalize_adni_diagnosis("CN") == "NC"
    assert normalize_adni_diagnosis("LMCI") == "MCI"
    assert normalize_adni_diagnosis("AD") == "AD"
    assert normalize_adni_diagnosis("unknown") is None


def test_discover_adni_examples_matches_existing_nii(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "ADNI_001.nii"
    image_path.write_bytes(b"fake")

    metadata_path = tmp_path / "adni.csv"
    metadata_path.write_text(
        "PTID,DX_bl,AGE,PTGENDER,MMSE,copied_file,file_name,source_file\n"
        "001_S_0001,CN,70.0,Female,29,C:\\data\\ADNI_001.nii,,\n",
        encoding="utf-8",
    )

    examples = discover_adni_mmse_examples(metadata_path=metadata_path, image_dir=image_dir)

    assert len(examples) == 1
    assert examples[0].subject_id == "001_S_0001"
    assert examples[0].mmse == 29.0
