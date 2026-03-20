from brainage.data.split_builders import build_lodo_split_rows


def test_lodo_split_builder_assigns_holdout_to_test():
    records = [
        {"subject_id": "a1", "cohort": "hcp"},
        {"subject_id": "a2", "cohort": "hcp"},
        {"subject_id": "b1", "cohort": "oasis"},
        {"subject_id": "c1", "cohort": "adni"},
    ]

    rows = build_lodo_split_rows(records, holdout_cohort="adni", validation_ratio=0.5, seed=42)
    adni_rows = [row for row in rows if row["cohort"] == "adni"]

    assert adni_rows
    assert all(row["split"] == "test" for row in adni_rows)
