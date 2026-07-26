from fraud_detection.data import generate_synthetic_fraud_data, split_dataset, validate_dataset


def test_synthetic_data_schema_and_split():
    df = generate_synthetic_fraud_data(n_transactions=1500, seed=7)
    validate_dataset(df)
    train, val, test = split_dataset(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert 0 < df["is_fraud"].mean() < 0.20
    assert set(train["is_fraud"].unique()) == {0, 1}
