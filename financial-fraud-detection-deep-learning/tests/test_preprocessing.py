from fraud_detection.data import generate_synthetic_fraud_data
from fraud_detection.preprocessing import FraudPreprocessor


def test_unknown_category_maps_to_zero():
    df = generate_synthetic_fraud_data(n_transactions=1200, seed=4)
    preprocessor = FraudPreprocessor.create().fit(df.iloc[:1000])
    probe = df.iloc[[0]].copy()
    probe["merchant"] = "never_seen_merchant"
    cats, nums = preprocessor.transform(probe)
    merchant_index = preprocessor.categorical_features.index("merchant")
    assert cats[0, merchant_index] == 0
    assert nums.shape[1] == 4
