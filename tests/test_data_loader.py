from src.data_loader import load_data


def test_load_data_returns_split_with_expected_sizes():
    X_train, X_test, y_train, y_test = load_data(test_size=0.2, random_state=42)

    total_samples = len(X_train) + len(X_test)
    assert total_samples == 178
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
