from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, random_state=42):
    wine = load_wine()
    X = wine.data
    y = wine.target

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
