"""Reproducible stratified train/validation/test split."""
from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.15, validation_size=0.15, random_state=42):
    """Return train, validation and test splits, stratified by target.

    The requested fractions are fractions of the full valid-label dataset.
    """
    if not (0 < test_size < 1 and 0 < validation_size < 1 and test_size + validation_size < 1):
        raise ValueError("test_size and validation_size must be positive and sum to less than 1.")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_relative = validation_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative,
        random_state=random_state, stratify=y_train_val
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
