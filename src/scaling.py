"""
The scaler is fitted ONLY on training data.
The same fitted scaler is then used to transform
the test data.
"""

from sklearn.preprocessing import StandardScaler


def create_scaler():
    """
    Create a StandardScaler.

    Returns:     StandardScaler
    """

    return StandardScaler()


def fit_scaler(X_train):
    """
    Fit the scaler using training data only.

    Parameters
    ----------
    X_train : pandas.DataFrame

    Returns
    -------
    scaler : StandardScaler
    """

    scaler = create_scaler()

    scaler.fit(X_train)

    return scaler


def transform_data(scaler, X):
    """
    Transform data using an already-fitted scaler.

    Parameters
    ----------
    scaler : StandardScaler
    X : pandas.DataFrame

    Returns
    -------
    numpy.ndarray
    """

    return scaler.transform(X)


def scale_train_test(X_train, X_test):
    """
    Fit scaler on training data and transform both
    training and test data.

    IMPORTANT:
    X_test is never used to fit the scaler.
    """

    scaler = fit_scaler(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )