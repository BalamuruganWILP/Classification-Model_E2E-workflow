"""

SMOTE utilities for the CTG classification project.

IMPORTANT:
SMOTE must ONLY be applied to training data.

The test dataset must never be oversampled.
"""

from imblearn.over_sampling import SMOTE


def apply_smote(
    X_train,
    y_train,
    random_state=42,
    k_neighbors=5
):
    """
    Apply SMOTE to training data.

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
    y_train : pandas.Series or numpy.ndarray

    random_state : int
        Reproducibility seed.

    k_neighbors : int
        Number of nearest neighbors used by SMOTE.

    Returns
    -------
    X_resampled
    y_resampled
    smote
    """

    smote = SMOTE(
        random_state=random_state,
        k_neighbors=k_neighbors
    )

    X_resampled, y_resampled = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    return (
        X_resampled,
        y_resampled,
        smote
    )