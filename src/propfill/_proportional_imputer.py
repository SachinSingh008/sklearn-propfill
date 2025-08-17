from __future__ import annotations
from typing import Iterable, List, Optional, Union, Dict, Any, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]

class ProportionalImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing categorical values to match the observed distribution.

    How it works (per column):
    - During fit: compute category probabilities from non-missing values.
    - During transform: for m missing cells, fill exactly floor(p*m) plus the
      largest remainders so the total equals m (Hamilton/Largest Remainder).
      Then shuffle for randomness (controlled by random_state).

    Parameters
    ----------
    columns : list or None, default=None
        Which columns to impute.
        - If X is a DataFrame and columns=None, we auto-pick object/string/category columns.
        - If X is a NumPy array, columns=None means "all columns".
    exact : bool, default=True
        If True, match counts exactly to proportions using largest remainders.
        If False, sample with numpy.random.choice using probabilities (approximate).
    random_state : int or None, default=None
        Controls the randomization (tie-breaking and fill order).
    copy : bool, default=True
        If True, work on a copy of X. If False, modify in place (if possible).

    Notes
    -----
    - If a column has all values missing at fit-time, we raise ValueError.
    - Only intended for categorical/text/string-like columns.
    """

    def __init__(
        self,
        columns: Optional[Iterable[Union[int, str]]] = None,
        exact: bool = True,
        random_state: Optional[int] = None,
        copy: bool = True,
    ):
        self.columns = columns
        self.exact = exact
        self.random_state = random_state
        self.copy = copy

    def _is_dataframe(self, X: ArrayLike) -> bool:
        return isinstance(X, pd.DataFrame)

    def _detect_categorical_columns(self, X: ArrayLike) -> Sequence[Union[int, str]]:
        # object, string dtype, pandas 'category'
        if isinstance(X, pd.DataFrame):
            cats = list(X.select_dtypes(include=["object", "string", "category"]).columns)
        elif isinstance(X, np.ndarray):
            # For ndarray, assume all columns are categorical
            n_cols = 1 if X.ndim == 1 else X.shape[1]
            cats = list(range(n_cols))
        else:
            raise TypeError("Input X must be a pandas DataFrame or numpy ndarray.")
        return cats

    def fit(self, X: ArrayLike, y: Any = None):
        rng = check_random_state(self.random_state)

        # Work with DataFrame or ndarray
        if self._is_dataframe(X):
            X_df = X
            if self.columns is None:
                cols = self._detect_categorical_columns(X_df)
            else:
                cols = list(self.columns)
        else:
            # Treat as ndarray
            X_arr = np.asarray(X)
            n_cols = 1 if X_arr.ndim == 1 else X_arr.shape[1]
            if self.columns is None:
                cols = list(range(n_cols))
            else:
                cols = list(self.columns)

        self.columns_ = cols
        self.distributions_: Dict[Union[int, str], Dict[Any, float]] = {}
        self.categories_: Dict[Union[int, str], List[Any]] = {}

        for col in self.columns_:
            if self._is_dataframe(X):
                s = X[col]
            else:
                # Ensure col is an integer index for numpy arrays
                if not isinstance(col, int):
                    raise TypeError("For numpy arrays, columns must be specified as integer indices.")
                col_idx = col
                s = X_arr[:, col_idx] if X_arr.ndim == 2 else X_arr

            s = pd.Series(s)  # unify ops
            non_missing = s[~pd.isna(s)]
            if non_missing.empty:
                raise ValueError(
                    f"Column '{col}' has all values missing; cannot learn distribution."
                )

            vc = non_missing.value_counts(normalize=True, dropna=True)
            probs = vc.to_dict()
            cats = list(vc.index)

            self.distributions_[col] = probs
            self.categories_[col] = cats

        # store RNG state to keep deterministic behavior across transforms
        self._rng_state_ = rng.get_state()
        return self

    def _largest_remainder_counts(self, probs: np.ndarray, m: int, rng: np.random.RandomState) -> np.ndarray:
        """Return integer counts that sum to m, close to probs*m."""
        if m == 0:
            return np.zeros_like(probs, dtype=int)
        exact_counts = probs * m
        base = np.floor(exact_counts).astype(int)
        remainder = exact_counts - base
        k = m - base.sum()
        if k > 0:
            # break ties randomly but reproducibly
            order = np.argsort(-remainder)  # largest first
            # tie-breaking: shuffle groups of equal remainders
            # to keep simple, add tiny noise
            noise = rng.uniform(low=0.0, high=1e-9, size=remainder.shape)
            order = np.argsort(-(remainder + noise))
            base[order[:k]] += 1
        return base

    def transform(self, X: ArrayLike) -> ArrayLike:
        check_is_fitted(self, attributes=["columns_", "distributions_", "categories_", "_rng_state_"])
        rng = check_random_state(self.random_state)
        rng.set_state(self._rng_state_)  # continue from fit's state

        # decide output container
        if self._is_dataframe(X):
            out = X.copy() if self.copy else X
        else:
            X_arr = np.asarray(X, dtype=object)  # ensure room for strings
            out = X_arr.copy() if self.copy else X_arr

        for col in self.columns_:
            if self._is_dataframe(out):
                s = out[col]
                mask = pd.isna(s)
            else:
                # Ensure col is an integer index for numpy arrays
                if not isinstance(col, int):
                    raise TypeError("For numpy arrays, columns must be specified as integer indices.")
                if out.ndim == 2:
                    s = pd.Series(out[:, col])
                else:
                    if out.ndim == 2:
                        s = pd.Series(out[:, col])
                    else:
                        s = pd.Series(np.asarray(out).ravel())
                mask = pd.isna(s)

            m = int(mask.sum())
            if m == 0:
                continue

            cats = np.array(self.categories_[col], dtype=object)
            p = np.array([self.distributions_[col][c] for c in cats], dtype=float)

            if self.exact:
                counts = self._largest_remainder_counts(p, m, rng)
                fill_values = np.repeat(cats, counts)
                rng.shuffle(fill_values)
            else:
                fill_values = rng.choice(cats, size=m, p=p)

            # write back
            if self._is_dataframe(out):
                s = s.copy()
                if isinstance(s, pd.Series):
                    s.loc[mask] = fill_values  # s is a pandas Series here, so .loc is correct
                else:
                    # s is a numpy array, use boolean indexing
                    s[mask] = fill_values
                out[col] = s
            else:
                if out.ndim == 2:
                    idx = np.where(mask)[0]
                    out[idx, col] = fill_values
                else:
                    idx = np.where(mask)[0]
                    out[idx] = fill_values

        # update stored RNG state so repeated calls keep progressing deterministically
        self._rng_state_ = rng.get_state()
        return out
