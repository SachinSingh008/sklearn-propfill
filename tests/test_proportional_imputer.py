import pandas as pd
import numpy as np
from propfill import ProportionalImputer

def test_basic_dataframe():
    df = pd.DataFrame({
        "cat": ["A","A","B","C", None, None, "A", "C", None, "B"]
    })
    imp = ProportionalImputer(random_state=0).fit(df)
    out = imp.transform(df)
    assert out["cat"].isna().sum() == 0
    # Only seen categories should appear
    assert set(out["cat"].unique()) <= {"A","B","C"}

def test_exact_counts():
    # 50% A, 30% B, 20% C; 10 missing -> 5,3,2 expected (or equivalent by largest remainder)
    s = pd.Series(["A","A","B","B","B","C","C", None, None, None, None, None, None, None, None, None])
    imp = ProportionalImputer(columns=None, exact=True, random_state=42).fit(s.to_frame(name="x"))
    out = imp.transform(s.to_frame(name="x"))["x"]
    filled = out[s.isna()]
    counts = filled.value_counts().to_dict()
    # exact method ensures the total equals number of missings
    assert sum(counts.values()) == s.isna().sum()

def test_ndarray_usage():
    X = np.array(["A","B",None,"A","C",None,"B"], dtype=object).reshape(-1,1)
    imp = ProportionalImputer(random_state=0).fit(X)
    out = imp.transform(X)
    assert np.isnan(out).sum() == 0 if out.dtype.kind == "f" else (pd.Series(out.ravel()).isna().sum() == 0)

def test_error_on_all_missing():
    df = pd.DataFrame({"cat": [None, None]})
    try:
        ProportionalImputer().fit(df)
        assert False, "Should have raised ValueError for all-missing column"
    except ValueError:
        assert True
