# sklearn-propfill

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-compatible-green)]()
[![PyPI](https://img.shields.io/pypi/v/sklearn-propfill.svg)]()

**ProportionalImputer** is a scikit-learn compatible transformer that fills missing categorical values so the filled values match the observed distribution.

---

## ✨ Features

- Works with **pandas DataFrames** and **NumPy arrays**  
- Integrates directly into **scikit-learn Pipelines**  
- Supports both **exact proportional filling** and **probabilistic sampling**  
- Deterministic with `random_state` for reproducibility  

---

## 📦 Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/your-username/sklearn-propfill.git
cd sklearn-propfill
pip install -e .
```

---

## 🚀 Usage Example

### Basic Example with pandas

```python
import pandas as pd
from propfill import ProportionalImputer

data = pd.DataFrame({
    "city": ["A", "B", None, "A", "C", None, "B", "A"]
})

imp = ProportionalImputer(random_state=42, exact=True)
imp.fit(data)
filled = imp.transform(data)
print(filled)
```

### In a scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from propfill import ProportionalImputer
import pandas as pd

X = pd.DataFrame({
    "city": ["A", "B", None, "A", "C", None, "B", "A"],
    "age":  [22, 30, 27, 24, 19, 28, None, 26],
})
y = [0, 1, 0, 0, 1, 1, 0, 1]

ct = ColumnTransformer([
    ("city_imp", ProportionalImputer(random_state=42), ["city"]),
], remainder="passthrough")

pipe = Pipeline([
    ("prep", ct),
    ("clf", LogisticRegression(max_iter=500))
])

pipe.fit(X, y)
```

---

## ⚖️ Exact vs Approximate Filling

The `exact` parameter controls how missing values are filled:

### `exact=True` (default)
- Ensures missing values are filled in **exact proportion** to the observed distribution.
- **Example:** if 10 values are missing and the non-missing data is 50% A, 30% B, 20% C, the imputer fills exactly 5 A, 3 B, 2 C.
- Great for **reproducibility** and **fairness**.

### `exact=False`
- Fills missing values by **random sampling** according to probabilities.
- **Same example:** each missing value is filled by drawing randomly with probabilities [0.5, 0.3, 0.2].
- The final counts may differ slightly (e.g., 6 A, 2 B, 2 C) depending on randomness.
- Useful when you want **natural variation** instead of strict counts.

---

## ✅ Testing

Run unit tests with:

```bash
pytest -q
```

---
## ✅ Example
from propfill import ProportionalImputer
import pandas as pd
import numpy as np

# Example dataset with missing values
df = pd.DataFrame({
    "category": ["A", "B", "C", "A", "B", np.nan, np.nan, np.nan, "C", "A",
                 np.nan, "B", "C", np.nan, "A", "B", np.nan, "C", "A", "B"]
})

# Make a copy so we can keep the original column
df_copy = df.copy()

# Apply ProportionalImputer
imp = ProportionalImputer()
df_copy["category_imputed"] = imp.fit_transform(df)["category"]

# Show comparison
print(df_copy.head(20))

---

## 📜 License

MIT License – feel free to use, modify, and share.