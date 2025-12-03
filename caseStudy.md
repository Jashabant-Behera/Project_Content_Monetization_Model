***

# Project Case Study and Concept Guide: YouTube Content Monetization Model

This file is only for learning and can be deleted later. It explains the problem, the pipeline, and every major concept with code-style examples based on this specific project.

***

## 1. Problem statement

- **Goal:** Predict the **Ad Revenue (USD)** a YouTube video will generate based on its performance metrics and metadata.
- **Type:** Regression (predicting a continuous number).
- **Input Features:**
  - **Metrics:** Views, Likes, Comments, Watch Time (minutes), Video Length, Subscribers.
  - **Metadata:** Category (Gaming, Music, etc.), Device (Mobile, TV, etc.), Country.
- **Output:** `ad_revenue_usd`.
- **Business Context:** This helps content creators estimate potential earnings for their videos and understand which factors (like Watch Time or Engagement) drive the most revenue.

***

## 2. Project structure and files

- `src/Data/raw/`: Contains the original `youtube_ad_revenue_dataset.csv`.
- `src/Data/processed/`: Contains `youtube_ad_revenue_processed.csv` (cleaned, imputed, and encoded).
- `src/notebooks/`:
  - `EDA.ipynb`: Exploratory Data Analysis (finding correlations, outliers).
  - `Preprocessing.ipynb`: Cleaning, imputation, feature engineering, and encoding.
  - `Model_Building.ipynb`: Scaling, training multiple models, and saving the best one.
- `src/models/`:
  - `best_model.pkl`: The trained Ridge Regression model.
  - `scaler.pkl`: The StandardScaler object used to normalize inputs.
  - `feature_names.pkl`: List of columns to ensure the app sends data in the correct order.
- `app.py`: A Streamlit web application for end-users to make predictions.
- `caseStudy.md`: This learning file.

***

## 3. Data loading and inspection

### 3.1 Importing libraries

```python
import pandas as pd       # For handling tabular data (DataFrames)
import numpy as np        # For numerical calculations
import matplotlib.pyplot as plt  # For basic plotting
import seaborn as sns     # For advanced statistical plots (Heatmaps)
from sklearn.model_selection import train_test_split  # To split data for validation
```

**Why these libraries?**
- **Pandas**: The standard tool for data manipulation in Python. It lets us load CSVs and treat them like SQL tables or Excel sheets.
- **Seaborn**: Used in `EDA.ipynb` to create the correlation heatmap, which was crucial for finding that `watch_time_minutes` is highly correlated with revenue.

### 3.2 Reading the dataset

```python
df = pd.read_csv("src/Data/raw/youtube_ad_revenue_dataset.csv")
df.head()  # View first 5 rows
df.info()  # Check data types and missing values
df.describe() # Statistical summary (mean, min, max)
```

**Key Observation:** `df.info()` revealed missing values in `likes`, `comments`, and `watch_time_minutes`, which we had to fix in the preprocessing step.

***

## 4. Data cleaning and preprocessing

### 4.1 Handling missing values

```python
# From Preprocessing.ipynb
numeric_cols_with_nan = ['likes', 'comments', 'watch_time_minutes']
for col in numeric_cols_with_nan:
    df[col] = df[col].fillna(df[col].median())
```

**Explanation:**
- We used **Median Imputation**.
- **Why?** The median is the middle value. If a viral video has 10M likes (outlier), the *mean* (average) would be skewed high. The median is robust to these outliers, making it a safer guess for missing data.

### 4.2 Feature Engineering

```python
# Creating a new meaningful feature
df['engagement_rate'] = (df['likes'] + df['comments']) / df['views']
```

**Explanation:**
- Raw numbers (e.g., 100 likes) aren't as useful as rates. 100 likes on 100 views is amazing; 100 likes on 1M views is terrible. Creating `engagement_rate` gives the model a better signal of video quality.

### 4.3 Encoding categorical variables

```python
# One-Hot Encoding
model_df = pd.get_dummies(model_df, columns=['category', 'device', 'country'], drop_first=True)
```

**Explanation:**
- Machine learning models only understand numbers.
- **One-Hot Encoding** turns "Category: Gaming" into a column `category_Gaming` with a `1` (True) or `0` (False).
- `drop_first=True` removes the first category to prevent multicollinearity (dummy variable trap).

### 4.4 Feature scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Explanation:**
- **Why?** `Views` can be in the millions, while `Video Length` is usually < 20. Models like Linear/Ridge Regression will think `Views` is more important just because the number is bigger.
- **StandardScaler** forces all features to have a mean of 0 and variance of 1, putting them on a level playing field.

***

## 5. Train–test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

**Explanation:**
- We hide 20% of the data (`X_test`) to evaluate the model later. This simulates "new" data the model hasn't seen, ensuring we aren't just memorizing the training set (overfitting).

***

## 6. Model training

### 6.1 Choosing a model

We tested multiple models, but **Ridge Regression** performed best.

```python
from sklearn.linear_model import Ridge

model = Ridge()
model.fit(X_train, y_train)
```

**Explanation:**
- **Ridge Regression** is Linear Regression with L2 Regularization.
- **Why it won:** We found a high correlation (0.99) between `watch_time` and `revenue`. Standard Linear Regression can become unstable with highly correlated features (multicollinearity). Ridge adds a penalty to the coefficients, shrinking them to prevent overfitting and handling correlation better.

***

## 7. Model evaluation

```python
from sklearn.metrics import r2_score, mean_squared_error

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
```

**Explanation:**
- **R2 Score (~0.95):** Means our model explains 95% of the variance in ad revenue. This is an excellent score.
- **RMSE (Root Mean Squared Error):** Tells us, on average, how many dollars our prediction is off by.

***

## 8. Saving artifacts (Pickle/Joblib)

### 8.1 Why save multiple files?

In `Model_Building.ipynb`, we saved three files:

```python
import joblib

# 1. The Model (The Brain)
joblib.dump(best_model, 'src/models/best_model.pkl')

# 2. The Scaler (The Translator)
joblib.dump(scaler, 'src/models/scaler.pkl')

# 3. Feature Names (The Map)
joblib.dump(X.columns.tolist(), 'src/models/feature_names.pkl')
```

**Explanation:**
- **Model:** Stores the learned weights.
- **Scaler:** Crucial! If we train on scaled data (0 to 1), we MUST scale the new data in the app the exact same way. If we pass raw views (e.g., 10,000) to the model, it will crash or give wrong answers.
- **Feature Names:** Ensures the app sends columns in the exact order the model expects (e.g., `[Views, Likes]` vs `[Likes, Views]`).

### 8.2 Loading in `app.py`

```python
model = joblib.load('src/models/best_model.pkl')
scaler = joblib.load('src/models/scaler.pkl')
```

**Joblib vs Pickle:**
- We used `joblib` because it is more efficient for objects containing large numpy arrays (like sklearn models) compared to Python's built-in `pickle`.

***

## 9. Example “cell-by-cell” explanation

**From `EDA.ipynb`:**

```python
# Correlation Matrix
sns.heatmap(df.corr(), annot=True)
```
**Explanation:** This visual showed us that `ad_revenue_usd` and `watch_time_minutes` had a correlation of nearly 1.0. This insight told us that Watch Time is the single most important predictor, guiding our feature selection and model choice (Ridge).

**From `Model_Building.ipynb`:**

```python
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor()
}
```
**Explanation:** We defined a dictionary of models to loop through. This allows us to quickly "spot check" algorithms. While Random Forest is powerful, Ridge was faster and just as accurate for this linear-style problem.

***

## 10. Suggested mini-exercises

1.  **Try a different model:** Import `GradientBoostingRegressor` from `sklearn.ensemble` and see if it beats Ridge.
2.  **Feature selection:** Try dropping `views` (since it's correlated with watch time) and see if the model performs better or worse.
3.  **App improvement:** Add a "Profit Calculator" to `app.py` that subtracts estimated production costs from the predicted revenue.
4.  **Robustness:** In `Preprocessing.ipynb`, try using `KNNImputer` instead of median imputation for missing values.

***
