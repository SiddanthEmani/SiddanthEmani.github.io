+++
date = 2025-01-30
title = 'Linear Regression'
weight = 10
tags = ["Machine Learning", "Regression", "Linear Models", "Supervised Learning"]
+++
{{< katex >}}

## Definition

A model that fits a linear equation to the data.
$$ \hat{y}=w^Tx+\phi $$ where \\( w \\) = weight vector, \\( x \\) = feature vector, \\( \phi \\) = bias

**Residual** : \\(r_i=y_i-\hat{y_{i}}=y_{i}-(w^Tx_{i}+\phi)\\)

**Loss Function** : Mean Squared Error

**Optimization Method** : Gradient Descent

## Assumptions
- **Linearity**: Dependent and independent variables have linear relationship. Check for scatterplot trends. Curved/parabolic lines indicate non-linear relationships.
- **Independence**: Observations are independent of each other.
- **Homoscedasticity**: Constant variance of residuals.
- **Normality**: Residuals are normally distributed.
- **No Multicollinearity**: Independent variables are not highly correlated.

## Limitations
- **Non-Linear Relationships** : Cannot model non-linear relationships. 
  - Use polynomial regression or apply transformations.
- **Outliers** : Sensitive to outliers. 
  - Detect and remove outliers using IQR or Z-score analysis.
- **Categorical Features** : Does not handle categorical features directly. 
  - Use One-Hot Encoding for nominal features or Ordinal Encoding for ordered features.
- **High Multicollinearity** - Inflates variance and makes coefficients unreliable. 
  - Check VIF or remove redundant features using PCA or reduce overfitting using Ridge or Lasso Regression.
- **Heteroscedasticity** - Unequal variance will lead to uncertain predictions and biased hypothesis tests. 
  - Apply log transformations or WLS regression.
- **Non Normal Residuals** - Leads to unreliable p-values and confidence intervals. 
  - Apply Box-Cox transformation or use Quantile Regression
- **High Dimensionality** : Apply PCA or feature selection. Features with weak correlation may be removed.

## Diagnostic Decision Table

| **Observation**                       | **Implication**                                 | **Action**                                                                             |
| ------------------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------- |
| **High feature-feature correlation**  | Multicollinearity risk                          | Calculate VIF, drop/reduce features or use Ridge Regression                                                    |
| **Strong feature-target correlation** | Good predictor candidate                        | Prioritize in model. Check for data leakage                                            |
| **Non-linear patterns**               | Violates linearity assumption                   | Add polynomial terms or transformations                                                |
| **Bimodal distributions**             | Potential subgroups in data                     | Investigate data collection. Try separate models for each subgroup or use interactions |
| **Outliers in pairplot**              | Potential data errors/edge cases                | Apply robust scaling or winsorization                                                  |
| **Heteroscedasticity**                | Violates constant variance assumption           | Use log transformation or weighted regression                                          |
| **Non-normal residuals**              | Violates normality assumption                   | Apply Box-Cox transformation or increase data                                          |
| **Autocorrelation in residuals**      | Violates independence assumption                | Use Durbin-Watson test; consider lag variables                                         |
| **High leverage points**              | Can disproportionately affect model             | Detect using Cook’s Distance and consider removal                                      |
| **Omitted variable bias**             | Model missing important predictors              | Include relevant features or domain knowledge                                          |
| **Irrelevant features**               | Increases model complexity, reduces performance | Perform feature selection (LASSO, backward elimination)                                |

## Steps

**Goal**: Minimize MSE

1. Check all assumptions.
2. Split data into training and testing sets.
3. Initialize random weights for each feature.
4. Calculate predictions.
5. Calculate MSE.
6. Calculate steps using Gradient Descent
7. Evaluate using \\(R^2\\):
    - If \\(R^2\\) is low, check for missing predictor or non-linearity.
    - If \\(R^2\\) is too high, possible overfitting (Adjusted \\(R^2\\))
8. Calculate p-value for \\(R^2\\) using F-Test.
9. Plot feature coefficients, residual distribution and actual vs predicted values scatter plot.

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, f_oneway
import statsmodels.api as sm

# Load dataset
df = pd.read_csv('winequality-red.csv', sep=';')

# Define target column
target_column = 'quality'

# Step 1: Validate Assumptions Before Applying Linear Regression

# 1.1 Check for missing values
if df.isnull().sum().sum() > 0:
    print("⚠️ Dataset contains missing values. Handle them before proceeding.")

# 1.2 Check multicollinearity using Variance Inflation Factor (VIF)
X = df.drop(columns=[target_column])
y = df[target_column]

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

if vif_data["VIF"].max() > 10:
    print("⚠️ High multicollinearity detected. Consider removing/reducing features using PCA, Ridge Regression, or feature selection.")

# 1.3 Check linearity assumption using a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 1.4 Check residual normality using the Shapiro-Wilk test (valid for small samples)
_, p_value = shapiro(df[target_column])
if p_value < 0.01:
    print("⚠️ Residuals are not normally distributed. Consider log transformation or robust regression.")

# 1.5 Check homoscedasticity using ANOVA
_, p_homo = f_oneway(*[df[target_column][df[col] > df[col].median()] for col in df.drop(columns=[target_column]).columns])
if p_homo < 0.05:
    print("⚠️ Heteroscedasticity detected. Consider applying log transformation or using Weighted Least Squares (WLS).")

print("Proceeding with Linear Regression...")

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 3: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate Model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:\nRMSE: {rmse:.3f}\nR-squared (R²): {r2:.3f}")

# Step 6: Feature Importance
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance, palette='coolwarm')
plt.title("Feature Importance in Linear Regression")
plt.show()

# Step 7: Residual Analysis
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Quality")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()

# Step 8: Actual vs Predicted Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Perfect fit line
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Actual vs. Predicted Wine Quality")
plt.show()

# Step 9: Model Statistics (p-value for R²)
X_train_sm = sm.add_constant(X_train)  # Add constant for intercept
ols_model = sm.OLS(y_train, X_train_sm).fit()

print("\n📊 Model Summary:")
print(ols_model.summary())

# If R² is too high, check for overfitting
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
if adjusted_r2 < r2 - 0.05:
    print("⚠️ Possible overfitting detected. Consider cross-validation or feature selection.")
```