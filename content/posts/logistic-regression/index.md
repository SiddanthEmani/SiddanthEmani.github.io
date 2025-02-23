+++
date = 2025-02-14
title = 'Logistic Regression'
weight = 10
tags = ["Machine Learning", "Classification", "Linear Models", "Supervised Learning"]
+++
{{< katex >}}

{{< chart >}}
type: 'scatter',
data: (function(){
  let class0 = [];
  let class1 = [];
  let logisticCurve = [];
  const numPoints = 100;
  
  // Simulate binary classification data:
  // Predictor x is uniformly drawn from 0 to 6.
  // Logistic function: p = 1 / (1 + exp(-(x - 3)))
  for (let i = 0; i < numPoints; i++) {
    let x = Math.random() * 6;
    let p = 1/(1 + Math.exp(-(x - 3)));
    // Assign class label based on probability p:
    let outcome = Math.random() < p ? 1 : 0;
    if (outcome === 1) {
      class1.push({ x: x, y: outcome });
    } else {
      class0.push({ x: x, y: outcome });
    }
  }
  
  // Generate logistic regression curve data:
  // Create 50 points evenly spaced over x in [0,6].
  const numCurvePoints = 50;
  for (let i = 0; i < numCurvePoints; i++) {
    let x = (6 / (numCurvePoints - 1)) * i;
    let p = 1/(1 + Math.exp(-(x - 3)));
    logisticCurve.push({ x: x, y: p });
  }
  
  return {
    datasets: [
      {
        label: 'Class 0',
        data: class0,
        backgroundColor: 'rgba(54, 162, 235, 1)', // Blue for Class 0
        pointRadius: 4
      },
      {
        label: 'Class 1',
        data: class1,
        backgroundColor: 'rgba(255, 99, 132, 1)', // Red for Class 1
        pointRadius: 4
      },
      {
        label: 'Logistic Regression Curve',
        type: 'line',
        data: logisticCurve,
        borderColor: 'rgba(255, 206, 86, 1)', // Yellowish line
        fill: false,
        pointRadius: 0,
        tension: 0,
        borderDash: [5, 5]
      }
    ]
  };
})(),
options: {
  plugins: {
    legend: {
      labels: {
        color: 'white'
      }
    }
  },
  scales: {
    x: {
      type: 'linear',
      position: 'bottom',
      title: {
        display: true,
        text: 'Predictor (x)',
        color: 'white'
      },
      ticks: {
        color: 'white'
      },
      grid: {
        color: 'rgba(255, 255, 255, 0.1)'
      },
      min: 0,
      max: 6
    },
    y: {
      type: 'linear',
      title: {
        display: true,
        text: 'Probability / Class Label',
        color: 'white'
      },
      ticks: {
        color: 'white'
      },
      grid: {
        color: 'rgba(255, 255, 255, 0.1)'
      },
      min: -0.1,
      max: 1.1
    }
  }
}
{{< /chart >}}

A classification technique which computes the probability or likelihood that the data belongs to a class of interest.
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n$$

**Goal** : Maximize likelihood

**Optimization Method** : Gradient Descent

Generally fits an ‘S’ curve (‘S’ shaped logistic function) for the data to belong to separate classes. 

## Binary Classification

Sigmoid transforms the linear combination of inputs into probability.
$$\sigma(z) = \frac{1}{1+e^{-z}}$$
-  \\(z = \beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n\\)

The curve which gives the maximum likelihood is selected as the end result.

<img src="img/Logistic Regression - Binary Classification.png" alt="Logistic Regression - Binary Classification" style="display: block; margin: 0 auto;">

## Multinomial Logistic Regression
    
**One-vs-Rest (OvR) Classification** :
- For \\(K\\) classes, train \\(K\\) separate binary classifiers (one class vs the rest).
- For a new observation, output \\(K\\) probabilities and select the class with the highest probability.
- Different binary classifiers may produce overlapping decision regions which may lead to ambiguous classification, especially when classes are imbalanced.

<br>

**Softmax Logistic Function** :

Softmax ensures the predicted probabilities of all classes sum to 1.
$$P(y=k|x)=\frac{e^{x^T\beta_{k}}}{\sum_{j=1}^Ke^{x^T\beta_{j}}}$$
- \\(x\\) = feature vector
- \\(K\\) = number of classes
- \\(\beta_{k}\\) = parameter vector for class \\(k\\)
- Provides a coherent probability model.
- Model coefficients can be interpreted as the change in log-odds of being in a given class.
<br>

## Maximum Likelihood Estimation

**Binary Classification** : 
$$L(\beta) = \prod_{i=1}^{N} p_i^{y_i} (1-p_i)^{1-y_i}$$
- \\(p_i\\) = predicted probability for observation \(\phi\) 
- \\(y_i\\) = actual label (0 or 1)  

**Log Transformation (Log-Likelihood)** :
- This results in underflow for large datasets since all values are between 0 and 1.
- We apply log transformation to change it into a summation.
$$\ell(\beta)=\sum_{i=1}^{N}(y_i \log(p_i) + (1-y_i) \log(1-p_i))$$

**Multi-Classification using Softmax** :
$$\ell(\{\beta_{k}\}) = \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log\left( \frac{e^{x_i^\top \beta_{k}}}{\sum_{j=1}^{K} e^{x_i^\top \beta_j}} \right)$$


## Assumptions
- An S shaped curve is assumed to fit the data. The independent variable should have strong correlation with the likelihood of the class.
- **Linearity in the Logit** : The log-odds (logits) of the outcome are assumed to be a linear combination of the independent variables.
    $$\log\left( \frac{p}{1-p} \right)=\beta_{0}+\beta_{1}x_{1}+\dots+\beta_{n}x_{n}$$
    <img src="img/Logit graphs.png" alt="Logistic Regression - Linearity in the Logit" style="display: block; margin: 0 auto;">
    - The coefficients are in the log-odds scale. All [Linear Regression](/posts/linear-regression) tests can be done on this line.
- **Independence** : Observations are assumed to be independent of each other.
- **No or Little Multicollinearity** : The independent variables should not be highly correlated with each other.
- A sufficiently large sample is preferred to obtain reliable estimates.

## Limitations

- **Linear Decision Boundary in Logit Space** : Assuming a linear relationship between the predictors and the log-odds of the outcome, might not capture more complex, non-linear relationships.
- **Sensitivity to Outliers** : Extreme values can disproportionately affect the model.
- **Multicollinearity** : High correlation among independent variables can destabilize coefficient estimates.
- **Feature transformations** : For data that is not linearly separable (in the logit domain), performance may suffer unless features are transformed or interaction terms are added.

## Diagnostic Decision Table
| **Observation**                                                                                                                                   | **Implication**                                                                                                                               | **Action**                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Poor Overall Model Fit** : High deviance, low pseudo R-squared, or significant lack-of-fit (e.g., failing the Hosmer-Lemeshow test).             | The model may be mis-specified, missing important predictors, or not capturing the true relationship between variables.                       | Reassess model specification: consider adding relevant predictors, interaction terms, or non-linear transformations.                                                                          |
| **Non-significant Coefficients** : Predictors with high p-values or wide confidence intervals.                                                     | These predictors may not contribute meaningfully to predicting the outcome, possibly diluting the model’s effectiveness.                      | Remove or re-specify insignificant variables, and ensure that only meaningful predictors remain in the model.                                                                                 |
| **Low Predictive Performance** : Low classification accuracy, poor ROC-AUC, or imbalanced confusion matrix (e.g., high false positives/negatives). | The model might be underfitting, failing to capture important data patterns, or struggling with class imbalance.                              | Perform feature engineering, consider re-balancing the dataset, explore alternative model specifications, or try regularization techniques.                                                   |
| **High Multicollinearity** : Indicators such as high Variance Inflation Factor (VIF) values or inflated standard errors.                           | Predictor variables are highly correlated, leading to unstable coefficient estimates and difficulty in interpreting their individual effects. | Remove, combine, or transform correlated predictors, or apply regularization (L1/L2) to mitigate multicollinearity issues.                                                                    |
| **Influential Observations** : A few data points exhibit high Cook’s distance or leverage values.                                                  | These observations may disproportionately affect model estimates, potentially skewing results and interpretations.                            | Investigate and validate these outliers. Consider robust regression techniques or, if justified, remove problematic observations after careful analysis.                                      |
| **Patterned Residuals** : Residual plots show systematic patterns or non-random distribution.                                                      | This suggests that the model is missing key relationships (e.g., interactions or non-linear effects), indicating mis-specification.           | Consider adding interaction terms, polynomial terms, or applying variable transformations to better capture the underlying data structure.                                                    |
| **Poor Calibration in Multi-Class Models (OvR approach)** : Inconsistent or poorly calibrated predicted probabilities across classes.              | The individual binary classifiers may not provide directly comparable probabilities, potentially leading to ambiguous class assignments.      | For multi-class problems, consider switching to multinomial (softmax) logistic regression, applying probability calibration techniques, or using methods to better balance class predictions. |

## Steps
**Goal** : Maximize likelihood

**Model** : \\(\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n\\)

1. Compute the likelihood for each observation.
2. Apply the log transformation to obtain the log-likelihood.
3. Optimize the parameters using Gradient Descent.
4. Calculate evaluation metrics and diagnose.
5. Use Wald’s Test to determine whether a feature is useful in computing the prediction.
6. Validate model assumptions such as linearity and independence.