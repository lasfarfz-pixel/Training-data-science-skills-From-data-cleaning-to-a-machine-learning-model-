# Applied Predictive Analytics Pipeline: From Preprocessing to Non-Linear Ensemble Modeling

## Project Overview
This repository implements a production-grade machine learning and data engineering workflow across two distinct analytical tracks:
1. **Classification Preprocessing & Impurity Analysis:** A robust data cleaning and exploratory data analysis (EDA) pipeline designed to handle structural irregularities, assess multi-collinearity, and compute baseline information-theoretic metrics (Gini Impurity, Shannon Entropy).
2. **Non-Linear Regression Benchmarking:** An evaluation pipeline comparing linear, regularized linear, and tree-based ensemble models on highly non-linear feature spaces to predict a continuous target variable.

---

## Key Capabilities & Workflow
* **Data Sanitization:** Implemented automated duplicate filtering, targeted missing-value imputation schemas based on downstream task requirements, and statistical outlier identification using the Interquartile Range (IQR) method.
* **Feature Engineering & Co-dependence Mapping:** Leveraged Pearson correlation matrices and skewness metrics to diagnose distributional asymmetries and multi-collinearity.
* **Information Theory Baselines:** Modeled data uncertainty prior to decision-tree execution by computing deterministic class probabilities and baseline misclassification boundaries.
* **Ensemble Modeling & Cross-Validation:** Trained and optimized Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting models backed by cross-validation matrices.

---

## Phase 1: Data Engineering, EDA, and Classification Metrics

### Data Engineering Diagnostics
The initial classification dataset contained 1,375 records across 5 attributes (`X1`, `X2`, `X3`, `X4`, `Label`). A deterministic sanitization pipeline was deployed to resolve structural anomalies:

| Identified Data Defect | Diagnostic Evidence | Remediation Action |
| :--- | :--- | :--- |
| Duplicate Records | 25 duplicate rows identified | Dropped permanently; dataset reduced to 1,350 records |
| Missing/Null Fields | 1 null value in the target `Label` column | Retained for global feature EDA; isolated and removed for metric modeling ($n = 1,349$) |
| Asymmetrical Metrics | Severe skewness detected in `X4` versus `X1` | Isolated for distribution profiling |
| Statistical Outliers | IQR thresholds flag minor variances | Reported and retained to preserve edge-case variance |

### Statistical Insights & Correlation Analysis
* **Distribution Profiles:** A comparative analysis between features `X1` and `X4` revealed that while `X1` demonstrates a broader dispersion (higher standard deviation and variance), `X4` displays high structural skewness and asymmetry. This variance makes `X4` highly sensitive to scale-dependent algorithms.
* **Linear Co-dependence:** Strong inverse relationships were detected between `X2` and both `X3` ($r \approx -0.80$) and `X4` ($r \approx -0.66$).
* **Predictive Value:** `X1` and `X2` demonstrated strong-to-moderate linear correlations with the target variable (`Label`), scoring approximately $-0.60$ and $-0.41$ respectively, identifying them as prime candidate predictors over the weak variance of `X4`.

### Information Theory & Class Impurity Indexing
Following target array validation ($n = 1,349$), class counts were extracted (Class 0: 739, Class 1: 610) to compute data mixing indices:

* **Class Distribution:** Class 0 = **54.78%**, Class 1 = **45.22%** (demonstrating a highly functional, relatively balanced dataset).
* **Impurities & Uncertainty:** Due to the near-equal class distribution, Gini Impurity and Shannon Entropy scores sat close to their theoretical maximums, indicating substantial uncertainty prior to feature partitioning.
* **Baseline Misclassification Boundaries:** Established a rigid deterministic prediction floor of **45.22%**. Any production-grade classification model must significantly outperform this maximum-voting baseline to be viable.

---

## Phase 2: Non-Linear Regression Benchmarking

Phase 2 focused on predicting a continuous outcome variable, $f(t)$, using a singular predictor, $t$, across 1,000 clean observations. While the global correlation coefficient was strongly negative ($r = -0.75$), empirical scatter mapping exposed an explicitly non-linear, curved trajectory. 

To map this complex curvature, traditional linear baselines were benchmarked against non-linear architectures.

### Performance Metrics Evaluation Matrix
Models were rigorously validated against Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$).

| Model Architecture | Test MAE | Test RMSE | Test $R^2$ | Cross-Validation Mean $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | *Baseline* | *Baseline* | *Poor* | *Inadequate* |
| **Ridge Regression** | *Baseline* | *Baseline* | *Poor* | *Inadequate* |
| **Gradient Boosting Regressor** | *Low* | *Low* | *High* | *High* |
| **Random Forest Regressor** | **1.09** | **3.69** | **0.999947** | **0.999971** |

### Execution Takeaways
* **Parametric Failure:** Simple Linear and Regularized Ridge models failed to adequately minimize variance due to their rigid mathematical assumptions over a curved, non-linear feature path.
* **Ensemble Success:** The **Random Forest Regressor** delivered exceptional accuracy, capturing $99.99\%$ of the target variance. This performance was validated via out-of-sample Cross-Validation (Mean $RMSE = 2.56$, Mean $R^2 = 0.999971$).
* **Residual Analysis:** Diagnostic residual profiling confirmed errors were normally distributed and tightly centered around zero, proving that the model successfully extracted signal from noise without systematic under- or over-prediction bias.

---

## Future Development Scale-Up
1. **Pipeline Scalability:** Refactor existing code into modular, parameterized object-oriented scripts utilizing `scikit-learn` Pipelines.
2. **Hyperparameter Tuning:** Introduce automated grid and randomized search spaces to optimize tree depths and ensemble estimators.
3. **Advanced Architectures:** Deploy Support Vector Regression (SVR) with radial basis function kernels to evaluate alternative non-linear boundaries.
