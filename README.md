# learning-fair-representations
Python numba implementation of Zemel et al. 2013
http://www.cs.toronto.edu/~toni/Papers/icml-final.pdf

# Implementation Data Wranging

# Learning Fair Representations (LFR) – Bias Mitigation

## 1. Project Description

This project evaluates the **Learning Fair Representations (LFR)** algorithm, introduced by Zemel et al. (2013).

LFR is a pre-processing bias mitigation algorithm. It attempts to reduce discrimination by transforming the original dataset into a new representation that preserves useful information while reducing the influence of sensitive attributes.

The goal of this project is to investigate whether LFR improves fairness while maintaining good predictive performance.

## 2. Datasets

The algorithm is evaluated on three datasets:

- **Adult Census Income:** Used to predict whether a person earns more than $50,000 per year.
- **German Credit:** Used to predict whether a person has good or bad credit.
- **COMPAS:** Used to study recidivism risk.

The experiments use the AIF360 dataset loaders.

## 3. Algorithm

The selected algorithm is Learning Fair Representations (LFR).

LFR learns a new representation of the data using a set of prototypes. It aims to balance three objectives:

- Fairness: reduce differences between protected groups.
- Reconstruction: preserve information from the original data.
- Prediction: retain information useful for predicting the outcome.

The algorithm is applied before training the final classification model.

## 4. Experimental Method

The following steps are performed for each dataset:

1. Load and prepare the dataset.
2. Calculate fairness metrics using the original labels, before training a classifier.
3. Divide the dataset into training and test sets using a stratified split.
4. Train a baseline Logistic Regression classifier on the original features.
5. Apply LFR to the training data and transform the test data using the learned representation.
6. Train a second Logistic Regression classifier using the transformed features.
7. Evaluate and compare the results before and after applying LFR.

A random seed of 42 is used to make the experiments reproducible.

## 5. Evaluation Metrics

The following metrics are used:

| Metric | Description |
|---|---|
| DP_Diff | Difference in positive prediction rates between unprivileged and privileged groups. |
| DIR | Ratio of positive prediction rates between unprivileged and privileged groups. |
| Consistency | Measures whether similar individuals receive similar predictions. |
| Accuracy | Measures the proportion of correct predictions. |
| F1-score | Measures the balance between precision and recall. |

A DP_Diff closer to zero and a DIR closer to one generally indicate better demographic parity. A higher consistency score indicates greater consistency.

Accuracy and F1-score are used to evaluate predictive performance.

## 6. Repository Structure

The project contains the following files:

- `LFR_experiments.ipynb` – Jupyter notebook containing the complete experiment.
- `LFR_results.csv` – Results of the fairness and performance evaluations.
- `LFR_results.tex` – Results table formatted for use in LaTeX and Overleaf.
- `README.md` – Project description and reproduction instructions.

## 7. Requirements

The following Python packages are required:

- AIF360
- PyTorch
- scikit-learn
- pandas
- NumPy
- matplotlib
- seaborn

Install the packages using:

```bash
pip install "aif360[LFR]" scikit-learn pandas numpy matplotlib seaborn
```

## 8. How to Reproduce the Experiments

1. Clone or download this repository.
2. Install the required Python packages.
3. Open `LFR_experiments.ipynb` in Jupyter Notebook or Visual Studio Code.
4. Run all notebook cells in order.
5. The notebook downloads and processes the datasets, applies LFR, and evaluates the results.
6. The results are saved as CSV and LaTeX files.

The experiments use a fixed random seed of 42.

## 9. References

Zemel, R., Wu, Y., Swersky, K., Pitassi, T., & Dwork, C. (2013). Learning Fair Representations. *Proceedings of the 30th International Conference on Machine Learning (ICML)*, 325–333.

AIF360 documentation: https://aif360.readthedocs.io/

Original LFR implementation: https://github.com/algofairness/fairness-comparison


