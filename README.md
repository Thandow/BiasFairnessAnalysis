# BiasFairnessAnalysis
# ⚖️ Recidivism Bias & Fairness Analysis (Responsible AI)

> An end-to-end fairness, bias mitigation, and ethics evaluation of a criminal-justice machine learning system

---

## 📌 Project Overview

This project analyzes **algorithmic bias, fairness metrics, and ethical risks** in a machine learning model trained on the **Recidivism__Beginning_2008.csv** dataset.  
It demonstrates how predictive systems in **high-stakes domains** such as criminal justice can unintentionally reinforce inequality, and how these risks can be **measured, mitigated, and governed responsibly**.

The project follows a **Responsible AI pipeline**, combining technical evaluation with ethical decision-making.

---

## 🎯 Objectives

- Select and analyze a **potentially biased real-world dataset**
- Identify **protected attributes** and affected demographic groups
- Implement **three or more quantitative fairness metrics**
- Visualize and statistically analyze bias patterns
- Apply **two or more bias mitigation techniques**
- Compare model performance **before and after mitigation**
- Develop a **use-case-specific ethics framework**
- Connect findings to **real-world harms and implications**

---

## 📂 Dataset

**File:** `Recidivism__Beginning_2008.csv`  
**Domain:** Criminal Justice  
**Context:** Post-release outcomes for individuals released beginning in 2008

### 🔑 Variables

| Category | Column |
|--------|--------|
| Features | Age at Release, County of Indictment, Release Year |
| Target Label | Return Status (Returned / Not Returned) |
| Protected Attribute | Gender |

> ⚠️ Gender is **not used for model training** and is used **only for fairness evaluation**.

---

## 🧠 Methodology

### 1. Data Preprocessing
- Missing value handling
- Encoding categorical features
- Binary transformation of target variable

### 2. Model Training
- Baseline model: **Logistic Regression**
- Consistent train/test split

### 3. Fairness Metrics Implemented

| Metric | Description |
|------|------------|
| Demographic Parity Difference | Outcome rate disparity |
| Equal Opportunity Difference | True positive rate disparity |
| Disparate Impact Ratio | Proportional fairness |
| False Positive Rate Gap | Over-supervision harm |
| False Negative Rate Gap | Under-intervention harm |

---

## 🔍 Bias Identification

Key findings:
- Gender-based disparities in predictions
- Unequal error distributions across groups
- Geographic variables acting as **proxy features**

**Most affected group:**  
- Female individuals

---

## 🛠 Bias Mitigation Techniques

The following techniques were applied:

1. **Reweighing**
   - Adjusts sample weights to balance protected groups
2. **Threshold Adjustment**
   - Modifies decision thresholds to reduce outcome disparities

---

## 📊 Performance Comparison

| Aspect | Before Mitigation | After Mitigation |
|------|------------------|------------------|
| Accuracy | Higher | Slightly Lower |
| Fairness Metrics | Violations Present | Improved |
| Harm Disparities | Unequal | Reduced |

> This highlights the **fairness–accuracy trade-off**, which is ethically justified in high-risk systems.

---

## 🧭 Ethics Framework

A **custom ethics framework** was developed specifically for recidivism prediction systems.

### Ethical Principles
- Fairness
- Non-maleficence
- Accountability
- Transparency
- Human oversight

### Framework Features
- Quantitative ethical thresholds
- Automated ethics evaluation
- Deployment approval logic
- Human-in-the-loop enforcement
- Continuous ethics monitoring

Ethical principles are **operationalized in code**, ensuring enforceability rather than abstract guidance.

---

## 🌍 Real-World Implications

- **False positives** may lead to excessive supervision or restricted freedom
- **False negatives** may deny access to rehabilitation programs
- Algorithmic bias can reinforce systemic inequality
- Ethical failures can undermine trust in justice systems

---

## 📁 Project Structure

---

## 🚀 How to Run (Google Colab)

```python
from google.colab import files
files.upload()

import pandas as pd
df = pd.read_csv("/content/Recidivism__Beginning_2008.csv")
