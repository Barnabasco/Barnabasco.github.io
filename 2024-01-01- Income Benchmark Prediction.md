---
layout: post
title: Income Benchmark Prediction Using Decision Trees and SVM
subtitle: Machine learning approach for income classification
author: Chimaobi Barnabas Opara
categories: machine-learning data-science economics
banner:
  image: https://bit.ly/3xTmdUP
  opacity: 0.618
  background: "#000"
  heading_style: "font-size: 4.25em; font-weight: bold; text-decoration: underline"
  subheading_style: "color: gold"
tags: decision-trees svm classification income-prediction
top: 3
sidebar: []
---

## Project Overview:
This project predicts income benchmarks (above/below $50K) using demographic features from census data. We compared Decision Tree and SVM classifiers on a dataset of 48,842 records with 14 attributes including age, education, and occupation.

## Methodology
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load and preprocess data
data = pd.read_csv('adult.csv')
X = data.drop('income', axis=1)
y = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Feature engineering
X = pd.get_dummies(X, columns=['workclass', 'education', 'marital-status', 
                               'occupation', 'relationship', 'race', 'sex', 
                               'native-country'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize models
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
svm = SVC(kernel='rbf', C=1.0, gamma='scale')

# Train and evaluate
dt.fit(X_train, y_train)
svm.fit(X_train, y_train)

```

## Performance Metrics
Model	Accuracy	Precision	Recall	F1-Score	Training Time (s)
Decision Tree	85.7%	0.79	0.65	0.71	3.2
SVM (RBF Kernel)	86.3%	0.81	0.67	0.73	28.5

Key mathematical formulation:
$ \text{Gini Index} = 1 - \sum_{i=1}^{c} (p_i)^2 $

## Feature Importance
Feature Importance Plot
Top predictors: Education level, capital gain, and work hours

# Applications
* Financial services targeting: Identify potential premium customers
* Policy impact analysis: Simulate effects of education reforms
* Economic research: Study income distribution patterns

# Project Resources:
* Jupyter Notebook
* Dataset Source
* Interactive Demo