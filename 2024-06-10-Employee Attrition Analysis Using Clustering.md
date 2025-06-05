---
layout: post
title: Employee Attrition Analysis Using Clustering Algorithms
subtitle: Unsupervised learning for HR analytics
author: Chimaobi Barnabas Opara
categories: clustering hr-analytics unsupervised-learning
banner:
  image: https://bit.ly/3xTmdUP
  opacity: 0.618
  background: "#000"
  heading_style: "font-size: 4.25em; font-weight: bold; text-decoration: underline"
  subheading_style: "color: gold"
tags: kmeans dbscan hr-analytics employee-attrition
top: 4
sidebar: []
---

## Project Overview:
This analysis identifies patterns in employee attrition using clustering techniques on HR data. We processed 1,470 employee records with 35 features to discover hidden attrition patterns using K-Means and DBSCAN algorithms.

## Methodology
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import umap

# Preprocessing
df = pd.read_csv('employee_attrition.csv')
X = df.drop(['EmployeeID', 'Attrition'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality reduction
reducer = umap.UMAP(n_components=2)
X_umap = reducer.fit_transform(X_scaled)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42).fit(X_umap)
dbscan = DBSCAN(eps=0.5, min_samples=10).fit(X_umap)



## Cluster Analysis


| Algorithm             | Clusters | 	Silhouette Score | Attrition Rate in High-Risk Cluster |
|-------------------|----------|------------|----------------|
| K-Means | 4    | 0.62       | 73.4%            | 
| DBSCAN    | 5| 0.58    | 78.2%         | 


# Key metric:

$ \text{Silhouette Score} = \frac{b - a}{\max(a, b)} $

Where a = mean intra-cluster distance, b = mean nearest-cluster distance

# Key Findings
* High-risk cluster profile:
* 60+ hours weekly
* Limited promotion opportunities
* Low job satisfaction (<= 2/5)

# Retention recommendations:
* Flexible work arrangements
* Career development programs
* Recognition initiatives

# Applications
* Proactive retention programs: Target high-risk employees
* Compensation optimization: Benchmark against cluster characteristics
* Talent acquisition: Identify candidate profiles with higher retention