---
layout: post
title: Yelp Sentiment Analysis
subtitle: Deep learning for review classification
author: Chimaobi Barnabas Opara
categories: nlp deep-learning sentiment-analysis
banner:
  image: https://bit.ly/3xTmdUP
  opacity: 0.618
  background: "#000"
  heading_style: "font-size: 4.25em; font-weight: bold; text-decoration: underline"
  subheading_style: "color: gold"
tags: lstm transformers bert yelp-reviews
top: 5
sidebar: []
---

## Project Overview:
This system classifies Yelp reviews into positive/negative sentiments using NLP techniques. Trained on 500,000 reviews with 1-5 star ratings, we compared traditional and transformer-based approaches.

## Model Architecture
```python
from transformers import BertForSequenceClassification
import torch

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# BERT Model
bert_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=2
)
```

## Performance Comparison

| Model             | Accuracy | F1-Score | Inference (ms) | Model Size |
|-------------------|----------|------------|----------------|-----------|
| LSTM | 97.1%    | 0.88      | 12           | 128 mb     |
| BERT-base    | 92.7%| 0.91    | 56         | 440 mb   |
| DistilBERT    | 91.8%| 0.90    | 28         | 250 mb  |

# Key metric:

$ \text{Cross-Entropy Loss} = -\sum_{c=1}^M y_c \log(p_c) $

## Error Analysis
Confusion Matrix img

Common misclassifications: Sarcastic reviews and nuanced criticisms

# Applications
* Business intelligence: Track sentiment trends over time
* Customer service: Identify urgent complaints
* Market research: Compare competitor sentiment profiles

# Project Resources:
* Model Zoo
* Live API
* Dataset
