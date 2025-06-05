---
layout: post
title: Deep Learning-Driven Comparative Analysis of Pre-trained CNNs and Transformers
subtitle: Benchmarking architectures for respiratory disease detection in medical imaging
author: Chimaobi Barnabas Opara
categories: deep-learning computer-vision medical-imaging
banner:
  video: https://vjs.zencdn.net/v/oceans.mp4
  loop: true
  volume: 0.8
  start_at: 8.5
  image: https://bit.ly/3xTmdUP
  opacity: 0.618
  background: "#000"
  height: "100vh"
  min_height: "38vh"
  heading_style: "font-size: 4.25em; font-weight: bold; text-decoration: underline"
  subheading_style: "color: gold"
tags: cnn transformer xray ct-scan covid-19 pneumonia
top: 1
sidebar: []
---

This project presents a comprehensive comparison of pre-trained convolutional neural networks (CNNs) and transformer architectures for detecting respiratory diseases from chest X-rays and CT scans. We evaluated models including ResNet, EfficientNet, ViT, and Swin Transformers on multi-source datasets containing COVID-19, pneumonia, and normal cases.

## Methodology

Implemented a dual-path framework to process both X-ray and CT scan images using PyTorch. Key technical components include:


```python
from torchvision import models
from transformers import ViTForImageClassification

# CNN model initialization
resnet = models.resnet50(pretrained=True)
efficientnet = models.efficientnet_b7(pretrained=True)

# Transformer model initialization
vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
swin = SwinTransformer(hidden_dim=128, layers={2, 2, 18, 2}, heads={4, 8, 16, 32})

# Custom fusion module for multi-modal inputs
class FusionModel(nn.Module):
    def __init__(self, cnn_backbone, transformer_backbone):
        super().__init__()
        self.cnn = cnn_backbone
        self.transformer = transformer_backbone
        self.classifier = nn.Linear(2048 + 768, 3)  # 3 disease classes
        
    def forward(self, x_img, x_ct):
        cnn_features = self.cnn(x_img)
        trans_features = self.transformer(x_ct)
        fused = torch.cat((cnn_features, trans_features), dim=1)
        return self.classifier(fused)
```

## Experimental Results

Our evaluation on 15,000+ medical images revealed significant findings:

| Architecture       | X-ray Accuracy | CT Scan Accuracy | Params (M) | Inference (ms) |
|--------------------|----------------|------------------|------------|----------------|
| ResNet-50          | 92.3%          | 89.7%            | 25.6       | 42             |
| EfficientNet-B7    | 94.1%          | 91.5%            | 66.3       | 68             |
| ViT-Base           | 93.7%          | 94.2%            | 86.6       | 85             |
| Swin-Large         | **96.2%**      | **95.8%**        | 197.1      | 127            |

Key mathematical formulations used in our analysis:

$ \text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $


## Implementation Highlights


```python

# Multi-modal data augmentation pipeline
transform = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Gradient accumulation for large transformers
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
for i, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i+1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

```

## Conclusion and Resources
Transformers consistently outperformed CNNs on CT scans (+3.8% accuracy), while CNNs showed faster inference times (2-3× speedup). The Swin Transformer achieved state-of-the-art results but required extensive computational resources.


# GitHub Repository
