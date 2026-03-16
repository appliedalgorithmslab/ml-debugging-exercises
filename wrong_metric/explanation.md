# Wrong Evaluation Metric

## Problem

The broken example emphasizes accuracy on an imbalanced classification problem.

Accuracy may not reflect how well the model separates the positive and negative classes, especially when the minority class is important.

## Why This Matters

In imbalanced settings, a model can achieve high accuracy by mostly predicting the majority class.

This can hide weak performance on the outcomes that matter most.

## Correct Approach

The fixed example includes:

- ROC AUC to evaluate ranking quality
- Average Precision to better reflect minority-class performance
- A classification report for class-specific precision and recall

## Key Takeaway

Evaluation metrics should match the problem. Accuracy alone is often insufficient for imbalanced classification.
