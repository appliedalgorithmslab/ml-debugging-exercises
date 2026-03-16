# Class Imbalance

## Problem

The broken example evaluates an imbalanced classification problem mainly through accuracy.

When one class is much more common than the other, a model can achieve high accuracy while performing poorly on the minority class.

## Why This Matters

In many practical machine learning settings, the minority class is often the class of greatest interest.

Examples include fraud detection, disease diagnosis, and anomaly detection.

## Correct Approach

The fixed example uses:

- class weighting to reduce bias toward the majority class
- balanced accuracy to better reflect performance across classes

## Key Takeaway

Accuracy alone can be misleading on imbalanced datasets. Always inspect class-specific metrics and consider better evaluation measures.
