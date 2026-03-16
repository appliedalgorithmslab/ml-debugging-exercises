# Data Leakage

## Problem

The broken example scales the full dataset before performing the train/test split.

This leaks information from the test set into the training process.

## Why This Matters

When preprocessing steps such as scaling or feature selection are fit on the entire dataset, the model indirectly sees information from the test data.

This leads to overly optimistic performance estimates.

## Correct Approach

Split the data first, then fit preprocessing only on the training data.

Using a pipeline ensures that preprocessing is applied correctly during both training and evaluation.

## Key Takeaway

Always split the dataset before fitting any preprocessing steps.
