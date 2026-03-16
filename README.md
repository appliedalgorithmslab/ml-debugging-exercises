
# ML Debugging Exercises

A collection of practical machine learning debugging scenarios illustrating common failure modes, incorrect evaluation practices, and corrected implementations.

## Purpose

This repository demonstrates machine learning engineering workflows for:

- identifying model failure modes
- diagnosing data leakage
- handling class imbalance
- correcting evaluation mistakes
- comparing broken and fixed pipelines

It is designed as a public demonstration repository using standard datasets and simplified workflows. Proprietary systems and internal datasets are not included.

## Exercise Areas

- Data Leakage
- Class Imbalance
- Wrong Evaluation Metric

## Repository Structure

```text
data_leakage/
    broken_pipeline.py
    fixed_pipeline.py
    explanation.md

class_imbalance/
    broken_training.py
    fixed_training.py
    explanation.md

wrong_metric/
    broken_evaluation.py
    fixed_evaluation.py
    explanation.md
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run an example:

```bash
python data_leakage/broken_pipeline.py
python data_leakage/fixed_pipeline.py
```

## Notes

This repository contains demonstration implementations intended to illustrate practical machine learning debugging patterns using public datasets.
