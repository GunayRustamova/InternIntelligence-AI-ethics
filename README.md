# AI Bias Detection and Fairness Evaluation

## Overview

This project evaluates gender bias in an income prediction model and tests two different techniques to reduce that bias. Using the Adult Income dataset, I trained a baseline model and then applied bias mitigation methods to see how much we can improve fairness without sacrificing too much accuracy.

## The Problem

Machine learning models learn from historical data, which often contains biases. When we build models to predict things like income, they can end up treating different demographic groups unfairly. This project investigates how bad the bias is and what we can do about it.

## Dataset

Adult Income Dataset - 32,561 people with information about their age, education, occupation, etc. The goal is to predict whether someone earns more than $50K per year. I focused on gender as the protected attribute.

## What I Built

**6 phases:**

1. Train a baseline logistic regression model
2. Measure bias using fairness metrics (statistical parity, equal opportunity, equalized odds)
3. Apply reweighing technique to reduce bias
4. Apply Fairlearn's exponentiated gradient method
5. Compare all three models
6. Document findings and recommendations

## Results

The baseline model had significant gender bias - 17.67% statistical parity difference. After applying mitigation:

- **Reweighing**: Reduced bias by 68% with only 1.4% accuracy drop
- **Fairlearn**: Reduced bias by 97% with 2.2% accuracy drop

Reweighing gives the best balance between fairness and performance.

## Tools Used

- AIF360 and Fairlearn for bias detection and mitigation
- scikit-learn for the ML models
- pandas/numpy for data processing
- matplotlib for visualizations

## Outputs

- Detailed console output with all metrics
- Comparison chart showing accuracy vs fairness tradeoffs
- Text report with findings and recommendations

## Why This Matters

Income prediction models affect real decisions about loans, jobs, and opportunities. Biased models can systematically disadvantage certain groups and perpetuate inequality. This work shows we can build fairer models without giving up much accuracy.

## How to Run

Open the provided notebook in **Google Colab**, follow the setup instructions, and run all cells sequentially.  
All required dependencies are installed automatically within the notebook.
