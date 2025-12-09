# Zero-Code Data Science Acceleration Project

This project provides a hands-on demonstration of how to accelerate a common data science workflow using popular Python libraries with minimal to zero code changes.

It benchmarks a workflow consisting of data loading, data processing (`groupby`), and machine learning model training (`RandomForestClassifier`) across four different environments:
1.  **Baseline:** Standard Pandas and Scikit-learn on a single CPU core.
2.  **Modin:** Multi-core CPU acceleration for Pandas operations.
3.  **Intel® Extension for Scikit-learn:** CPU acceleration for Scikit-learn algorithms.
4.  **NVIDIA RAPIDS:** End-to-end GPU acceleration for the entire workflow.

