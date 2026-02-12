# Machine Learning Assignment 2
**Course:** Machine Learning  
**Program:** M.Tech (AIML / DSE)  
**Institution:** BITS Pilani – Work Integrated Learning Programmes  

**BITS ID:** 2025AA05729
**Name:** ASHMIT BHANDARI


---

## 1. Problem Statement

The objective of this assignment is to implement multiple machine learning classification models on a real-world dataset and deploy them using an interactive Streamlit web application. The application enables users to select different models, upload test data, and evaluate model performance using standard classification metrics.

---

## 2. Dataset Description

- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository (accessed via scikit-learn)
- **Problem Type:** Binary Classification
- **Number of Instances:** 569
- **Number of Features:** 30 numeric features
- **Target Variable:**
  - `0` → Malignant
  - `1` → Benign

The dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast masses. Since all features are numerical, minimal preprocessing is required.

---

## 3. Machine Learning Models Used

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

---

## 4. Model Evaluation Metrics

Each model was evaluated using the following metrics:

- Accuracy  
- AUC (Area Under the ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

### 4.1 Comparison of Model Performance

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.98 | 0.995 | 0.98 | 0.99 | 0.98 | 0.96 |
| Decision Tree | 0.95 | 0.94 | 0.95 | 0.95 | 0.95 | 0.90 |
| KNN | 0.97 | 0.98 | 0.97 | 0.98 | 0.97 | 0.94 |
| Naive Bayes | 0.96 | 0.97 | 0.96 | 0.97 | 0.96 | 0.92 |
| Random Forest | 0.98 | 0.996 | 0.98 | 0.99 | 0.98 | 0.96 |
| XGBoost | 0.99 | 0.998 | 0.99 | 0.99 | 0.99 | 0.97 |

*(Metric values may vary slightly depending on the random train-test split.)*

---

## 5. Observations on Model Performance

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Performs very well due to effective feature scaling and near-linear separability of the dataset. |
| Decision Tree | Provides good interpretability but may overfit without proper depth control. |
| KNN | Shows strong performance but is sensitive to feature scaling and choice of k value. |
| Naive Bayes | Performs reasonably well despite its assumption of feature independence. |
| Random Forest | Achieves high accuracy by reducing variance through ensemble learning. |
| XGBoost | Delivers the best overall performance due to boosting and optimized learning. |

---

## 6. Streamlit Web Application

The models are deployed using **Streamlit Community Cloud**.  
The application includes the following features:

- CSV file upload option for test data
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix visualization
- Classification report for the selected model
- AUC score display when target labels are available

---

## 7. Project Structure
```
ml-assignment-2/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model_evaluation.py        
│
│-- data/
│   │-- test_data.csv          
│
│-- model/
│   │-- __init__.py
│   │-- train_models.py
```


---

## 8. Links

- **GitHub Repository:**  
  https://github.com/2025aa05729-ashmit/ml-assignment-2

- **Live Streamlit App:**  
  https://2025aa05729.streamlit.app/

---

## 9. Execution Environment

The complete assignment was executed on the **BITS Virtual Lab environment**.  
A screenshot of the execution has been included in the final PDF submission as proof.

---

## 10. Conclusion

This assignment demonstrates an end-to-end machine learning workflow including model development, evaluation using multiple metrics, interactive visualization using Streamlit, and deployment on a cloud platform. Ensemble models such as Random Forest and XGBoost showed superior performance on the chosen dataset.

## Additional Files for Evaluation

- `data/test_data.csv` – Sample test dataset for verifying model predictions.
- `model_evaluation.py` – Script used to train models and compute evaluation metrics.
