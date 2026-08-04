# Customer Churn Prediction — Bank Customer Data

This is a complete, submission-ready university data science project that predicts customer churn for a bank. The goal is to identify which customers are most likely to exit the bank based on their demographic and financial attributes, allowing the business to deploy targeted retention strategies.

## Dataset
The project uses a genuine, publicly sourced dataset from Kaggle:
- **Source:** [shrutimechlearn/churn-modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- **Size:** 10,000 rows × 14 columns
- **Features:** Demographics (Age, Gender, Geography) and financial attributes (CreditScore, Balance, NumOfProducts, EstimatedSalary).
- **Proof of Authenticity:** See `report/dataset_source_proof.txt`.

## Tech Stack
- **Python 3.11**
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Reporting:** `jupyter` (notebooks), `python-docx` (Word report)

## How to Run
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute the Notebook:**
   You can either open `notebooks/main_analysis.ipynb` in Jupyter/VS Code and run it cell-by-cell, or execute it headlessly via `nbconvert`:
   ```bash
   jupyter nbconvert --execute --inplace notebooks/main_analysis.ipynb
   ```
3. **Generate the Report:**
   To build the final DOCX report using the generated plots and metrics:
   ```bash
   python src/generate_report.py
   ```

## Key Results Summary
- **Full Report:** [Download/View PDF](report/Customer_Churn_Report.pdf) | [Download DOCX](report/Customer_Churn_Report.docx)
- **Best Model:** Random Forest out-performed Logistic Regression and XGBoost.
- **Top Drivers of Churn:** Age (older customers churn more) and Account Balance (higher balance customers churn more).
- **Geographic Trend:** Customers in Germany showed a significantly higher churn rate compared to France and Spain.
- **Recommendation:** Proactively target older and high-balance demographics with personalized retention campaigns to preserve revenue.

## Key Visuals

### Churn Class Imbalance
![Churn Class Imbalance](outputs/figures/eda_1_class_dist.png)
*Approximately 20% of the customers in the dataset have churned, creating a class imbalance that required careful evaluation metrics (ROC-AUC).*

### ROC Curves
![ROC Curves](outputs/figures/roc_curves.png)
*Random Forest achieved the highest ROC-AUC score, indicating it is the most capable model at distinguishing between retained and churned customers.*

### Confusion Matrix (Random Forest)
![Confusion Matrix](outputs/figures/confusion_matrix.png)
*The confusion matrix for the best model shows strong performance in identifying true negatives, while maximizing the detection of true positives.*

### Feature Importance
![Feature Importance](outputs/figures/feature_importance.png)
*Age is overwhelmingly the most predictive feature for customer churn, followed by account balance and the number of products.*


## Folder Structure
```
churn_project/
├── data/
│   ├── raw/             # Original untouched CSV downloaded directly from Kaggle
│   └── processed/       # Cleaned, encoded, and scaled data ready for modeling
├── notebooks/           # Main Jupyter Notebook (main_analysis.ipynb)
├── outputs/
│   ├── figures/         # Generated EDA and modeling charts (.png)
│   └── models/          # Saved best model (.pkl) and metrics (.csv)
├── report/              # Final DOCX report and dataset authenticity proof
├── src/                 # Python scripts used to build the notebook and report
├── README.md            # This file
└── requirements.txt     # Exact pip package versions used
```
