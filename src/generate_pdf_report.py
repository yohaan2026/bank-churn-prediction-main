import os
import pandas as pd
from fpdf import FPDF
from datetime import datetime

def generate_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # 1. Title Page
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, text="Customer Churn Prediction - Bank Customer Data", ln=True, align='C')
    pdf.ln(20)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, text="Author: Yohaan Saji", ln=True, align='C')
    
    pdf.add_page()
    
    # 2. Introduction & Problem Statement
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, text="1. Introduction & Problem Statement", ln=True, align='L')
    pdf.set_font("Arial", size=12)
    text1 = "Customer churn is a critical metric for any financial institution. It represents the rate at which customers stop doing business with the bank, leading to a direct loss of recurring revenue and a lower return on customer acquisition costs. In highly competitive markets, retaining existing customers is significantly more cost-effective than acquiring new ones."
    pdf.multi_cell(190, 10, text=text1)
    pdf.ln(5)
    
    text2 = "This project aims to predict whether a given bank customer will churn (exit) based on their demographic and financial attributes. By identifying high-risk customers before they leave, the bank can proactively deploy targeted retention strategies, such as offering better interest rates, personalized financial products, or enhanced customer service."
    pdf.multi_cell(190, 10, text=text2)
    pdf.ln(10)
    
    # 3. Dataset Description
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, text="2. Dataset Description", ln=True, align='L')
    pdf.set_font("Arial", size=12)
    
    text3 = "The dataset used for this project is a real, publicly available dataset sourced from Kaggle: 'shrutimechlearn/churn-modelling'. It contains 10,000 records of bank customers and 14 features, including demographic information (Age, Gender, Geography) and financial behavior (CreditScore, Balance, NumOfProducts, EstimatedSalary)."
    pdf.multi_cell(190, 10, text=text3)
    pdf.ln(5)
    
    text4 = "An authenticity proof has been generated and validated, confirming that the dataset contains exactly 10,000 rows, 14 columns, and zero missing values across all fields. No synthetic data generation or unauthorized alterations have been applied to this data."
    pdf.multi_cell(190, 10, text=text4)
    pdf.ln(10)
    
    # 4. Methodology
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, text="3. Methodology", ln=True, align='L')
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, text="Exploratory Data Analysis (EDA)", ln=True, align='L')
    pdf.set_font("Arial", size=12)
    text5 = "The exploratory phase revealed several key insights. First, the dataset is highly imbalanced, with approximately 20% of customers having churned. Age is the strongest numeric predictor of churn, with older customers showing a significantly higher churn rate. Additionally, female customers and those located in Germany exhibit higher churn rates than other groups. Interestingly, customers with higher account balances also demonstrated a higher likelihood of leaving the bank."
    pdf.multi_cell(190, 10, text=text5)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, text="Preprocessing", ln=True, align='L')
    pdf.set_font("Arial", size=12)
    text6 = "Irrelevant identifiers, such as CustomerId, Surname, and RowNumber, were removed as they provide no predictive value and could lead to overfitting. Categorical features (Geography and Gender) were converted into numerical formats using one-hot encoding. To ensure models like Logistic Regression perform optimally, all continuous numerical features were scaled using StandardScaler, normalizing their distributions. The dataset was then split into an 80% training set and a 20% testing set, stratified by the target variable to maintain the original class distribution."
    pdf.multi_cell(190, 10, text=text6)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, text="Modeling", ln=True, align='L')
    pdf.set_font("Arial", size=12)
    text7 = "Three distinct machine learning models were trained to predict churn: Logistic Regression (serving as a baseline), Random Forest, and XGBoost. These models were evaluated using multiple metrics, including Accuracy, Precision, Recall, F1 Score, and ROC-AUC. Given the class imbalance, ROC-AUC and F1 Score were prioritized over simple Accuracy for final model selection."
    pdf.multi_cell(190, 10, text=text7)
    pdf.ln(10)
    
    pdf.add_page()
    
    # 5. Results
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, text="4. Results", ln=True, align='L')
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, text="Model Comparison", ln=True, align='L')
    metrics_path = '../outputs/models/metrics.csv'
    pdf.set_font("Arial", size=10)
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        
        # Determine column widths
        col_width = pdf.w / (len(df_metrics.columns) + 1)
        row_height = pdf.font_size * 2
        
        for col in df_metrics.columns:
            pdf.cell(col_width, row_height, str(col), border=1)
        pdf.ln(row_height)
        
        for row in df_metrics.itertuples(index=False):
            for item in row:
                pdf.cell(col_width, row_height, str(item)[:15], border=1)
            pdf.ln(row_height)
    else:
        pdf.cell(200, 10, text="Metrics data not found.", ln=True)
        
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, text="ROC Curves", ln=True, align='L')
    roc_img = '../outputs/figures/roc_curves.png'
    if os.path.exists(roc_img):
        pdf.image(roc_img, x=None, y=None, w=150)
    else:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, text="ROC Curve image not found.", ln=True)
        
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, text="Confusion Matrix (Best Model)", ln=True, align='L')
    cm_img = '../outputs/figures/confusion_matrix.png'
    if os.path.exists(cm_img):
        pdf.image(cm_img, x=None, y=None, w=150)
    else:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, text="Confusion Matrix image not found.", ln=True)
        
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, text="Feature Importance", ln=True, align='L')
    feat_img = '../outputs/figures/feature_importance.png'
    if os.path.exists(feat_img):
        pdf.image(feat_img, x=None, y=None, w=150)
    else:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, text="Feature Importance image not found.", ln=True)
        
    pdf.add_page()
    
    # 6. Key Findings
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, text="5. Key Findings", ln=True, align='L')
    pdf.set_font("Arial", size=12)
    findings = [
        "- Age is the most dominant factor in determining customer churn; middle-aged and older customers are at the highest risk.",
        "- The Random Forest model achieved the best overall performance, particularly in terms of ROC-AUC and balancing Precision with Recall.",
        "- Customers with high account balances are surprisingly more likely to churn, which may indicate that wealthier clients are being lured away by better offers from competitors.",
        "- The number of products a customer holds is highly predictive; customers with only 1 product or 3+ products show much higher churn rates compared to those with exactly 2 products.",
        "- Geographic location matters: customers in Germany churned at a noticeably higher rate compared to those in France or Spain."
    ]
    for finding in findings:
        pdf.multi_cell(190, 10, text=finding)
    pdf.ln(10)
    
    # 7. Conclusion & Business Recommendation
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, text="6. Conclusion & Business Recommendation", ln=True, align='L')
    pdf.set_font("Arial", size=12)
    text8 = "By utilizing the Random Forest model, the bank can reliably identify a large portion of customers who are about to churn. The business should proactively target high-balance customers and older demographics in Germany with tailored retention campaigns. Specifically, offering premium services or better interest rates to high-balance clients, and encouraging single-product users to adopt a second product, could significantly reduce the overall churn rate and preserve valuable revenue."
    pdf.multi_cell(0, 10, text=text8)
    pdf.ln(10)
    
    # 8. Limitations
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, text="7. Limitations", ln=True, align='L')
    pdf.set_font("Arial", size=12)
    limitations = [
        "- The dataset lacks temporal data (e.g., transaction frequency, recent customer service interactions), which are often strong real-time indicators of an intent to churn.",
        "- The models rely heavily on static demographic information, which may not capture sudden shifts in a customer's financial situation.",
        "- There is no information regarding the reason for churn, making it difficult to distinguish between unavoidable churn (e.g., death, relocation) and preventable churn (e.g., competitor offers)."
    ]
    for limit in limitations:
        pdf.multi_cell(0, 10, text=limit)
        pdf.ln(5)
        
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(0, 10, text="Contributor: Yohaan Saji", ln=True, align='C')
        
    # Save document
    pdf.output('../report/Customer_Churn_Report.pdf')
    print("Report generated successfully at report/Customer_Churn_Report.pdf")

if __name__ == '__main__':
    generate_pdf_report()
