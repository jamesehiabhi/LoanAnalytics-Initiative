# LoanAnalytics-Initiative
## Machine Learning: The Future of Loan Risk Prediction

<img src="https://github.com/jamesehiabhi/LoanAnalytics-Initiative/blob/main/Displays/LoanAlytics_Cover.png" width="800" height="200"/> 
 
### INTRODUCTION
**LoanAnalytics Inc.**, a leading financial services provider, seeks to enhance its loan evaluation process by implementing an innovative loan default prediction model. With rising economic uncertainties and increasing default rates, traditional methods relying on static borrower data are no longer sufficient. This project aims to leverage machine learning to predict default risks more accurately, enabling faster loan approvals, improved risk management and minimized financial losses. By integrating dynamic borrower data with advanced algorithms, LoanAnalytics Inc. can make smarter, data-driven lending decisions and maintain a competitive edge in the industry.

 <img src="https://github.com/jamesehiabhi/LoanAnalytics-Initiative/blob/main/Displays/Cover1.png" alt="Displays" width="900" height="400"/> 

### EXECUTIVE SUMMARY
This report analyzes a dataset of over 255,000 loan records to identify factors contributing to loan defaults. Using machine learning models and feature importance analysis, we uncover key drivers behind default behavior and provide actionable recommendations for mitigating risks. The findings aim to assist financial institutions in refining their credit risk strategies and improving loan portfolio performance.

**Key highlights:**
- **Default Rate:** 11.6% of loans in the dataset defaulted.
- **Top Predictors of Default:** Income, interest rate, loan amount, and age are the most influential factors.
- **Actionable Insights:** Tailored strategies can be implemented for high-risk borrowers based on these predictors.

### DATASET OVERVIEW
The dataset contains demographic, financial, and loan-related attributes for borrowers. Key features include:
- **Demographics:** Age, education level, marital status, employment type.
- **Financial Metrics:** Income, credit score, debt-to-income (DTI) ratio, number of credit lines.
- **Loan Details:** Loan amount, interest rate, loan term, loan purpose.
- **Default Indicator:** Binary variable indicating whether a borrower defaulted (1) or not (0).

<img src="https://github.com/jamesehiabhi/LoanAnalytics-Initiative/blob/main/Displays/Summary.png" alt="Displays" width="600" height="200"/> 

### METHODOLOGY
- **STEP 1: Data Cleaning:**
  - Handle missing values using appropriate imputation techniques.
  - Remove duplicate records and irrelevant columns that do not contribute to prediction.
  - Identify and correct anomalies in the dataset to ensure data quality.

- **STEP 2: Exploratory Data Analysis (EDA):**
  - Visualize feature distributions, relationships, and correlations using plots like histograms and heatmaps.
  - Identify patterns, trends, and anomalies that may influence loan defaults.
  - Formulate hypotheses to guide feature engineering and model selection.

- **STEP 3: Data Preprocessing:**
  - Scale or normalize numerical features and encode categorical variables for compatibility with machine learning models.
  - Split the data into training, validation, and test sets to ensure robust evaluation.

- **STEP 4: Model Training:**
  - Select and train machine learning models such as Logistic Regression, Random Forest, or Gradient Boosting.
  - Conduct hyperparameter tuning and k-fold cross-validation for model improvement.
  - Experiment with multiple algorithms and compare their performance.

- **STEP 5: Model Evaluation:**
  - Assess model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
  - Analyze performance across subsets (e.g., borrower income levels) and perform error analysis.
  - Compare results to a baseline model to measure improvements.

- **STEP 6: Model Optimization:**
  - Fine-tune hyperparameters using techniques like Grid Search or Random Search.
  - Apply regularization or ensemble methods to address overfitting and enhance performance.
  - Refine feature selection and ensure the model generalizes well to unseen data.

### EXPECTED OUTCOME
- **Accurate Default Predictions:** A machine learning model capable of predicting loan default risk with high precision and reliability.
- **Streamlined Loan Approvals:** Faster and more efficient loan approval processes through predictive insights and automation.
- **Reduced Financial Losses:** Early identification of high-risk borrowers to mitigate defaults and improve profitability.
- **Data-Driven Lending Strategies:** Insights into borrower behavior and risk factors to refine lending policies and tailor loan terms.
- **Competitive Advantage:** Enhanced decision-making and risk management, positioning LoanAnalytics Inc. as a leader in the lending industry.

<img src="https://github.com/jamesehiabhi/LoanAnalytics-Initiative/blob/main/Displays/Feature%20Imp1.png" alt="Displays" width="600" height="200"/> 
<img src="https://github.com/jamesehiabhi/LoanAnalytics-Initiative/blob/main/Displays/Feature%20Imp.png" alt="Displays" width="800" height="400"/>

### Exploratory Data Analysis (EDA)
**1. Numerical Data**

During the EDA process on numerical data, univariate and bivariate analyses revealed that the distributions of **Age, Income, Loan Amount, Credit Score, Months Employed, Interest Rate**, and **DTI Ratio** are uniform, with no outliers detected. The correlation analysis showed moderate relationships among the numerical features, with Interest Rate exhibiting a positive correlation of **0.13** with loan defaults, while **Age** displayed a negative correlation of **-0.17** with default.

<img src="https://github.com/jamesehiabhi/LoanAnalytics-Initiative/blob/main/Displays/Numerical.png" alt="Displays" width="800" height="400"/> 
 
**2. Categorical Data**

During the EDA process on categorical data, univariate analysis revealed that the distributions of **Education** and **Marital Status** are uniform. Bivariate analysis of Education against loan defaults indicated that customers with a **high school education** have the highest likelihood of defaulting, followed by those with a **bachelor's degree, master's degree,** and then **PhD holders**, who exhibit the lowest default risk. 

<img src="https://github.com/jamesehiabhi/LoanAnalytics-Initiative/blob/main/Displays/Categorical.png" alt="Displays" width="900" height="500"/> 
 
**3. Data Preprocessing**

During the data preprocessing phase, categorical variables were converted to numerical features using **Label Encoding.** The dataset was then split into **80% training** and **20% testing** using the train_test_split() function from the **sklearn.model_selection** library. Additionally, feature scaling was applied to ensure consistency in the dataset for model training. 

<img src="https://github.com/jamesehiabhi/LoanAnalytics-Initiative/blob/main/Displays/Data%20Split.png" alt="Displays" width="800" height="400"/> 
 
**4. Model Training**

During the model training phase, **Logistic Regression** was initially used to train the dataset, but it yielded unsatisfactory results after evaluation. To improve performance, the **Min-Max Scaler** was applied, as the dataset had a uniform distribution. Following this optimization, additional supervised machine learning models, including **DecisionTreeClassifier, SGDClassifier,** and **RandomForestClassifier,** were explored to enhance predictive accuracy and performance.

<img src="https://github.com/jamesehiabhi/LoanAnalytics-Initiative/blob/main/Displays/Train%20model.png" alt="Displays" width="700" height="500"/> 
 
**5. Model Evaluation**

Model evaluation was conducted to verify the performance and readiness of the trained model for deployment. The results were commendable, achieving a **precision of 89%, recall of 100%, F1-score of 94%,** and **accuracy of 89%.** These metrics were attained after applying feature scaling and further optimizations to fine-tune the model for improved performance.

<img src="https://github.com/jamesehiabhi/LoanAnalytics-Initiative/blob/main/Displays/Model%20Evaluation.png" alt="Displays" width="700" height="500"/> 
 
**KEY INSIGHTS**
- **Data Quality:** The dataset was clean, with no missing values or duplicates, ensuring a strong foundation for analysis and modeling.

- **Numerical Feature Analysis:**
  - No outliers were detected in key numerical features such as Age, Income, Loan Amount, and Credit Score.
  - Short-term loans (<24 months) have higher default rates due to potentially higher monthly repayment burdens.
  - Loans for discretionary purposes (e.g., "Other") show elevated risk compared to auto or business loans.
  - **Interest Rate** showed a moderate positive correlation of **0.13** with loan defaults, while **Age** exhibited a negative correlation of **-0.17,** indicating older borrowers are less likely to default.

- **Categorical Feature Analysis:**
  - Customers with a high school education had the highest likelihood of defaulting, followed by those with a bachelor's degree, master's degree, and PhD holders.
  - Unemployed individuals are significantly more prone to default.
  - Marital Status: Divorced borrowers exhibit higher default rates than married or single individuals

- **Model Training and Optimization:**
  - Initial Logistic Regression performance was unsatisfactory, but applying the Min-Max Scaler and exploring advanced models such as RandomForestClassifier significantly improved outcomes.

- **Model Evaluation:**
  - After fine-tuning, the model achieved a precision of 90%, recall of 89%, F1-score of 89%, and accuracy of 81%, indicating robust predictive performance.

**Business Impact:** _The project provides actionable insights to identify high-risk borrowers early, optimize lending strategies, and reduce financial losses while maintaining efficiency in loan approvals._

### RECOMMENDATIONS
🚀 **Dynamic Risk Scoring:** Utilize the model's predictions to develop a dynamic risk scoring system, enabling tailored loan terms such as adjusted interest rates or collateral requirements.

🚀 **Monitor Model Performance:** Regularly evaluate the model using real-world data to ensure its effectiveness remains consistent over time and refine it as necessary.

🚀 **Targeted Financial Education:** Provide financial literacy programs for high-risk groups, such as younger borrowers or those with limited financial experience.

🚀 **Loan Restructuring Options:** Offer flexible repayment plans, such as extended loan terms or temporary payment deferrals, to support struggling borrowers.


________________________________________
### CONCLUSION
This analysis highlights critical factors influencing loan defaults and provides actionable recommendations to reduce risk exposure for financial institutions. By leveraging data-driven insights and implementing targeted strategies, lenders can improve portfolio performance while supporting borrowers in achieving financial stability. By adopting these recommendations and continuously refining the model, **LoanAnalytics Inc.** can strengthen its position in the competitive lending industry and make smarter, data-driven decisions.
________________________________________

<br>

### *Kindly share your feedback and I am happy to Connect 🌟*

<img src="https://github.com/jamesehiabhi/LoanAnalytics-Initiative/blob/main/Displays/My%20Card1.jpg" alt="Displays" width="600" height="150"/>


