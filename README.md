# Heart-Diseases-Prediction-In-Youngster
🛠️ Steps in the Project
1️⃣ Exploratory Data Analysis (EDA)
Checking dataset structure, missing values, and duplicate records

Visualizing key attributes such as age distribution, cholesterol levels, and stress impact

Creating correlation heatmaps to understand feature relationships

2️⃣ Data Preprocessing
Handling missing values using imputation

Encoding categorical variables with Label Encoding

Feature scaling using StandardScaler and MinMaxScaler

Outlier detection and removal using the IQR method

Splitting data into training and test sets (80% train, 20% test)

3️⃣ Machine Learning Models Used
Four different models are trained and compared based on their accuracy and performance metrics:
✔️ Logistic Regression – A simple and interpretable model
✔️ XGBoost Classifier – A powerful gradient boosting technique
✔️ Decision Tree Classifier – A rule-based model for classification
✔️ Support Vector Machine (SVM) – A robust classifier for complex decision boundaries

4️⃣ Model Evaluation & Comparison
Metrics used: Accuracy, Precision, Recall, F1-score

The best-performing model is identified based on overall evaluation metrics

📊 Results
After training all four models, their performance is compared using classification reports and accuracy scores.

The best model is selected for heart attack prediction based on its accuracy and reliability.

📌 How to Run the Project
Install the required dependencies:

bash
Copy
Edit
pip install numpy pandas seaborn matplotlib scikit-learn xgboost
Run the Python script (heart_attack_in_youngster_of_india.py) in Google Colab or a Jupyter Notebook.

The script will perform data preprocessing, visualization, model training, and evaluation automatically.
