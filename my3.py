import streamlit as st
import hashlib
import base64
import pandas as pd
from xgboost import XGBClassifier
#THIS IS MAIN CODE OF HEART DISEASES
from sklearn.metrics import accuracy_score

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Sample credentials (store securely in a real app)
users = {"admin": hash_password("password123"), "user": hash_password("1234")}

# Load and train model
@st.cache_resource
def train_model():
    data = pd.read_csv(r"C:\Users\Hp\Downloads\heart attack in youngster.csv")
    data.fillna(data.mean(numeric_only=True), inplace=True)
    data = pd.get_dummies(data, drop_first=True)
    
    X = data.drop('Heart Attack Likelihood', axis=1)
    y = data['Heart Attack Likelihood']
    
    import sklearn
    from sklearn import model_selection

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model trained with accuracy:", accuracy)
    
    return xgb_model

# Train model on app startup
model = train_model()

# Load Image & Convert to Base64
IMAGE_PATH = r"C:\Users\Hp\Downloads\img2.jpg"
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# Custom CSS Styling
def set_custom_css():
   st.markdown(
        f"""
        <style>
            [data-testid="stAppViewContainer"] {{
                background: url("data:image/jpeg;base64,{get_base64_image(IMAGE_PATH)}");
                background-size: cover;
                background-repeat: no-repeat;
                background-size: 100% auto;
                background-position: center;
                position: relative;
                height: 100vh;
            }}
            .title-container {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 100%;
                text-align: center;
            }}
            .title {{
                font-size: 46px;
                font-weight: bold;
                color: linear-gradient(to right, #1d2671, #c33764);
                text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
            }}
            .feature-container {{
                font-size: 24px;
                margin: 20px 0;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
            }}
            .stButton>button {{
                width: 85%;
                height: 50px;
                font-size: 25px;
                font-weight: bold;
                color: black;
                background-color: #F2DEDE;
                border-radius: 8px;
                border: none;
                transition: 0.3s;
            }}
            .stButton>button:hover {{
                background-color:rgb(235, 150, 139);
            
            }}
            .login-container {{
                max-width: 400px;
                margin: auto;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .login-title {{
                font-size: 28px;
                font-weight: bold;
                color: linear-gradient(to right, #faaca8, #ddd6f3);
                margin-bottom: 20px;
            }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Home Page
def home():
    set_custom_css()
    with st.sidebar:
        st.header("Navigation")
        if st.button("🏠 Home"):
            st.session_state["page"] = "home"
            st.rerun()
        if st.button("🔑 Login"):
            st.session_state["page"] = "login"
            st.rerun()
        if st.button("🔍 Prediction"):
            st.session_state["page"] = "prediction"
            st.rerun()
    st.markdown('<div class="title-container"><div class="title">Heart Disease Prediction With Machine Learning</div></div>', unsafe_allow_html=True)

# Login Page
def login():
    set_custom_css()
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🔒 Secure Your Heart: Log In to Predict</div>', unsafe_allow_html=True)
    st.markdown('<div class="ecg-line"></div>', unsafe_allow_html=True)
    
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    
    if st.button("🔍 Predict My Heart Health"):
        if username in users and users[username] == hash_password(password):
            st.success("✅ Login Successful! Redirecting...")
            st.session_state["authenticated"] = True
            st.session_state["page"] = "prediction"
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Please try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)
# Prediction Page
def prediction():
    set_custom_css()
    with st.sidebar:
        st.header("Navigation")
        if st.button("🏠 Home"):
            st.session_state["page"] = "home"
            st.rerun()
        if st.button("🔑 Logout"):
            st.session_state["authenticated"] = False
            st.session_state["page"] = "login"
            st.rerun()
    
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.title("Heart Disease Prediction")
    st.markdown("<h3>Enter Your Details Below</h3>", unsafe_allow_html=True)
    
    # User Inputs inside a feature card
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.radio("Gender", ["Male", "Female"])
    smoking_status = st.selectbox("Smoking Status", ["Never", "Occasionally", "Regularly"])
    alcohol_consumption = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"])
    diet_type = st.selectbox("Diet Type", ["Vegan", "Vegetarian", "Non-Vegetarian"])
    physical_activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "High"])
    screen_time = st.number_input("Screen Time (hrs/day)")
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    blood_pressure = st.text_input("Blood Pressure (systolic/diastolic mmHg)")
    ecg_results = st.selectbox("ECG Results", ["Normal", "ST-T abnormality", "Hypertrophy"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "Non-Anginal", "Asymptomatic"])
    max_heart_rate = st.number_input("Maximum Heart Rate Achieved")
    spo2 = st.number_input("Blood Oxygen Levels (SpO2%)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Predict"):
        # Parse blood pressure
        systolic, diastolic = map(int, blood_pressure.split('/'))
        
        input_data = pd.DataFrame([[age, gender, smoking_status, alcohol_consumption, diet_type,
                                    physical_activity, screen_time, stress_level, systolic, diastolic,
                                    ecg_results, chest_pain, max_heart_rate, spo2]],
                                  columns=["Age", "Gender", "Smoking Status", "Alcohol Consumption", "Diet Type",
                                           "Physical Activity Level", "Screen Time (hrs/day)", "Stress Level",
                                           "Systolic Blood Pressure", "Diastolic Blood Pressure", "ECG Results", 
                                           "Chest Pain Type", "Maximum Heart Rate Achieved", "Blood Oxygen Levels (SpO2%)"])
        
        # Convert categorical inputs to numerical using one-hot encoding
        input_data = pd.get_dummies(input_data)
        
        # Align input data with training data
        missing_cols = set(model.get_booster().feature_names) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[model.get_booster().feature_names]
        
        prediction = model.predict(input_data)[0]
        
        # Check thresholds for high-risk conditions
        high_risk_conditions = [
            age > 45,
            systolic > 140 or diastolic > 90,
            max_heart_rate < 100,
            spo2 < 95,
            smoking_status == "Regularly",
            stress_level == "High",
            physical_activity == "Sedentary",
            ecg_results in ["ST-T abnormality", "Hypertrophy"],
            chest_pain == "Typical"
        ]
        
        if prediction == 1:
            st.error("High Risk of Heart Disease")
            if any(high_risk_conditions):
                st.warning("You have one or more high-risk conditions. Please consult a doctor.")
        else:
            st.success("Low Risk of Heart Disease")
        
    st.markdown('</div>', unsafe_allow_html=True)
# Page Routing
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["page"] == "home":
    home()
elif st.session_state["page"] == "login":
    login()
elif st.session_state["page"] == "prediction" and st.session_state["authenticated"]:
    prediction()