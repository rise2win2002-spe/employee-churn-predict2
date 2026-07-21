
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Churn Prediction", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("employee_churn_pipeline.pkl")

model = load_model()

st.markdown("""
<style>
.block-container{padding-top:1.5rem;}
.big-title{font-size:2.7rem;font-weight:700;}
.footer{text-align:center;color:gray;font-size:0.9rem;margin-top:40px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📊 Employee Churn Prediction System</div>', unsafe_allow_html=True)
st.caption("Decision Tree + PCA | MCA Final Project")

with st.sidebar:
    st.header("Employee Information")
    if st.button("Load Sample Employee"):
        st.session_state.sample=True
    sample=st.session_state.get("sample",False)

    age=st.number_input("Age",18,65,35 if sample else 30)
    gender=st.selectbox("Gender",["Female","Male"],index=1 if sample else 0)
    marital=st.selectbox("Marital Status",["Divorced","Married","Single"],index=1 if sample else 0)
    dept=st.selectbox("Department",["Finance","HR","IT","Marketing","Sales"],index=4 if sample else 0)
    role=st.selectbox("Job Role",["Analyst","Assistant","Executive","Manager"],index=3 if sample else 0)
    job_level=st.slider("Job Level",1,5,3 if sample else 2)
    monthly_income=st.number_input("Monthly Income",1000,50000,18000 if sample else 12000)
    hourly_rate=st.number_input("Hourly Rate",10,200,60)
    years_company=st.slider("Years at Company",0,40,6 if sample else 5)
    years_role=st.slider("Years in Current Role",0,20,4 if sample else 3)
    years_promo=st.slider("Years Since Last Promotion",0,15,2)
    worklife=st.slider("Work Life Balance",1,5,3)
    satisfaction=st.slider("Job Satisfaction",1,5,3)
    performance=st.slider("Performance Rating",1,5,3)
    training=st.slider("Training Hours",0,100,20)
    overtime=st.selectbox("Overtime",["No","Yes"],index=1 if sample else 0)
    projects=st.slider("Project Count",1,20,6 if sample else 5)
    hours=st.slider("Hours Worked/Week",20,80,48 if sample else 45)
    absent=st.slider("Absenteeism",0,30,2)
    env=st.slider("Work Environment Satisfaction",1,5,3)
    manager=st.slider("Relationship with Manager",1,5,3)
    involvement=st.slider("Job Involvement",1,5,3)
    distance=st.slider("Distance From Home",1,50,10)
    companies=st.slider("Companies Worked",0,10,2)

def build_df():
    return pd.DataFrame([{
        "Age":age,
        "Gender":1 if gender=="Male" else 0,
        "Job_Level":job_level,
        "Monthly_Income":monthly_income,
        "Hourly_Rate":hourly_rate,
        "Years_at_Company":years_company,
        "Years_in_Current_Role":years_role,
        "Years_Since_Last_Promotion":years_promo,
        "Work_Life_Balance":worklife,
        "Job_Satisfaction":satisfaction,
        "Performance_Rating":performance,
        "Training_Hours_Last_Year":training,
        "Overtime":1 if overtime=="Yes" else 0,
        "Project_Count":projects,
        "Average_Hours_Worked_Per_Week":hours,
        "Absenteeism":absent,
        "Work_Environment_Satisfaction":env,
        "Relationship_with_Manager":manager,
        "Job_Involvement":involvement,
        "Distance_From_Home":distance,
        "Number_of_Companies_Worked":companies,
        "Marital_Status_Married":1 if marital=="Married" else 0,
        "Marital_Status_Single":1 if marital=="Single" else 0,
        "Department_HR":1 if dept=="HR" else 0,
        "Department_IT":1 if dept=="IT" else 0,
        "Department_Marketing":1 if dept=="Marketing" else 0,
        "Department_Sales":1 if dept=="Sales" else 0,
        "Job_Role_Assistant":1 if role=="Assistant" else 0,
        "Job_Role_Executive":1 if role=="Executive" else 0,
        "Job_Role_Manager":1 if role=="Manager" else 0
    }])

if st.button("🚀 Predict Churn", use_container_width=True):
    X=build_df()
    pred=model.predict(X)[0]
    prob=float(model.predict_proba(X)[0][1])

    c1,c2,c3=st.columns(3)
    c1.metric("Attrition Probability",f"{prob:.1%}")
    c2.metric("Stay Probability",f"{1-prob:.1%}")
    if prob<0.30:
        risk="🟢 Low"
    elif prob<0.60:
        risk="🟡 Moderate"
    else:
        risk="🔴 High"
    c3.metric("Risk Level",risk)

    st.progress(prob)

    if pred==1:
        st.error("⚠ Employee is likely to leave.")
    else:
        st.success("✅ Employee is likely to stay.")

    st.subheader("HR Recommendations")
    if prob>=0.60:
        st.markdown("- Salary review\n- Career discussion\n- Manager intervention\n- Training opportunities")
    elif prob>=0.30:
        st.markdown("- Monitor engagement\n- Monthly feedback\n- Recognition program")
    else:
        st.markdown("- Continue engagement\n- Reward good performance")

with st.expander("Model Information"):
    st.write("""
**Algorithm:** Decision Tree

**Preprocessing:** StandardScaler + PCA

**Input Features:** 30 encoded features

**Dataset:** Kaggle Employee Attrition Dataset
""")

st.markdown('<div class="footer">Developed by Satgur Prasad Rao | MCA Final Project | Amrita Ahead University</div>', unsafe_allow_html=True)
