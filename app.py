
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Churn Prediction", page_icon="📊", layout="wide")

model = joblib.load("employee_churn_pipeline_full.pkl")

st.title("📊 Employee Churn Prediction")
st.caption("Decision Tree + PCA Model")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age",18,70,30)
    gender = st.selectbox("Gender",["Female","Male"])
    marital = st.selectbox("Marital Status",["Divorced","Married","Single"])
    dept = st.selectbox("Department",["Finance","HR","IT","Marketing","Sales"])
    role = st.selectbox("Job Role",["Analyst","Assistant","Executive","Manager"])
    job_level = st.slider("Job Level",1,5,2)
    monthly_income = st.number_input("Monthly Income",1000,50000,12000)
    hourly_rate = st.number_input("Hourly Rate",10,200,60)

with col2:
    years_company = st.slider("Years at Company",0,40,5)
    years_role = st.slider("Years in Current Role",0,25,3)
    years_promo = st.slider("Years Since Last Promotion",0,20,2)
    worklife = st.slider("Work Life Balance",1,5,3)
    satisfaction = st.slider("Job Satisfaction",1,5,3)
    performance = st.slider("Performance Rating",1,5,3)
    training = st.slider("Training Hours Last Year",0,100,20)
    overtime = st.selectbox("Overtime",["No","Yes"])
    projects = st.slider("Project Count",1,20,5)
    hours = st.slider("Average Hours Worked Per Week",20,80,45)
    absent = st.slider("Absenteeism",0,30,2)
    env = st.slider("Work Environment Satisfaction",1,5,3)
    manager = st.slider("Relationship with Manager",1,5,3)
    involvement = st.slider("Job Involvement",1,5,3)
    distance = st.slider("Distance From Home",1,50,10)
    companies = st.slider("Number of Companies Worked",0,10,2)

def build_df():
    d = {
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
    }
    return pd.DataFrame([d])

if st.button("Predict Churn", type="primary"):
    X = build_df()
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    if pred==1:
        st.error(f"⚠ High Risk of Attrition\n\nProbability: {prob:.1%}")
        st.markdown("""
### Recommended HR Actions
- Review compensation
- Career discussion
- Additional training
- Manager feedback session
""")
    else:
        st.success(f"✅ Employee Likely to Stay\n\nProbability of Attrition: {prob:.1%}")
