import streamlit as st
import requests

st.title("Annual Income Prediction")

# Input data
age = st.number_input("Age", min_value=18, max_value=100, value=30)
work_class = st.selectbox("Work Class", ['Private', 'Local-gov', 'Federal-gov', 'Self-emp-not-inc', 'State-gov', 
                                        'Self-emp-inc', 'Without-pay', 'Never-worked'])
education = st.selectbox("Education Level", ['HS-grad', 'Masters', 'Bachelors', 'Assoc-acdm', 'Some-college', '11th', '10th', 
                                             'Assoc-voc', 'Preschool', '7th-8th', 'Doctorate', '5th-6th', 'Prof-school', 
                                             '1st-4th', '9th', '12th'])
education_years = st.slider("Years of Education", 1, 20, 12)
marital_status = st.selectbox("Marital Status", ['Never-married', 'Divorced', 'Married-civ-spouse', 'Separated', 'Widowed', 
                                                'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.selectbox("Occupation", ['Machine-op-inspct', 'Other-service', 'Prof-specialty', 'Craft-repair', 
                                         'Protective-serv', 'Exec-managerial', 'Sales', 'Farming-fishing', 'Transport-moving', 
                                         'Handlers-cleaners', 'Adm-clerical', 'Tech-support', 'Priv-house-serv', 'Armed-Forces'])
relationship = st.selectbox("Relationship", ['Not-in-family', 'Unmarried', 'Husband', 'Own-child', 'Other-relative', 'Wife'])
race = st.selectbox("Race", ['Black', 'White', 'Asian-Pac-Islander', 'Other', 'Amer-Indian-Eskimo'])
gender = st.radio("Gender", ['Male', 'Female'])
capital_gains = st.number_input("Capital Gains ($)", min_value=0, value=0)
capital_losses = st.number_input("Capital Losses ($)", min_value=0, value=0)
weekly_hours = st.slider("Hours Worked Per Week", 1, 100, 40)
country_origin = st.selectbox("Country of Origin", ['United-States', 'Philippines', 'Puerto-Rico', 'Mexico', 'Cuba', 'India', 
                                                   'Jamaica', 'South', 'Laos', 'England', '?', 'El-Salvador', 'Germany', 
                                                   'Thailand', 'Poland', 'China', 'Greece', 'Trinadad&Tobago', 'Canada', 
                                                   'France', 'Dominican-Republic', 'Columbia', 'Nicaragua', 'Haiti', 
                                                   'Cambodia', 'Peru', 'Taiwan', 'Hungary', 'Italy', 'Japan', 'Vietnam', 
                                                   'Honduras', 'Portugal', 'Guatemala', 'Iran', 'Ecuador', 'Scotland', 
                                                   'Yugoslavia', 'Hong', 'Outlying-US(Guam-USVI-etc)', 'Ireland', 
                                                   'Holand-Netherlands'])

# Llamar a la API
if st.button("Predict Income"):
    api_url = "http://127.0.0.1:8000/predict"
    data = {
        "age": age,
        "work_class": work_class,
        "education": education,
        "education_years": education_years,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "gender": gender,
        "capital_gains": capital_gains,
        "capital_losses": capital_losses,
        "weekly_hours": weekly_hours,
        "country_origin": country_origin
    }
    response = requests.post(api_url, json=data)
    
    if response.status_code == 200:
        result = response.json()["prediction"]
        st.success(f"The model predicts that the annual income is: {result}")
    else:
        st.error("Error in API request!")
