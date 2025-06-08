import streamlit as st
import joblib
import pandas as pd

model = joblib.load('model.pkl')

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
fare = st.slider("Fare", 0, 250, 32)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.slider("Parents/Children Aboard", 0, 6, 0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

if st.button("Predict"):
    data = {
        'Pclass': pclass,
        'Sex': 0 if sex == "male" else 1,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': {"C": 0, "Q": 1, "S": 2}[embarked]
    }

    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    result = "Survived 🎉" if pred else "Did Not Survive 😢"
    st.success(f"Prediction: {result}")
