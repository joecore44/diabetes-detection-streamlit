import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.write("""
# AI Diabetes Detection
With a Machine Learning Model Trained on an International Diabetes Dataset
""")
st.write('Author: Joe Shepard - www.joeshepard.dev')

image = Image.open('dd-header.jpg')
st.image(image, caption='Machien Learning', use_column_width=True)

data_frame = pd.read_csv('Diabetes.csv')
st.subheader('Display Diabetes Data: ')
st.dataframe(data_frame, width=1000)
st.write(data_frame.describe())
chart = st.bar_chart(data_frame)

X = data_frame.iloc[:, 0:8].values
Y = data_frame.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucoses', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 3)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    user_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'BMI': BMI,
        'DPF': DPF,
        'age': age
    }

    features = pd.DataFrame(user_data, index = [0])
    return features

user_input = get_user_input()

st.subheader('User Input:')
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100)+'%')

prediction = RandomForestClassifier.predict(user_input)

st.subheader('Classification:')
st.write(prediction)