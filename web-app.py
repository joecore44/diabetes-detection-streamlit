import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.write("""
# Detect Diabetes
With a Machine Learning Model Trained on International Diabetes Dataset
""")

image = Image.Open('\Users\josephshepard\Documents\web\Python\Machine-Learning\diabetes-detection-app\dd-header.jpg')
st.image(image, caption='Machien Learning', use_column_width=True)

data_frame = pd.read_csv('Python/Machine-Learning/diabetes-detection-app/Diabetes.csv')
st.subheader('Display Diabetes Data: ')
st.dataframe(data_frame)
st.write(data_frame.describe())
chart = st.bar_chart(data_frame)





