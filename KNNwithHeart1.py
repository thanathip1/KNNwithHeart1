from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

st.title('การตรวจสอบโรคหัวใจด้วยเทคนิค Machine Learning')
col1,col2 = st.columns(2)

with col1:
   st.image("./img/5.jpg")

with col2:
   st.image("./img/1.jpg")

html_7 = """
<div style="background-color:#fec8ff;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h3 style="color:black;">ข้อมูลโรคหัวใจสำหรับทำนาย</h3></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
st.markdown("")


st.subheader("ข้อมูลส่วนแรก 10 แถว")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h3 style="color:black;">ทำนายข้อมูล</h3></center>
</div>
"""
st.markdown("")
st.markdown("")
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

Age = st.number_input("กรุณาเลือกข้อมูล Age")
Sex = st.number_input("กรุณาเลือกข้อมูล Sex")
ChestPainType = st.number_input("กรุณาเลือกข้อมูล ChestPainType")
RestingBP = st.number_input("กรุณาเลือกข้อมูล RestingBP")
Cholesterol = st.number_input("กรุณาเลือกข้อมูล Cholesterol")
FastingBS = st.number_input("กรุณาเลือกข้อมูล FastingBS")
RestingECG = st.number_input("กรุณาเลือกข้อมูล RestingECG")
MaxHR = st.number_input("กรุณาเลือกข้อมูล MaxHR")
ExerciseAngina = st.number_input("กรุณาเลือกข้อมูล ExerciseAngina")
Oldpeak = st.number_input("กรุณาเลือกข้อมูล Oldpeak")
ST_Slope = st.number_input("กรุณาเลือกข้อมูล ST_Slope")

if st.button("ทำนายผล"):
    #st.write("ทำนาย")
   X = dt.drop('HeartDisease', axis=1)
   y = dt.HeartDisease

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  
    
   x_input = np.array([[Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == '1':
    st.write("เป็นโรคหัวใจ")
    st.image("./img/2.png")
   else:
    st.write("ไม่เป็นโรคหัวใจ")
    st.image("./img/1.jpg")
else:
    st.write("ไม่ทำนาย")