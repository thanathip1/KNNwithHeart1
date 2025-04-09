from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 🎨 ตั้งค่าหน้าหลัก
st.set_page_config(page_title="ตรวจสอบโรคหัวใจ", layout="wide")
st.title('🫀 ตรวจสอบโรคหัวใจด้วย Machine Learning (KNN)')

# รูปภาพหัวเรื่อง
col1, col2 = st.columns(2)
with col1:
    st.image("./img/5.png", use_container_width=True)

with col2:
    st.image("./img/1.png", use_container_width=True)


# โหลดข้อมูล
dt = pd.read_csv("./data/Heart3.csv")

# 🧾 ส่วนแสดงข้อมูล
st.markdown("""
<div style="background-color:#F9D3FF;padding:12px;border-radius:10px;border:2px solid #ff85c1;">
<center><h3 style="color:#3c096c;">📊 ข้อมูลโรคหัวใจ (แสดงตัวอย่าง)</h3></center>
</div>
""", unsafe_allow_html=True)

st.subheader("🔹 ข้อมูลส่วนแรก 10 แถว")
st.dataframe(dt.head(10), use_container_width=True)

st.subheader("🔹 ข้อมูลส่วนสุดท้าย 10 แถว")
st.dataframe(dt.tail(10), use_container_width=True)

# 📌 กล่องทำนาย
st.markdown("""
<div style="background-color:#C3FBD8;padding:12px;border-radius:10px;border:2px solid #28c76f;">
<center><h3 style="color:#065f46;">🔍 ป้อนข้อมูลเพื่อทำนายการเป็นโรคหัวใจ</h3></center>
</div>
""", unsafe_allow_html=True)
st.markdown("")

# 🧾 รับข้อมูลจากผู้ใช้ (จัดเป็น 2 คอลัมน์)
input_col1, input_col2 = st.columns(2)
with input_col1:
    Age = st.number_input("📌 อายุ (Age)", min_value=0.0)
    Sex = st.number_input("📌 เพศ (Sex)", min_value=0.0)
    ChestPainType = st.number_input("📌 ประเภทอาการเจ็บหน้าอก (ChestPainType)", min_value=0.0)
    RestingBP = st.number_input("📌 ความดันขณะพัก (RestingBP)", min_value=0.0)
    Cholesterol = st.number_input("📌 คอเลสเตอรอล (Cholesterol)", min_value=0.0)
    FastingBS = st.number_input("📌 น้ำตาลในเลือดขณะอดอาหาร (FastingBS)", min_value=0.0)

with input_col2:
    RestingECG = st.number_input("📌 ECG ขณะพัก (RestingECG)", min_value=0.0)
    MaxHR = st.number_input("📌 อัตราการเต้นของหัวใจสูงสุด (MaxHR)", min_value=0.0)
    ExerciseAngina = st.number_input("📌 เจ็บหน้าอกขณะออกกำลัง (ExerciseAngina)", min_value=0.0)
    Oldpeak = st.number_input("📌 ST depression (Oldpeak)", min_value=0.0)
    ST_Slope = st.number_input("📌 ความชัน ST segment (ST_Slope)", min_value=0.0)

# 🔮 ทำนายผล
if st.button("🎯 ทำนายผล"):
    # เตรียมข้อมูล
    X = dt.drop('HeartDisease', axis=1)
    y = dt['HeartDisease']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_scaled, y)

    input_data = np.array([[Age, Sex, ChestPainType, RestingBP, Cholesterol,
                            FastingBS, RestingECG, MaxHR, ExerciseAngina,
                            Oldpeak, ST_Slope]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.markdown("---")
    if prediction[0] == 1:
        st.markdown("""
        <div style="background-color:#ff8fa3;padding:20px;border-radius:10px;text-align:center;">
        <h2 style="color:white;">⚠️ ตรวจพบความเสี่ยง! คุณอาจเป็นโรคหัวใจ</h2>
        </div>
        """, unsafe_allow_html=True)
        st.image("./img/2.png", use_container_width=True)
    else:
        st.markdown("""
        <div style="background-color:#a3ffd6;padding:20px;border-radius:10px;text-align:center;">
        <h2 style="color:#064e3b;">✅ ไม่พบความเสี่ยงของโรคหัวใจ</h2>
        </div>
        """, unsafe_allow_html=True)
        st.imagest.image("./img/1.png", use_container_width=True)
else:
    st.info("📝 กรอกข้อมูลให้ครบแล้วกดปุ่มเพื่อทำนาย")
