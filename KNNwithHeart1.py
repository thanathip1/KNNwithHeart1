from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.header('💓 โปรแกรมวิเคราะห์ข้อมูลสุขภาพหัวใจ')
st.title('🔍 Heart Data Classification (KNN)')

# โหลดข้อมูล
dt = pd.read_csv("/mnt/data/Heart3.csv")

# ส่วนแสดงข้อมูล
html_7 = """
<div style="background-color:#f2b5d4;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ข้อมูลสุขภาพหัวใจเพื่อการวิเคราะห์</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
st.markdown("")

st.subheader("📌 ตัวอย่างข้อมูล 10 แถวแรก")
st.write(dt.head(10))
st.subheader("📌 ตัวอย่างข้อมูล 10 แถวสุดท้าย")
st.write(dt.tail(10))

# สถิติพื้นฐาน
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล
st.subheader("📊 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล (Boxplot)")
target_col = 'target'  # ปรับให้ตรงกับไฟล์ถ้าชื่อไม่ตรง
feature = st.selectbox("เลือกฟีเจอร์", dt.columns.drop(target_col))

fig, ax = plt.subplots()
sns.boxplot(data=dt, x=target_col, y=feature, ax=ax)
st.pyplot(fig)

# pairplot
if st.checkbox("🔎 แสดง Pairplot (อาจใช้เวลาโหลดเล็กน้อย)"):
    fig2 = sns.pairplot(dt, hue=target_col)
    st.pyplot(fig2)

# ทำนายด้วย KNN
html_8 = """
<div style="background-color:#91e3e9;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูลสุขภาพ</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)

# แสดง input เฉพาะ 4 ฟีเจอร์แรกเพื่อความง่าย (สามารถเปลี่ยนได้)
feature_names = dt.columns.drop(target_col).tolist()
input_vals = []

st.write("กรุณาเลือกค่าต่าง ๆ สำหรับทำนาย:")

for f in feature_names[:4]:  # ใช้แค่ 4 ฟีเจอร์แรก
    val = st.number_input(f"ป้อนค่า {f}", float(dt[f].min()), float(dt[f].max()), float(dt[f].mean()))
    input_vals.append(val)

if st.button("🧠 ทำนายผล"):
    X = dt[feature_names]
    y = dt[target_col]

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    prediction = model.predict([input_vals])
    st.success(f"✅ ผลการทำนาย: {prediction[0]}")
else:
    st.info("กรุณาใส่ข้อมูลและกดปุ่มเพื่อทำนาย")

