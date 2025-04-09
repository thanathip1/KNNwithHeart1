from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


st.header('kairung')
st.title('TESTPYTHON')

st.subheader("ข้อมูลส่วนแรก 10 แถว")
dt = pd.read_csv("./data/Heart.csv")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))


