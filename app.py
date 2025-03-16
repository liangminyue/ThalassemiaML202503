# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 加载模型和标准化器
model = joblib.load('xgb_regressor.pkl')
scaler = joblib.load('scaler.pkl')  # 需要预先加载scaler

# 设置页面标题
st.title('HGB预测系统')

# 创建表单以获取用户输入
with st.form("prediction_form"):
    age = st.number_input('年龄', min_value=0, max_value=100)
    gender = st.selectbox('性别', ['男', '女'])  # 更新性别选择框的选项
    height = st.number_input('身高')  # 新增身高输入
    weight = st.number_input('体重')  # 新增体重输入
    blood_volume = st.number_input('本次输血量')  # 新增本次输血量输入
    hgb_before = st.number_input('HGB前')  # 新增HGB前输入

    submitted = st.form_submit_button("预测")

if submitted:
    # 将输入数据转换为DataFrame
    input_data = pd.DataFrame([[age, 1 if gender == '男' else 0, height, weight, blood_volume, hgb_before]],
                             columns=['年龄', '性别', '身高', '体重', '本次输血量', 'HGB前'])  # 更新列名

    # 对输入数据进行标准化
    scaled_input = scaler.transform(input_data)

    # 进行预测
    prediction = model.predict(scaled_input)
    st.success(f'预测HGB值为: {prediction[0]:.2f}')
