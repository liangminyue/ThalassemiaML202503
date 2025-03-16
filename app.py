# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 性能优化：添加缓存装饰器
@st.cache_data
def load_resources():
    return (
        joblib.load('xgb_regressor.pkl'),
        joblib.load('scaler.pkl')
    )
model, scaler = load_resources()

# 设置页面标题
st.title('HGB预测系统')

# 用户体验优化：添加侧边栏说明
with st.sidebar:
    st.header("使用说明")
    st.markdown("""
    - 所有数值输入必须 0 或以上
    - 身高单位：厘米(cm)
    - 体重单位：千克(kg)
    - 输血量单位：毫升(ml)
    - HGB单位：克/升(g/L)
    """)

# 代码结构优化：封装输入创建逻辑
def create_inputs():
    with st.form("prediction_form"):
        inputs = {
            'age': st.number_input('年龄', 0, 100),
            'gender': st.selectbox('性别', options=['男', '女']),
            'height': st.number_input('身高(cm)', min_value=0.0),
            'weight': st.number_input('体重(kg)', min_value=0.0),
            'blood_volume': st.number_input('本次输血量(ml)', min_value=0),
            'hgb_before': st.number_input('输血前HGB值(g/L)', min_value=0.0)
        }
        return inputs, st.form_submit_button("预测")

# 辅助函数优化：独立数据处理函数
def process_inputs(inputs):
    GENDER_MAPPING = {'男': 1, '女': 0}
    
    df = pd.DataFrame([[
        inputs['age'],
        GENDER_MAPPING[inputs['gender']],
        inputs['height'],
        inputs['weight'],
        inputs['blood_volume'],
        inputs['hgb_before']
    ]], columns=['年龄', '性别', '身高', '体重', '本次输血量', 'HGB前'])
    
    return scaler.transform(df)

# 获取用户输入
inputs, submitted = create_inputs()

# 功能增强：添加数据验证与异常处理
if submitted:
    try:
        # 数值范围校验（排除性别字段）
        numeric_fields = ['age', 'height', 'weight', 'blood_volume', 'hgb_before']
        if not all(isinstance(inputs[field], (int, float)) for field in numeric_fields):
            raise ValueError("数值输入格式错误")
        if any(inputs[field] < 0 for field in numeric_fields):
            raise ValueError("输入值不能为负数")
            
        # 数据处理逻辑
        processed_data = process_inputs(inputs)
        
        # 预测结果展示优化
        with st.spinner('预测中...'):
            prediction = model.predict(processed_data)
            st.metric(label="预测HGB值", value=f"{prediction[0]:.2f} g/L")
            
    except Exception as e:
        st.error(f"发生错误: {str(e)}")