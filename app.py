import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt

# 添加环境变量设置，禁用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 性能优化：添加缓存装饰器
@st.cache_data
def load_resources():
    model = joblib.load('xgb_regressor.pkl')
    # 双重保障设置
    model.set_params(**{
        'tree_method': 'hist',
        'predictor': 'cpu_predictor',
        'gpu_id': -1
    })
    return model, joblib.load('scaler.pkl')
model, scaler = load_resources()

# 配置集中管理
FEATURE_NAMES = ['年龄', '性别', '身高', '体重', '本次输血量', 'HGB前']
GENDER_MAPPING = {'男': 1, '女': 0}

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
    df = pd.DataFrame([[
        inputs['age'],
        GENDER_MAPPING[inputs['gender']],
        inputs['height'],
        inputs['weight'],
        inputs['blood_volume'],
        inputs['hgb_before']
    ]], columns=['年龄', '性别', '身高', '体重', '本次输血量', 'HGB前'])

    return scaler.transform(df)

# 输入验证增强
def validate_inputs(inputs):
    errors = []
    if inputs['height'] < 50 or inputs['height'] > 250:
        errors.append("身高应在50-250cm之间")
    if inputs['weight'] < 2 or inputs['weight'] > 200:
        errors.append("体重应在2-200kg之间")
    return errors

# SHAP可视化函数
def plot_shap_explanation(model, sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(10, 6))
    shap.force_plot(explainer.expected_value,
                   shap_values,
                   sample,
                   feature_names=FEATURE_NAMES,
                   matplotlib=True,
                   show=False)
    plt.tight_layout()
    return plt.gcf()

# 获取用户输入
inputs, submitted = create_inputs()

# 功能增强：添加数据验证与异常处理
if submitted:
    try:
        # 数据验证
        validation_errors = validate_inputs(inputs)
        if validation_errors:
            raise ValueError("\n".join(validation_errors))

        # 数据处理
        processed_data = process_inputs(inputs)

        # 预测结果
        with st.spinner('预测中...'):
            prediction = model.predict(processed_data)
            st.metric(label="预测HGB值", value=f"{prediction[0]:.2f} g/L")

            # SHAP可视化
            st.subheader("特征影响分析")
            sample_df = pd.DataFrame([[
                inputs['age'],
                GENDER_MAPPING[inputs['gender']],
                inputs['height'],
                inputs['weight'],
                inputs['blood_volume'],
                inputs['hgb_before']
            ]], columns=FEATURE_NAMES)

            fig = plot_shap_explanation(model, sample_df.values)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"输入验证错误: {str(e)}")