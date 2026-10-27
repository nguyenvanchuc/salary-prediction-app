import os
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dự đoán Lương Nhân Viên",
    page_icon="💼",
    layout="wide"
)

# 1. Tải Model an toàn
@st.cache_resource
def load_artifacts():
    # Tìm file joblib tại thư mục chứa app.py
    model_path = Path(__file__).resolve().parent / 'salary_model_pipeline.joblib'
    if not model_path.exists():
        st.error(f"Không tìm thấy file mô hình tại: {model_path}")
        st.stop()
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Lỗi khi nạp file mô hình: {e}")
        st.info("Gợi ý: Phiên bản scikit-learn lúc train và deploy có thể đang lệch nhau.")
        st.stop()

artifacts = load_artifacts()
pipeline = artifacts['pipeline']
categories = artifacts['categories']
metrics = artifacts.get('metrics', {'r2': 0.0, 'rmse': 0.0})

# Tiêu đề giao diện
st.title("💼 Ứng dụng Dự đoán Mức Lương Nhân Viên")
st.markdown("Mô hình hồi quy tuyến tính tích hợp tự động tiền xử lý dữ liệu qua Pipeline.")

# Sidebar
with st.sidebar:
    st.header("📊 Thông số Mô hình")
    st.metric(label="Độ chính xác (R² Test)", value=f"{metrics['r2']:.4f}")
    st.metric(label="Sai số RMSE", value=f"{metrics['rmse']:.2f}")

# 2. Form nhập liệu
st.subheader("Nhập thông tin nhân viên")

with st.form("salary_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Tuổi (Age)", min_value=18, max_value=70, value=30, step=1)
        working_year = st.number_input("Số năm kinh nghiệm (WorkingYear)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        trained_year = st.number_input("Số năm đào tạo chuyên môn (TrainedYear)", min_value=0.0, max_value=15.0, value=1.0, step=0.5)
        
    with col2:
        dept = st.selectbox("Phòng ban (Dept)", options=categories.get('Dept', ['IT', 'HR', 'Sales']))
        gender = st.selectbox("Giới tính (Gender)", options=categories.get('Gender', ['Male', 'Female']))
        skill = st.selectbox("Kỹ năng (Skill)", options=categories.get('Skill', ['Junior', 'Senior', 'Lead']))
        
    submitted = st.form_submit_button("Dự đoán Mức Lương", type="primary", use_container_width=True)

if submitted:
    if working_year > (age - 18):
        st.error("Cảnh báo: Số năm kinh nghiệm không thể lớn hơn (Tuổi - 18)!")
    else:
        input_data = pd.DataFrame([{
            'Age': age,
            'WorkingYear': working_year,
            'TrainedYear': trained_year,
            'Dept': dept,
            'Gender': gender,
            'Skill': skill
        }])
        
        predicted_salary = pipeline.predict(input_data)[0]
        st.success(f"### Mức lương ước tính: {predicted_salary:,.2f}")
        with st.expander("Xem chi tiết dữ liệu vừa nhập"):
            st.dataframe(input_data)
