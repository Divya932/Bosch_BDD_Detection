import streamlit as st
from data_analysis import (
    read_json,
    count_classes,
    plot_class_distribution,
    plot_bbox_size_distribution,
    plot_scene_distribution
)

st.set_page_config(page_title="BDD100K Data Analysis Dashboard", layout="wide")
st.title("📊 BDD100K Data Analysis Dashboard")

# --- Modify these paths based on where the files are inside the Docker container ---
train_json_path = "/bosch_od/bosch/datasets/labels/bdd100k_train.json"
val_json_path = "/bosch_od/bosch/datasets/labels/bdd100k_val.json"

# Load data from disk
try:
    train_data = read_json(train_json_path)
    val_data = read_json(val_json_path)
    st.sidebar.success("✅ JSON files loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load JSON files: {e}")
    st.stop()

# Tabs for different analyses
tabs = st.tabs(["Class Distribution", "BBox Size", "Scene Distribution"])

with tabs[0]:
    st.subheader("📌 Class Distribution")
    train_class_count = count_classes(train_data)
    val_class_count = count_classes(val_data)

    st.markdown("#### Train Class Distribution")
    plot_class_distribution(train_class_count, title="Train Class Distribution")
    st.pyplot()

    st.markdown("#### Val Class Distribution")
    plot_class_distribution(val_class_count, title="Val Class Distribution")
    st.pyplot()

    st.markdown("#### Train vs Val Class Distribution")
    plot_class_distribution([train_class_count, val_class_count], title="Train vs Val Class Distribution")
    st.pyplot()

with tabs[1]:
    st.subheader("📦 Bounding Box Size Distribution")
    normalize = st.checkbox("Normalize BBox Size by Image Size", value=True)
    bins = st.slider("Number of Histogram Bins", 10, 100, 30)
    plot_bbox_size_distribution(train_data, val_data, bins=bins, normalize=normalize)
    st.pyplot()

with tabs[2]:
    st.subheader("🌅 Scene/Time of Day Distribution")
    attr = st.selectbox("Select Attribute", ["scene", "timeofday", "weather"])
    plot_scene_distribution(train_data, val_data, attr=attr)
    st.pyplot()
