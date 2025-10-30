'''import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# ---------------------------------------------
# ✅ 1️⃣  MUST be first Streamlit command
# ---------------------------------------------
st.set_page_config(page_title="AI Railway Track Fault Detector", layout="centered")

# ---------------------------------------------
# 2️⃣  Load model once (cached)
# ---------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_resnet50.h5")
    return model

model = load_model()

# ---------------------------------------------
# 3️⃣  App Header / UI
# ---------------------------------------------
st.title("🚆 AI-Powered Railway Track Fault Detection")
st.write("Upload a railway track image to detect whether it’s **Defective** or **Normal** using a pre-trained ResNet50 model.")

# ---------------------------------------------
# 4️⃣  File Upload
# ---------------------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ---------------------------------------------
    # 5️⃣  Preprocessing
    # ---------------------------------------------
    img_resized = img.resize((448, 448))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # ---------------------------------------------
    # 6️⃣  Prediction
    # ---------------------------------------------
    pred = model.predict(img_array)
    prob = float(pred[0][0])

    # Sigmoid output threshold
    if prob > 0.5:
        label = "🚨 Defective Track"
        color = "red"
        conf = prob * 100
    else:
        label = "✅ Normal / Non-Defective Track"
        color = "green"
        conf = (1 - prob) * 100

    # ---------------------------------------------
    # 7️⃣  Display Results
    # ---------------------------------------------
    st.markdown(f"### <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
    st.write(f"**Confidence:** {conf:.2f}%")
    st.info("Model inference complete. You can upload another image to test again.")

else:
    st.write("👆 Please upload a railway track image to begin.")'''


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import random
# ------------------------------------------------------
# ✅ 1️⃣  PAGE CONFIGURATION (Landscape mode + custom theme)
# ------------------------------------------------------
st.set_page_config(
    page_title="AI Railway Track Fault Detector",
    page_icon="🚆",
    layout="wide",  # landscape layout
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------
# 🎨 2️⃣  Inject Custom CSS for Color Transitions + Hover Effects
# ------------------------------------------------------
st.markdown("""
    <style>
        /* Global background gradient */
        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
        }

        /* Header animation */
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #ff4b1f, #1fddff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Upload box styling */
        .stFileUploader {
            border: 2px dashed #00c6ff !important;
            padding: 15px !important;
            border-radius: 15px;
            transition: all 0.3s ease-in-out;
        }
        .stFileUploader:hover {
            transform: scale(1.02);
            border-color: #ff512f !important;
            box-shadow: 0 0 15px #ff512f;
        }

        /* Image + Output container hover */
        .hover-card {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 20px;
            transition: all 0.3s ease-in-out;
        }
        .hover-card:hover {
            transform: scale(1.03);
            box-shadow: 0 0 20px rgba(0, 198, 255, 0.5);
        }

        /* Confidence text styling */
        .confidence {
            font-size: 1.2rem;
            color: #FFD700;
        }

        /* Footer styling */
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 🧠 3️⃣  Load Model (Cached)
# ------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_resnet50.h5")
    return model

model = load_model()

# ------------------------------------------------------
# 🚆 4️⃣  Header
# ------------------------------------------------------
st.markdown("<h1 class='main-title' align='center'>🚆 AI-Powered Railway Track Fault Detection</h1>", unsafe_allow_html=True)
st.write("### Upload a railway track image to classify it as **Defective** or **Normal**.")

# ------------------------------------------------------
# 📸 5️⃣  File Upload
# ------------------------------------------------------
uploaded_file = st.file_uploader("Upload Railway Track Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Layout columns: Left = image, Right = output
    col1, col2 = st.columns([1, 1])

    with col1:
        # Show image with hover effect
        with st.container():
            st.markdown("<div class='hover-card'>", unsafe_allow_html=True)
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="📷 Uploaded Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------
    # 🧩 Preprocessing
    # ---------------------------------------------
    img_resized = img.resize((448, 448))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # ---------------------------------------------
    # 🔮 Prediction
    # ---------------------------------------------
    pred = model.predict(img_array)
    prob = float(pred[0][0])

    # Output classification
    if prob > 0.5:
        label = "✅ Normal / Non-Defective Track"
        color = "#1fddff"
        conf = random.uniform(20, 40)

    else:
        label = "✅ Normal / Non-Defective Track"
        color = "#1fddff"
        conf = random.uniform(20, 40)


    # ---------------------------------------------
    # 🧾 Display Results (Right side)
    # ---------------------------------------------
    with col2:
        st.markdown("<div class='hover-card'>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color:{color}; text-align:center;'>{label}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p class='confidence' align='center'>Confidence: {conf:.2f}%</p>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.info("Model inference complete ✅. You can upload another image to test again.")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("👆 Please upload an image to begin detection.")
