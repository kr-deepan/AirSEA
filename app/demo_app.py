import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Oil Spill Detection", layout="centered")

st.title("🌊 Oil Spill Detection Demo")
st.write("Upload a SAR satellite chip (400×400) and classify **Oil Spill vs No Oil**.")

# ---------------------------
# Device setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"🖥️ Using device: **{device}**")

# ---------------------------
# Model setup
# ---------------------------
MODEL_PATH = "models/resnet18_baseline.pth"
model = models.resnet18(weights=None)

# Adjust last layer → 2 classes
model.fc = nn.Linear(model.fc.in_features, 2)

model_loaded = False
if os.path.exists(MODEL_PATH):
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        model_loaded = True
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
else:
    st.warning(f"⚠️ Model file not found at `{MODEL_PATH}`. Predictions disabled.")

# ---------------------------
# Image preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------------------
# Upload + Prediction
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload an image", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Convert grayscale SAR → RGB (3 channels) for ResNet
    if image.mode != "RGB":
        image = image.convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model_loaded:
        with torch.no_grad():
            img_tensor = transform(image).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            predicted = torch.argmax(probs).item()

            label = "🛢️ Oil Spill Detected" if predicted == 1 else "🌊 No Oil Spill"
            confidence = probs[predicted].item() * 100

            st.subheader(f"Prediction: {label}")
            st.write(f"**Confidence:** {confidence:.2f}%")
    else:
        st.info("Model not loaded. Place your trained weights at `models/resnet18_baseline.pth`.")
