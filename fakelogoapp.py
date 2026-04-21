import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
from sklearn.metrics.pairwise import cosine_similarity
import os
import gdown

# ------------------- FILE IDS -------------------
MODEL_ID = "1MS-NCzgXxEPD0VnC2BWS4clrHgipQDxO"
EMB_ID   = "1Lb4Uf0mM5SZJUdATGp-VIuddDyyG1Pq0"
LABEL_ID = "1Cb0eczJeZMpw0czJLGlI8yzbLkLQIkTk"

# ------------------- DOWNLOAD (SAFE) -------------------
@st.cache_resource
def download_files():
    def download(file_id, output):
        url = f"https://drive.google.com/uc?id={file_id}"
        if not os.path.exists(output):
            with st.spinner(f"Downloading {output}..."):
                gdown.download(url, output, quiet=False, fuzzy=True)

    download(MODEL_ID, "model.pth")
    download(EMB_ID, "brand_embeddings.npy")
    download(LABEL_ID, "brand_labels.npy")

    return True

download_files()

# ------------------- VALIDATION -------------------
def validate_file(path, min_size_mb=5):
    if not os.path.exists(path):
        st.error(f"{path} missing ❌")
        st.stop()

    size = os.path.getsize(path) / (1024 * 1024)
    st.write(f"{path} size: {round(size,2)} MB")

    if size < min_size_mb:
        st.error(f"{path} corrupted ❌")
        st.stop()

validate_file("model.pth", 20)   # model must be large
validate_file("brand_embeddings.npy", 1)
validate_file("brand_labels.npy", 0.1)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
    state = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# ------------------- LOAD DATA -------------------
embeddings = np.load("brand_embeddings.npy")
labels = np.load("brand_labels.npy")

# ------------------- TRANSFORM -------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def get_embedding(img):
    with torch.no_grad():
        feat = model.forward_features(img.unsqueeze(0))
    return feat.numpy()

# ------------------- UI -------------------
st.title("Fake Logo Detection System")

file = st.file_uploader("Upload Image", type=["jpg","png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img)

    x = transform(img).unsqueeze(0)
    out = model(x)
    pred = torch.argmax(out).item()

    result = "REAL" if pred == 0 else "FAKE"

    emb = get_embedding(transform(img))
    sim = cosine_similarity(emb, embeddings)
    idx = np.argmax(sim)

    st.write("Prediction:", result)
    st.write("Confidence:", round(sim[0][idx]*100,2), "%")