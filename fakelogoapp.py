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
MODEL_ID = "1vn1Ilpm2dOK9zL3554_-N_SmJM8-2oi6"   
EMB_ID   = "1R0CJawIZZ_TUe6KXnOG7fkJXZ2mXCshg"
LABEL_ID = "1Cb0eczJeZMpw0czJLGlI8yzbLkLQIkTk"

# ------------------- DOWNLOAD FUNCTION (NO CACHE) -------------------
def download(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output):
        with st.spinner(f"Downloading {output}..."):
            gdown.download(url, output, quiet=False, fuzzy=True)

# ------------------- DOWNLOAD FILES -------------------
download(MODEL_ID, "model.pth")
download(EMB_ID, "brand_embeddings.npy")
download(LABEL_ID, "brand_labels.npy")

# ------------------- VALIDATION -------------------
def check(path, min_mb):
    if not os.path.exists(path):
        st.error(f"{path} missing ❌")
        st.stop()
    size = os.path.getsize(path) / (1024 * 1024)
    st.write(f"{path}: {round(size,2)} MB")
    if size < min_mb:
        st.error(f"{path} corrupted ❌")
        st.stop()

check("model.pth", 10)
check("brand_embeddings.npy", 1)
check("brand_labels.npy", 0.1)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
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

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

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