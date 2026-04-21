import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
from sklearn.metrics.pairwise import cosine_similarity
import os
import gdown

# ------------------- DOWNLOAD FUNCTION -------------------
def download_file(url, output):
    if not os.path.exists(output):
        st.write(f"Downloading {output}...")
        gdown.download(url, output, quiet=False)

# 🔥 USE FULL LINKS
model_url = "https://drive.google.com/uc?id=1fr23oaG3AfRncEwUqoImNgIKKGmm0M3M"
emb_url   = "https://drive.google.com/uc?id=1Lb4Uf0mM5SZJUdATGp-VIuddDyyG1Pq0"
label_url = "https://drive.google.com/uc?id=1Cb0eczJeZMpw0czJLGlI8yzbLkLQIkTk"

download_file(model_url, "model.pth")
download_file(emb_url, "brand_embeddings.npy")
download_file(label_url, "brand_labels.npy")

# ------------------- LOAD MODEL -------------------
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

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
    st.write("Closest Match Index:", idx)
    st.write("Confidence:", sim[0][idx])