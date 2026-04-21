import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
from sklearn.metrics.pairwise import cosine_similarity
import os
import gdown

# ------------------- FORCE DELETE OLD FILES -------------------
for f in ["model.pth", "brand_embeddings.npy", "brand_labels.npy"]:
    if os.path.exists(f):
        os.remove(f)

# ------------------- DOWNLOAD FUNCTION -------------------
def download_file(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    st.write(f"Downloading {output}...")
    gdown.download(url, output, quiet=False, fuzzy=True)

# ------------------- DOWNLOAD FILES -------------------
download_file("1fr23oaG3AfRncEwUqoImNgIKKGmm0M3M", "model.pth")
download_file("1Lb4Uf0mM5SZJUdATGp-VIuddDyyG1Pq0", "brand_embeddings.npy")
download_file("1Cb0eczJeZMpw0czJLGlI8yzbLkLQIkTk", "brand_labels.npy")

# ------------------- DEBUG CHECK -------------------
st.write("Model file size:", os.path.getsize("model.pth"))

with open("model.pth", "rb") as f:
    first_bytes = f.read(50)
    st.write("First bytes:", first_bytes)

# ------------------- VALIDATION -------------------
if os.path.getsize("model.pth") < 1000000:
    st.error("❌ Model file is corrupted or not downloaded properly")
    st.stop()

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

# ------------------- EMBEDDING FUNCTION -------------------
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