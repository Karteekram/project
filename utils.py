import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- PREDICTION -------------------
def predict_real_fake(model, img_tensor, device="cpu"):
    """
    Predict whether logo is REAL or FAKE
    Returns:
        pred_label (int): 0 = REAL, 1 = FAKE
        confidence (float): probability score
        probs (array): [real_prob, fake_prob]
    """
    model.eval()

    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)
        outputs = model(img_tensor)

        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred = np.argmax(probs)
        confidence = probs[pred]

    return pred, confidence, probs


# ------------------- EMBEDDING EXTRACTION -------------------
def get_embedding(model, img_tensor, device="cpu"):
    """
    Extract feature embedding from ViT model
    """
    model.eval()

    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # For timm ViT
        features = model.forward_features(img_tensor)

    return features.cpu().numpy()


# ------------------- TOP-K SIMILARITY -------------------
def get_top_k_matches(query_emb, database_embs, k=3):
    """
    Find top-k similar embeddings using cosine similarity
    Returns:
        indices (list)
        similarities (list)
    """
    sims = cosine_similarity(query_emb, database_embs)[0]

    top_k_idx = sims.argsort()[-k:][::-1]
    top_k_scores = sims[top_k_idx]

    return top_k_idx, top_k_scores


# ------------------- LABEL TO TEXT -------------------
def label_to_text(pred):
    """
    Convert numeric label to readable text
    """
    return "REAL" if pred == 0 else "FAKE"