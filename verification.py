import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# ==============================
# PATHS (Change these as needed)
# ==============================
embeddings_path = "/kaggle/working/cattle1_Grams_test_embeddings.npy"  # Numpy file containing embeddings dict
pairs_csv_path = "/kaggle/working/pairs_test.csv"  # CSV with columns: img1,img2,label

# ==============================
# LOAD DATA
# ==============================
# Load embeddings dictionary
embeddings = np.load(embeddings_path, allow_pickle=True).item()

# Load pairs CSV
pairs_df = pd.read_csv(pairs_csv_path)

# ==============================
# COMPUTE COSINE SIMILARITIES
# ==============================
similarities = []
labels = []

for _, row in pairs_df.iterrows():
    img1, img2, label = row['img1'], row['img2'], row['label']
    emb1 = embeddings.get(img1)
    emb2 = embeddings.get(img2)

    if emb1 is None or emb2 is None:
        print(f"⚠ Warning: Missing embedding for {img1} or {img2}")
        continue

    # Reshape if needed (ensure 2D for cosine_similarity)
    emb1 = emb1.reshape(1, -1) if emb1.ndim == 1 else emb1
    emb2 = emb2.reshape(1, -1) if emb2.ndim == 1 else emb2

    # Compute cosine similarity
    sim = cosine_similarity(emb1, emb2)[0][0]
    similarities.append(sim)
    labels.append(label)

# Convert to numpy arrays
similarities = np.array(similarities)
labels = np.array(labels)

# ==============================
# FIND BEST THRESHOLD
# ==============================
best_acc = 0
best_thresh = 0.5

for thresh in np.linspace(0, 1, 101):  # thresholds from 0.00 to 1.00
    preds = (similarities >= thresh).astype(int)
    acc = accuracy_score(labels, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"🔍 Best Threshold: {best_thresh:.2f} | Accuracy: {best_acc:.4f}")

# ==============================
# FINAL METRICS
# ==============================
final_preds = (similarities >= best_thresh).astype(int)
acc = accuracy_score(labels, final_preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, final_preds, average='binary')
roc_auc = roc_auc_score(labels, similarities)

print(f"✅ Final Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
