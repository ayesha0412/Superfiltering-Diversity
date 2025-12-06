import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

# ================== CONFIG ==================
FULL_DATA = "data/alpaca_data.json"              # Full Alpaca dataset
IFD_DATA = "alpaca_data_gpt2_scores.jsonl"       # Your IFD scores from Step 1
OUTPUT_DATA = "alpaca_data_kmeans_hybrid_10per.json"

TOP_IFD_PERCENT = 0.30     # use top 30% hardest samples
TARGET_PERCENT = 0.10      # create final 10% dataset
# =============================================

def load_ifd_scores():
    scores = {}
    with open(IFD_DATA, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            scores[obj["id"]] = obj["ifd"]
    return scores


def main():
    print("📌 Loading Alpaca dataset...")
    with open(FULL_DATA, "r", encoding="utf-8") as f:
        alpaca = json.load(f)

    print("📌 Loading IFD scores (informativeness/difficulty)...")
    scores = load_ifd_scores()

    # add scores into alpaca list
    for item in alpaca:
        item_id = item.get("id", None)
        if item_id in scores:
            item["ifd"] = scores[item_id]
        else:
            item["ifd"] = 0

    # =========================================================
    # STEP 1 — Select top 30% with highest IFD (high difficulty)
    # =========================================================
    print("📌 Selecting top 30% high-IFD instructions...")
    alpaca_sorted = sorted(alpaca, key=lambda x: x["ifd"], reverse=True)
    top_n = int(len(alpaca_sorted) * TOP_IFD_PERCENT)

    high_ifd_data = alpaca_sorted[:top_n]
    print(f"➡️ Selected {top_n} high-IFD samples")

    # =========================================================
    # STEP 2 — Compute embeddings
    # =========================================================
    print("📌 Computing embeddings using MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    instructions = [x["instruction"] for x in high_ifd_data]
    embeddings = model.encode(
        instructions,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # =========================================================
    # STEP 3 — K-means clustering on embeddings
    # =========================================================
    target_size = int(len(alpaca) * TARGET_PERCENT)  # 10% of total dataset
    print(f"📌 Running K-means clustering with {target_size} clusters...")

    kmeans = KMeans(n_clusters=target_size, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    # =========================================================
    # STEP 4 — Visualize clusters (2D PCA)
    # =========================================================
    print("📊 Creating 2D PCA visualization of clusters...")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_ids, cmap="tab20", s=8)
    plt.scatter(
        pca.transform(centers)[:, 0],
        pca.transform(centers)[:, 1],
        c='red',
        s=50,
        marker='x',
        label='Centroids'
    )

    plt.title("K-means Clusters of Instruction Embeddings (Hybrid IFD → KMeans)")
    plt.legend()
    plt.savefig("kmeans_clusters_visualization.png", dpi=300)
    plt.close()
    print("📁 Saved visualization: kmeans_clusters_visualization.png")

    # =========================================================
    # STEP 5 — Pick centroid-closest sample from each cluster
    # =========================================================
    print("📌 Selecting centroid-closest samples...")
    selected = []

    for cluster_idx in tqdm(range(target_size)):
        indices = np.where(cluster_ids == cluster_idx)[0]
        if len(indices) == 0:
            continue

        cluster_embs = embeddings[indices]
        centroid = centers[cluster_idx]

        distances = np.linalg.norm(cluster_embs - centroid, axis=1)
        best_idx = indices[np.argmin(distances)]

        selected.append(high_ifd_data[best_idx])

    # =========================================================
    # STEP 6 — Save final hybrid dataset
    # =========================================================
    print(f"📁 Saving filtered dataset to: {OUTPUT_DATA}")
    with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print(f"🎉 Done! Created a {len(selected)}-sample Hybrid IFD + K-means dataset.")
    print("👉 You can now train GPT-2 or LLaMA on this new dataset.")


if __name__ == "__main__":
    main()
