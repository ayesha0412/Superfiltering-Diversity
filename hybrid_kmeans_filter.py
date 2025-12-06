import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# ============ CONFIG ============
DATA_FILE = "alpaca_data_gpt2_data.json"          # IFD-annotated Alpaca
OUTPUT_DATA = "alpaca_data_kmeans_hybrid_10per.json"
META_FILE = "kmeans_hybrid_metadata.json"

TOP_IFD_PERCENT = 0.30      # use top 30% highest-IFD examples
FINAL_PERCENT = 0.10        # final dataset ≈ 10% of full size
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# your IFD key from sample
IFD_KEY = "ifd_ppl"


def main():
    # ---- 1. Load data with IFD ----
    print(f"📌 Loading IFD-annotated data from: {DATA_FILE}")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_total = len(data)
    print(f"Total samples: {n_total}")

    # ---- 2. Sort by IFD and select top X% ----
    print("📌 Sorting by IFD score (field:", IFD_KEY, ")...")
    for ex in data:
        if IFD_KEY not in ex:
            raise KeyError(f"Example missing '{IFD_KEY}' field. Check your JSON structure.")
        ex["_ifd_tmp"] = ex[IFD_KEY]

    data_sorted = sorted(data, key=lambda x: x["_ifd_tmp"], reverse=True)

    top_n = int(n_total * TOP_IFD_PERCENT)
    high_ifd = data_sorted[:top_n]
    print(f"➡️ Using top {TOP_IFD_PERCENT*100:.0f}% = {top_n} high-IFD examples")

    # clean temp key
    for ex in data_sorted:
        ex.pop("_ifd_tmp", None)

    # ---- 3. Compute embeddings for instructions ----
    print(f"📌 Loading embedding model: {EMBED_MODEL}")
    emb_model = SentenceTransformer(EMBED_MODEL)

    print("📌 Encoding instructions from high-IFD pool...")
    instructions = [ex["instruction"] for ex in high_ifd]
    embeddings = emb_model.encode(
        instructions,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # ---- 4. Run K-Means ----
    target_size = int(n_total * FINAL_PERCENT)
    if target_size <= 0:
        raise ValueError("Target size is 0; check FINAL_PERCENT.")

    print(f"📌 Running KMeans with k = {target_size} clusters...")
    kmeans = KMeans(n_clusters=target_size, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    # cluster size stats
    counts = np.bincount(cluster_ids, minlength=target_size)
    print("\n=== Cluster Size Stats ===")
    print(f"  Min size:  {counts.min()}")
    print(f"  Max size:  {counts.max()}")
    print(f"  Mean size: {counts.mean():.2f}")
    print(f"  Non-empty clusters: {(counts > 0).sum()} / {target_size}")

    # ---- 5. Select centroid-closest example per cluster ----
    print("\n📌 Selecting centroid-closest example per cluster...")
    selected_examples = []
    selected_indices = []

    for c in tqdm(range(target_size)):
        idx = np.where(cluster_ids == c)[0]
        if len(idx) == 0:
            continue

        cluster_embs = embeddings[idx]
        center = centers[c]

        dists = np.linalg.norm(cluster_embs - center, axis=1)
        best_pos = idx[np.argmin(dists)]
        selected_examples.append(high_ifd[best_pos])
        selected_indices.append(best_pos)

    print(f"✅ Final selected size: {len(selected_examples)}")

    # ---- 6. Save final dataset ----
    print(f"\n📁 Saving hybrid dataset to: {OUTPUT_DATA}")
    with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
        json.dump(selected_examples, f, indent=2, ensure_ascii=False)

    # ---- 7. Visualizations ----

    # 7a. PCA of embeddings (2D scatter)
    print("\n📊 Creating PCA visualization...")
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    centers_2d = pca.transform(centers)

    plt.figure(figsize=(10, 6))
    # all high-IFD points
    plt.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        s=8, alpha=0.2, label="High-IFD pool"
    )
    # selected points
    sel_idx_arr = np.array(selected_indices, dtype=int)
    plt.scatter(
        emb_2d[sel_idx_arr, 0], emb_2d[sel_idx_arr, 1],
        s=30, alpha=0.9, edgecolors="k", label="Selected (K-Means reps)"
    )
    # cluster centers
    plt.scatter(
        centers_2d[:, 0], centers_2d[:, 1],
        s=40, marker="x", label="Cluster centers"
    )

    plt.title("K-Means Clustering on High-IFD Instructions (PCA 2D)")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend()
    plt.tight_layout()

    vis_path = "kmeans_clusters_pca.png"
    plt.savefig(vis_path, dpi=300)
    plt.close()
    print(f"📁 Saved PCA cluster plot to: {vis_path}")

    # 7b. Cluster size histogram
    print("📊 Creating cluster size histogram...")
    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=20)
    plt.title("Distribution of Cluster Sizes (K-Means)")
    plt.xlabel("Cluster size (# examples)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    hist_path = "kmeans_cluster_sizes_hist.png"
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"📁 Saved cluster size histogram to: {hist_path}")

    # ---- 8. Save metadata ----
    meta = {
        "data_file": DATA_FILE,
        "output_data": OUTPUT_DATA,
        "num_total_examples": int(n_total),
        "num_high_ifd": int(top_n),
        "top_ifd_percent": TOP_IFD_PERCENT,
        "final_percent": FINAL_PERCENT,
        "num_clusters": int(target_size),
        "num_selected": int(len(selected_examples)),
        "cluster_size_min": int(counts.min()),
        "cluster_size_max": int(counts.max()),
        "cluster_size_mean": float(counts.mean()),
        "ifd_key": IFD_KEY,
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n📁 Saved metadata to: {META_FILE}")
    print("\n🎉 DONE: dataset + visuals + metadata generated.")


if __name__ == "__main__":
    main()
