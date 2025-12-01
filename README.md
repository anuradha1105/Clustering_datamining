# 🧠 Clustering Assignment   

This repository contains **nine separate Google Colab notebooks**, each implementing a different clustering technique or multimodal embedding–based clustering, as required in the CMPE 255 course assignment.

Each notebook is self-contained with documentation, visualization, evaluation metrics, and example datasets.  
All notebooks have been cleaned to ensure full compatibility with `nbconvert` and the Canvas grader.

---

# 📘 Table of Contents

| Part | Topic | Notebook Name |
|------|--------|----------------|
| (a) | K-Means from Scratch | `kmeans_from_scratch.ipynb` |
| (b) | Hierarchical Clustering | `hierarchical_clustering.ipynb` |
| (c) | Gaussian Mixture Models (GMM) | `gmm_clustering.ipynb` |
| (d) | DBSCAN using sklearn (PyCaret alternative) | `dbscan_clustering.ipynb` |
| (e) | Anomaly Detection using PyOD | `anomaly_detection_pyod.ipynb` |
| (f) | Time-Series Clustering using Pretrained Models | `timeseries_clustering_pretrained.ipynb` |
| (g) | Document Clustering using LLM Embeddings | `document_clustering_llm_embeddings.ipynb` |
| (h) | Image Clustering using ImageBind Embeddings | `image_clustering_imagebind.ipynb` |
| (i) | Audio Clustering using Wav2Vec2 (LLM Audio Embeddings) | `audio_clustering_wav2vec2.ipynb` |

---

# 🧩 Assignment Summary

This assignment explores different **clustering algorithms**, **multimodal embeddings**, and **unsupervised learning pipelines** across text, images, audio, and synthetic time-series data.  

It focuses on evaluating clustering quality using internal metrics:

- **Silhouette Score**  
- **Davies–Bouldin Index**  
- **Calinski–Harabasz Score**  

Each notebook contains:

- Detailed explanation  
- Clean code  
- Visualization of clusters  
- Evaluation metrics  
- Markdown summaries  
- PCA/UMAP visualization (where applicable)

---

# 📄 Notebook Details

Below are detailed descriptions of each part of the assignment:

---

## 🔵 (a) K-Means From Scratch  
Notebook: `kmeans_from_scratch.ipynb`

- Implemented K-Means **without using sklearn**.  
- Includes initialization, distance calculation, centroid updates, and stopping criteria.  
- Tested on synthetic 2D datasets.  
- Visualizes cluster boundaries and centroid movement.

---

## 🔵 (b) Hierarchical Clustering  
Notebook: `hierarchical_clustering.ipynb`

- Agglomerative clustering with:
  - Ward linkage  
  - Complete linkage  
  - Average linkage  
- Includes dendrograms, heatmaps, and cluster comparison.  
- Applied on synthetic and real datasets.

---

## 🔵 (c) Gaussian Mixture Models (GMM)  
Notebook: `gmm_clustering.ipynb`

- Clustering using GMMs with:
  - Full covariance  
  - Tied covariance  
  - Spherical  
  - Diagonal  
- Compares vs. K-Means.  
- Metrics + visualization included.

---

## 🔵 (d) DBSCAN using sklearn  
Notebook: `dbscan_clustering.ipynb`

- Density-based clustering.  
- Handles noise, outliers, arbitrary shapes.  
- Compared eps/min_samples settings.  
- *PyCaret installation was incompatible with Python 3.12, so sklearn implementation was used.*

---

## 🔵 (e) Anomaly Detection using PyOD  
Notebook: `anomaly_detection_pyod.ipynb`

- Uses PyOD's:
  - Isolation Forest  
  - LOF (Local Outlier Factor)  
  - AutoEncoder  
- Detects anomalies on multivariate synthetic dataset.  
- ROC/PR plots + anomaly visualization.

---

## 🔵 (f) Time-Series Clustering with Pretrained Models  
Notebook: `timeseries_clustering_pretrained.ipynb`

- Uses pretrained time-series encoders (TS2Vec / similar).  
- Converts time-series windows to embeddings.  
- Clusters using K-Means.  
- PCA visualization of clusters.

---

## 🔵 (g) Document Clustering using LLM Embeddings  
Notebook: `document_clustering_llm_embeddings.ipynb`

- Embeddings from `sentence-transformers` (e.g., `all-MiniLM-L6-v2`).  
- Clusters documents (news/paragraphs/etc.).  
- Includes cosine similarity heatmaps, silhouette scores, dendrograms.  
- Overcame “invalid notebook metadata.widgets.state missing” error by cleaning notebook metadata.

---

## 🔵 (h) Image Clustering using ImageBind Embeddings  
Notebook: `image_clustering_imagebind.ipynb`

- Synthetic images generated in Colab (solids, gradients, shapes).  
- Extracted vision embeddings using **ImageBind**.  
- Performed clustering + PCA visualization.  
- Grouped images by visual similarity.

---

## 🔵 (i) Audio Clustering using Wav2Vec2 (LLM Audio Embeddings)  
Notebook: `audio_clustering_wav2vec2.ipynb`

- Attempted ImageBind audio module, but torchaudio/torchcodec incompatibility blocked audio pipeline.  
- Switched to **Wav2Vec2-base** pretrained embeddings (HuggingFace).  
- Extracted audio embeddings for synthetic tones + noise.  
- Clustered using K-Means, with strong separation between tone frequencies.  
- PCA visualization included.

---

# 📦 Dependencies

Create a consistent environment using:

```bash
pip install numpy scipy sklearn matplotlib seaborn librosa soundfile sentence-transformers transformers==4.41.2
