import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1) Carga de datos
# ----------------------------
path = "dataset_no_supervisado.csv"
df = pd.read_csv(path)

features = [
    "pasajeros_promedio",
    "tiempo_espera_min",
    "velocidad_promedio_kmh",
    "num_transbordos_frecuentes",
]

X = df[features].copy()

# ----------------------------
# 2) Preprocesamiento
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 3) Método del codo y silueta
# ----------------------------
ks = range(2, 11)
inertia = []
silhouette_values = []

for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)
    inertia.append(model.inertia_)
    silhouette_values.append(silhouette_score(X_scaled, labels))

best_k_silhouette = ks[np.argmax(silhouette_values)]

# Gráfico del codo
plt.figure(figsize=(8, 5))
plt.plot(list(ks), inertia, marker="o", linewidth=2, color="royalblue")
plt.title("Método del codo (Elbow Method)")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia dentro del cluster")
plt.grid(True, linestyle="--", alpha=0.4)
plt.xticks(list(ks))
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=200)
plt.close()

# Gráfico de silueta
plt.figure(figsize=(8, 5))
plt.plot(list(ks), silhouette_values, marker="o", linewidth=2, color="darkgreen")
plt.axvline(best_k_silhouette, color="red", linestyle="--", label=f"Mejor k por silueta: {best_k_silhouette}")
plt.title("Coeficiente de silueta por número de clusters")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True, linestyle="--", alpha=0.4)
plt.xticks(list(ks))
plt.legend()
plt.tight_layout()
plt.savefig("silhouette_scores.png", dpi=200)
plt.close()

# ----------------------------
# 4) Elección del número de clusters
# ----------------------------
# El codo muestra una desaceleración clara alrededor de 5-6. La silueta alcanza su mejor valor
# en 8, pero el crecimiento es marginal respecto a 6 y 7, y 8 genera grupos más fragmentados.
# Por ello se elige k = 6 porque equilibra calidad, interpretabilidad y estabilidad.
k_final = 6

model = KMeans(n_clusters=k_final, random_state=42, n_init=30)
labels = model.fit_predict(X_scaled)

# Resumen de centroides en unidades originales
centroids_original = scaler.inverse_transform(model.cluster_centers_)
centroid_df = pd.DataFrame(centroids_original, columns=features)
centroid_df["cluster"] = range(k_final)

print("Valores de inercia por k:")
for k, value in zip(ks, inertia):
    print(f"k={k}: {value:.2f}")

print("\nValores de silueta por k:")
for k, value in zip(ks, silhouette_values):
    print(f"k={k}: {value:.4f}")

print(f"\nMejor k por silueta: {best_k_silhouette}")
print(f"k elegido para el análisis final: {k_final}")

print("\nCentroides en unidades originales:")
print(centroid_df.round(2).to_string(index=False))

# ----------------------------
# 5) Visualización 2D (PCA)
# ----------------------------
reducer = PCA(n_components=2)
X_pca = reducer.fit_transform(X_scaled)
centroid_pca = reducer.transform(model.cluster_centers_)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=45, alpha=0.75)
plt.scatter(centroid_pca[:, 0], centroid_pca[:, 1], c="red", marker="X", s=180, linewidths=1.5, label="Centroides")
plt.title("Agrupamiento K-Means (PCA 2D)")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.colorbar(scatter, label="Cluster")
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_clusters_pca.png", dpi=200)
plt.close()

# ----------------------------
# 6) Interpretación de clusters
# ----------------------------
cluster_profile = df.copy()
cluster_profile["cluster"] = labels
summary = cluster_profile.groupby("cluster")[features].mean().round(2)
print("\nPerfil promedio por cluster:")
print(summary)

print("\nInterpretación sugerida:")
for cluster_id in range(k_final):
    row = summary.loc[cluster_id]
    if row["pasajeros_promedio"] >= 80 and row["tiempo_espera_min"] >= 15:
        label = "Paradas críticas / congestionadas"
    elif row["velocidad_promedio_kmh"] >= 25 and row["num_transbordos_frecuentes"] <= 1:
        label = "Paradas de flujo eficiente"
    elif row["pasajeros_promedio"] <= 40 and row["tiempo_espera_min"] <= 10:
        label = "Paradas de bajo flujo"
    else:
        label = "Paradas con flujo medio y transbordos moderados"
    print(f"Cluster {cluster_id}: {label} | promedio = {row.to_dict()}")

print("\nArchivos generados:")
print("- elbow_method.png")
print("- silhouette_scores.png")
print("- kmeans_clusters_pca.png")
