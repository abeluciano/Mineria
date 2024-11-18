import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Paso 1: Cargar el archivo
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Leer el archivo con pandas
file_path = "processed.cleveland.data"
heart = pd.read_csv(file_path, header=None, names=columns)

# Reemplazar valores faltantes
heart.replace("?", pd.NA, inplace=True)  # Reemplazar '?' por NaN
heart = heart.dropna()  # Eliminar filas con valores faltantes
heart = heart.astype(float)  # Convertir todas las columnas a float

# Paso 2: Realizar Fixed-width binning en variables seleccionadas
def fixed_width_binning(series, n_bins):
    """Divide la serie en `n_bins` intervalos de igual tamaño."""
    min_val = series.min()
    max_val = series.max()
    bin_width = (max_val - min_val) / n_bins
    bins = [min_val + i * bin_width for i in range(n_bins + 1)]
    return pd.cut(series, bins=bins, labels=list(range(n_bins)), include_lowest=True)

# Aplicar Fixed-width binning a 'thalach' y 'oldpeak'
n_bins_thalach = 3
n_bins_oldpeak = 3

heart['thalach_binned'] = fixed_width_binning(heart['thalach'], n_bins_thalach)
heart['oldpeak_binned'] = fixed_width_binning(heart['oldpeak'], n_bins_oldpeak)

# Verificar si hay valores NaN después del binning
if heart['thalach_binned'].isna().any() or heart['oldpeak_binned'].isna().any():
    print("Hay valores NaN en las columnas binned.")

# Convertir los valores binned a enteros
heart['thalach_binned'] = heart['thalach_binned'].astype(int)
heart['oldpeak_binned'] = heart['oldpeak_binned'].astype(int)

# Paso 3: Seleccionar vector de características (con binned variables)
X = heart[['cp', 'thalach_binned', 'ca', 'oldpeak', 'slope', 'thal']]  # Vector reducido con binned variables
y = heart["target"]  # Variable objetivo

# Paso 4: Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 5: Configurar k-NN
knn = NearestNeighbors(n_neighbors=2, metric="euclidean")
knn.fit(X_scaled)

# Paso 6: Seleccionar un punto manualmente
index_point = 1  # Cambia este índice para seleccionar un punto específico
query_point = X_scaled[index_point].reshape(1, -1)
query_target = y.iloc[index_point]

# Paso 7: Encontrar vecinos más cercanos
distances, indices = knn.kneighbors(query_point)

# Paso 8: Crear DataFrame con resultados
neighbors = pd.DataFrame({
    "Index": indices[0][1:],  # Excluir el índice 0
    "Distance": distances[0][1:],  # Excluir la distancia 0
    "Target": y.iloc[indices[0][1:]].values  # Excluir su target
})

# Añadir las características originales al DataFrame de vecinos
neighbors_full = pd.concat([neighbors], axis=1)

# Imprimir resultados
print(f"Punto consultado: Índice={index_point}, Target={query_target}")
print("\nVecino más cercano:")
print(neighbors_full)
