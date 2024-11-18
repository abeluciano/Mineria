import pandas as pd
import numpy as np

# Paso 1: Cargar el archivo
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

file_path = "processed.cleveland.data"
heart = pd.read_csv(file_path, header=None, names=columns)

# Reemplazar valores faltantes
heart.replace("?", np.nan, inplace=True)  # Reemplazar '?' por NaN
heart.dropna(inplace=True)  # Eliminar filas con valores faltantes
heart = heart.astype(float)  # Convertir todas las columnas a float

# Paso 2: Realizar binning manual
def bin_values(value, bins, labels):
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return labels[i]
    return labels[-1]

thalach_bins = [70, 120, 160, 202]
thalach_labels = [0, 1, 2]
heart['thalach_binned'] = heart['thalach'].apply(lambda x: bin_values(x, thalach_bins, thalach_labels))

oldpeak_bins = [0, 1, 3, 6.2]
oldpeak_labels = [0, 1, 2]
heart['oldpeak_binned'] = heart['oldpeak'].apply(lambda x: bin_values(x, oldpeak_bins, oldpeak_labels))

# Paso 3: Seleccionar vector de características
X = heart[['cp', 'thalach_binned', 'ca', 'oldpeak', 'slope', 'thal']].values
y = heart['target'].values

# Paso 4: Escalar los datos manualmente
def scale_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std, mean, std

X_scaled, mean, std = scale_data(X)

# Paso 5: Calcular distancias euclidianas manualmente
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Seleccionar un punto manualmente
index_point = 1
query_point = X_scaled[index_point]
query_target = y[index_point]

# Calcular distancias a todos los demás puntos
distances = [
    (i, euclidean_distance(query_point, X_scaled[i]))
    for i in range(len(X_scaled)) if i != index_point
]

# Ordenar por distancia
distances.sort(key=lambda x: x[1])

# Crear DataFrame con los vecinos más cercanos
neighbors = pd.DataFrame({
    "Index": [idx for idx, _ in distances[:1]],  # Obtener los 5 vecinos más cercanos
    "Distance": [dist for _, dist in distances[:1]],
    "Target": [y[idx] for idx, _ in distances[:1]]
})

# Imprimir resultados
print(f"Punto consultado: Índice={index_point}, Target={query_target}")
print("\nVecino más cercano:")
print(neighbors)
