import pandas as pd
import math

# Paso 1: Cargar el archivo
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

file_path = "processed.cleveland.data"
heart = pd.read_csv(file_path, header=None, names=columns)

# Reemplazar valores faltantes
heart.replace("?", None, inplace=True)  # Reemplazar '?' por None
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
X = heart[['cp', 'thalach_binned', 'ca', 'oldpeak', 'slope', 'thal']].values.tolist()
y = heart['target'].tolist()

# Paso 4: Escalar los datos manualmente
def calculate_mean_and_std(data):
    means = []
    stds = []
    for col in range(len(data[0])):
        col_values = [row[col] for row in data]
        mean = sum(col_values) / len(col_values)
        variance = sum((x - mean) ** 2 for x in col_values) / len(col_values)
        std = math.sqrt(variance)
        means.append(mean)
        stds.append(std)
    return means, stds

def scale_data(data, means, stds):
    scaled_data = []
    for row in data:
        scaled_row = [(row[i] - means[i]) / stds[i] for i in range(len(row))]
        scaled_data.append(scaled_row)
    return scaled_data

means, stds = calculate_mean_and_std(X)
X_scaled = scale_data(X, means, stds)

# Paso 5: Calcular distancias euclidianas manualmente
def euclidean_distance(point1, point2):
    return math.sqrt(sum((point1[i] - point2[i]) ** 2 for i in range(len(point1))))

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
print("\nVecinos más cercanos:")
print(neighbors)
