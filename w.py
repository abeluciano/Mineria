import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

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

# Paso 2: Realizar binning automático con pd.qcut
# Para `thalach`
heart['thalach_binned'], thalach_bins = pd.qcut(heart['thalach'], q=3, retbins=True, labels=[0, 1, 2])

# Para `oldpeak`
heart['oldpeak_binned'], oldpeak_bins = pd.qcut(heart['oldpeak'], q=3, retbins=True, labels=[0, 1, 2])

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
    "Index": [idx for idx, _ in distances[:1]],
    "Distance": [dist for _, dist in distances[:1]],
    "Target": [y[idx] for idx, _ in distances[:1]]
})

print(f"Punto consultado: Índice={index_point}, Target={query_target}")
print("\nVecinos más cercanos:")
print(neighbors)

# Visualización del binning automático
sns.set(font_scale=1.25)
plt.rcParams["figure.figsize"] = (9, 6)

# Visualizar bins para 'thalach'
plt.figure(figsize=(9, 6))
sns.histplot(data=heart, x='thalach', bins=30, kde=False, color='blue', alpha=0.7)
for bin_edge in thalach_bins:
    plt.axvline(bin_edge, color='red', linestyle='--', ymax=0.95)
plt.xlabel('Thalach (frecuencia cardíaca máxima)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Thalach con Líneas de Binning Automático')
plt.show()

# Visualizar bins para 'oldpeak'
plt.figure(figsize=(9, 6))
sns.histplot(data=heart, x='oldpeak', bins=30, kde=False, color='green', alpha=0.7)
for bin_edge in oldpeak_bins:
    plt.axvline(bin_edge, color='red', linestyle='--', ymax=0.95)
plt.xlabel('Oldpeak (ST depression)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Oldpeak con Líneas de Binning Automático')
plt.show()
