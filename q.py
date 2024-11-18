import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

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

# Paso 2: Realizar binning en variables seleccionadas
# Asegurarse de que no haya NaN antes de realizar el binning
heart['thalach_binned'] = pd.cut(
    heart['thalach'], bins=[70, 120, 160, 202], labels=[0, 1, 2], right=False
)

heart['oldpeak_binned'] = pd.cut(
    heart['oldpeak'], bins=[0, 1, 3, 6.2], labels=[0, 1, 2], right=False
)

# Verificar si hay valores NaN después del binning
if heart['thalach_binned'].isna().any() or heart['oldpeak_binned'].isna().any():
    print("Hay valores NaN en las columnas binned.")

# Paso 3: Convertir los valores binned a enteros (si no hay NaN)
heart['thalach_binned'] = heart['thalach_binned'].fillna(0).astype(int)
heart['oldpeak_binned'] = heart['oldpeak_binned'].fillna(0).astype(int)

# Paso 4: Seleccionar vector de características (con binned variables)
X = heart[['cp', 'thalach_binned', 'ca', 'oldpeak', 'slope', 'thal']]  # Vector reducido con binned variables
y = heart["target"]  # Variable objetivo

# Paso 5: Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 6: Configurar k-NN
knn = NearestNeighbors(n_neighbors=2, metric="euclidean")
knn.fit(X_scaled)

# Paso 7: Seleccionar un punto manualmente
index_point = 1  # Cambia este índice para seleccionar un punto específico
query_point = X_scaled[index_point].reshape(1, -1)
query_target = y.iloc[index_point]

# Paso 8: Encontrar vecinos más cercanos
distances, indices = knn.kneighbors(query_point)

# Paso 9: Crear DataFrame con resultados
neighbors = pd.DataFrame({
    "Index": indices[0][1:],  # Excluir el índice 0
    "Distance": distances[0][1:],  # Excluir la distancia 0
    "Target": y.iloc[indices[0][1:]].values  # Excluir su target
})

# Añadir las características originales al DataFrame de vecinos
#neighbor_details = heart.iloc[indices[0]].reset_index(drop=True)
neighbors_full = pd.concat([neighbors], axis=1)

# Imprimir resultados
print(f"Punto consultado: Índice={index_point}, Target={query_target}")
#print("\nValores del punto consultado:")
#print(heart.iloc[index_point])

print("\nVecino más cercano:")
print(neighbors_full)
