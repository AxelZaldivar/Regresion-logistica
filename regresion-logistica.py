import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
file_path = "Employee.csv"
df = pd.read_csv(file_path)

# One-hot encoding para variables categóricas
categorical_columns = ['Education', 'City', 'Gender']
df_encoded = pd.get_dummies(df, columns=categorical_columns)
df_encoded['EverBenched'] = df_encoded['EverBenched'].map({'No': 0, 'Yes': 1})

# Ajustar los parámetros de entrenamiento y prueba, y definir las matrices
train_idx = np.random.rand(len(df)) < 0.8
train_data = df_encoded[train_idx]
test_data = df_encoded[~train_idx]
X_train = train_data.drop('LeaveOrNot', axis=1).values.astype(float)
y_train = train_data['LeaveOrNot'].values.astype(float)
X_test = test_data.drop('LeaveOrNot', axis=1).values.astype(float)
y_test = test_data['LeaveOrNot'].values.astype(float)

# Normalizar las características
mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)
X_train = (X_train - mean_X) / std_X
X_test = (X_test - mean_X) / std_X

# Inicialización de los parámetros del modelo
theta = np.random.randn(X_train.shape[1]) * 0.01
learning_rate = 0.01
iterations = 1000
lambda_ = 0.1

# Descenso del gradiente con regularización L2
m = len(y_train)
cost_history = np.zeros(iterations)

for i in range(iterations):
    h = 1 / (1 + np.exp(-(X_train @ theta)))
    gradient = X_train.T @ (h - y_train) / m
    regularization_term = lambda_ / m * np.concatenate(([0], theta[1:]))
    gradient += regularization_term
    theta -= learning_rate * gradient

    # Función Sigmoide
    m = len(y_train)
    h = 1 / (1 + np.exp(-(X_train @ theta)))
    epsilon = 1e-10
    cost = -1 / m * (y_train @ np.log(h + epsilon) + (1 - y_train) @ np.log(1 - h + epsilon))
    regularization_term = lambda_ / (2 * m) * np.sum(theta[1:]**2)
    cost += regularization_term
    cost_history[i] = cost

# Evaluar el modelo en los datos de prueba
y_pred_prob = 1 / (1 + np.exp(-(X_test @ theta)))
y_pred = (y_pred_prob >= 0.5).astype(int)
accuracy = np.mean(y_pred == y_test)

# Visualizar la función de costo a lo largo de las iteraciones
plt.plot(range(1, iterations + 1), cost_history, color='red', label='Función de costo')
plt.scatter(range(1, iterations + 1), cost_history, color='blue', s=40, label='Puntos de datos')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.title('Regresión logistica\nPrecisión del modelo: ' + str(accuracy))
plt.show()