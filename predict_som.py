import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import pickle
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Carregar o dataset MNIST de teste
(_, _), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(-1, 784)  # Flatten para 784 features

# Carregar a SOM e o scaler
with open("som_model.pkl", "rb") as som_file:
    som = pickle.load(som_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Normalizar os dados de teste
X_test = scaler.transform(X_test)

# Atribuir cada vetor de teste a um nó da SOM
mapped_test = np.array([som.winner(x) for x in X_test])

# Carregar o dataset de treino para associar os nós às classes
(X_train, y_train), _ = mnist.load_data()
X_train = X_train.reshape(-1, 784)  # Flatten para 784 features
X_train = scaler.transform(X_train)
mapped_train = np.array([som.winner(x) for x in X_train])

# Conversão das coordenadas de nó para índices únicos para cada classe
train_nodes = [n[0] * som_shape[1] + n[1] for n in mapped_train]

node_labels = {}
for t_node, label in zip(train_nodes, y_train):
    if t_node in node_labels:
        node_labels[t_node].append(label)
    else:
        node_labels[t_node] = [label]
node_labels = {k: np.bincount(v).argmax() for k, v in node_labels.items()}

# Predição com SOM
test_nodes = [n[0] * som_shape[1] + n[1] for n in mapped_test]
y_pred_test = [node_labels[node] if node in node_labels else -1 for node in test_nodes]

# Cálculo das métricas
test_accuracy = accuracy_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test, average='macro')

# Exibição das métricas
metrics = pd.DataFrame({
    "Dataset": ["Test"],
    "Accuracy": [test_accuracy],
    "F1 Score": [test_f1]
})

print(metrics)

# Visualizar o Mapa da SOM
plt.figure(figsize=(10, 10))
for i, (x, label) in enumerate(zip(X_test, y_test)):
    w = som.winner(x)
    plt.text(w[0] + .5, w[1] + .5, str(label), color=plt.cm.tab10(label / 10.), fontdict={'weight': 'bold', 'size': 9})
plt.xlim([0, som_shape[0]])
plt.ylim([0, som_shape[1]])
plt.title("Mapa Auto-Organizado de Dígitos MNIST")
plt.show()
