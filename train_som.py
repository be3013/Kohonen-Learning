import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
import pickle

# Carregar e pré-processar o dataset MNIST
(X_train, y_train), (_, _) = mnist.load_data()
X_train = X_train.reshape(-1, 784)  # Flatten para 784 features

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Configuração da rede SOM
som_shape = (10, 10)  # Tamanho da grade da SOM
som = MiniSom(som_shape[0], som_shape[1], X_train.shape[1], sigma=1.0, learning_rate=0.5)

# Treinamento da SOM
print("Treinando a SOM...")
som.train_random(X_train, 1000)  # 1000 iterações

# Salvar a SOM e o scaler
with open("som_model.pkl", "wb") as som_file:
    pickle.dump(som, som_file)
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Modelo SOM e scaler salvos com sucesso!")
