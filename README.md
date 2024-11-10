# Kohonen-Learning
https://colab.research.google.com/drive/1rx2DSsz8Pfk61b04QtvomQaFU85_FyTc?usp=sharing
https://drive.google.com/drive/folders/1dTZQBX3X1thLCA_BNFXrs--9fArzOt_v?usp=sharing

Este projeto demonstra o uso de uma Rede Auto-Organizada para organizar e classificar dados do conjunto MNIST. A implementação é dividida em dois scripts: um para o treinamento da rede e salvamento do modelo, e outro para carregamento do modelo treinado e predição.

## Estrutura do Projeto
- train_som.py: Script para treinamento e salvamento do modelo SOM.
- predict_som.py: Script para carregar o modelo SOM e realizar predições nos dados de teste.
- som_model.pkl: Arquivo que salva o modelo SOM treinado.
- scaler.pkl: Arquivo que salva o modelo de normalização dos dados.

## 1. Treinamento e Salvamento do Modelo (train_som.py)
- Carrega e normaliza os dados do conjunto MNIST.
- Configura e treina uma rede SOM com uma grade de 10x10 nós.
- Salva o modelo SOM treinado e o modelo de normalização dos dados em arquivos .pkl para uso posterior.
- Após o treinamento, os arquivos som_model.pkl e scaler.pkl serão gerados na pasta atual.

## 2. Carregamento do Modelo e Predição

- Carrega o modelo SOM e o normalizador salvos.
- Normaliza os dados de teste do conjunto MNIST.
- Associa cada vetor de teste ao nó vencedor na SOM.
- Mapeia as classes para os nós da SOM com base nos dados de treino.
- Calcula métricas de precisão (Accuracy) e pontuação F1 para os dados de teste.
- Exibe o mapa SOM com a representação dos dígitos do MNIST.

## Observações
- A rede SOM é não-supervisionada e, portanto, a precisão e a pontuação F1 servem apenas como uma avaliação aproximada, mapeando as classes mais frequentes para cada nó.
