import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import classification_report




# Leitura do arquivo Parquet
file_path = 'C:\\Users\\Andre\\Desktop\\GBM - dataset_SIN492\\Machine Learning using GBM algorithm\\dataset_SIN492.parquet'
df = pd.read_parquet(file_path)

# Convertendo para CSV
csv_path = 'C:\\Users\\Andre\\Desktop\\GBM - dataset_SIN492\\Machine Learning using GBM algorithm\\dataset_SIN492.csv'
df.to_csv(csv_path, index=False)

df.head()
df.info()


##########################################################################
######################## PRE TRATAMENTO DOS DADOS ########################
##########################################################################

#### Conversão para dados do mesmo tipo

# Lista de colunas a serem convertidas para float
colunas_para_converter = ['feature0', 'feature2', 'feature3', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14']

# Convertendo colunas para float, tratando possiveis erros
for coluna in colunas_para_converter:
    df[coluna] = pd.to_numeric(df[coluna], errors='coerce')

# Convertendo todas as colunas para float e verificando os tipos de dados e convertendo as colunas para float
df = df.astype(float)
print(df.dtypes)
df = df.apply(pd.to_numeric, errors='coerce')

#### Tratamento de Duplicatas

# Verificando duplicatas em todo o DataFrame e imprimindo as linhas duplicadas
duplicatas_totais = df.duplicated()
linhas_duplicadas = df[duplicatas_totais]
print("Linhas duplicadas:")
print(linhas_duplicadas)

# Verificando duplicatas em colunas específicas
colunas_para_verificar = ['feature0', 'feature2', 'feature3', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14']
duplicatas_colunas = df.duplicated(subset=colunas_para_verificar)

# Imprimindo as linhas duplicadas nas colunas específicas
linhas_duplicadas_colunas = df[duplicatas_colunas]
print("Linhas duplicadas nas colunas específicas:")
print(linhas_duplicadas_colunas)

# Eliminar todas as linhas duplicadas
df_sem_duplicatas = df.drop_duplicates()

# Imprimir o DataFrame após a remoção de duplicatas
print("DataFrame após a remoção de todas as linhas duplicadas:")
print(df_sem_duplicatas)

### Verificando colunas com possiveis dados faltantes

# Verificando colunas com dados faltantes
colunas_com_faltantes = df.columns[df.isnull().any()]

# Imprimindo as colunas com dados faltantes
print("Colunas com dados faltantes:")
print(colunas_com_faltantes)

df_sem_faltantes = df.dropna(axis=1)

# Imprimindo o DataFrame após a remoção
print("DataFrame após a remoção de colunas com dados faltantes:")
print(df_sem_faltantes)

df.head()

#### Plotando a matriz de correlacao com a variavel target(Não Classe e Classe)

# Matriz de correlação fornecida
correlation_matrix = df.corr()

# Selecionando correlações com o 'target'
correlation_with_target = correlation_matrix.iloc[-1, :-1]

# Criando um segundo DataFrame para visualização
df_correlation_with_target = pd.DataFrame({'Feature': correlation_with_target.index, 'Correlation with Target': correlation_with_target.values})

# Ordenando o DataFrame por correlação em ordem decrescente
df_correlation_with_target = df_correlation_with_target.sort_values(by='Correlation with Target', ascending=False)

# Plotando a correlação das características com o 'target'
plt.figure(figsize=(10, 7))
sns.barplot(x='Correlation with Target', y='Feature', data=df_correlation_with_target, palette='viridis')
plt.title('Correlação das características com o alvo')
plt.show()


#### Plotando o mapa de calor para a correlacao das variaveis

# Calculando a matriz de correlação e configurando o tamanho da figura
matriz_correlacao = df.corr()
plt.figure(figsize=(12, 10))

# Criar um mapa de calor com seaborn e definindo titulo
sns.heatmap(matriz_correlacao, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title('Matriz de Correlação entre Variáveis')

# Plotando mapa de calor
plt.show()


##########################################################################
########## CRIACAO DO MODELO GBM PARA ITERAÇÃO DAS FEATURES ##############
##########################################################################

# Supondo que 'df' é o seu DataFrame
# Substitua 'df' pelo nome real do seu DataFrame
seu_dataframe = df

# Substitua 'target' pelo nome da coluna que contém seus rótulos
X = seu_dataframe.drop('target', axis=1)
y = seu_dataframe['target']

# Criar um modelo GBM
modelo_gbm = GradientBoostingClassifier()
modelo_gbm.fit(X, y)

# Obter a importância das características
importancias = modelo_gbm.feature_importances_

# Criar um DataFrame para visualização
df_importancias = pd.DataFrame({'Feature': X.columns, 'Importância': importancias})

# Ordenar o DataFrame por importância em ordem decrescente
df_importancias = df_importancias.sort_values(by='Importância', ascending=False)

# Visualizar a importância das características
plt.figure(figsize=(10, 7))
sns.barplot(x='Importância', y='Feature', data=df_importancias, palette='viridis')
plt.title('Importância das características - GBM')
plt.show()

# TODO -> Drop das caracteristicas mais irrelevantes



##########################################################################
####################### TREINANDO O MODELO - GBM #########################
##########################################################################

####

# Substitua 'df' pelo seu DataFrame
X = df.drop('target', axis=1)
y = df['target']

# Dividir os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo
modelo_gbm = GradientBoostingClassifier()

# Treinar o modelo
modelo_gbm.fit(X_treino, y_treino)

# Fazer previsões
previsoes = modelo_gbm.predict(X_teste)

# Calcular métricas
report = classification_report(y_teste, previsoes)
print(report)

# Gerar matriz de confusão
matriz_confusao = confusion_matrix(y_teste, previsoes)

# Visualizar a matriz de confusão com seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt="d", cmap="Blues", xticklabels=['Não Fraude', 'Fraude'], yticklabels=['Não Fraude', 'Fraude'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Definir os parâmetros que deseja ajustar
parametros = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.00001, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Criar um modelo GBM
modelo_gbm = GradientBoostingClassifier()

# Criar uma grade de parâmetros
grid = ParameterGrid(parametros)



# Inicializar a barra de progresso
total_combinacoes = len(grid)
with tqdm(total=total_combinacoes) as pbar:
    for params in grid:
        modelo_gbm.set_params(**params)
        resultados_validacao_cruzada = cross_val_score(modelo_gbm, X_treino, y_treino, cv=5, scoring='accuracy')
        precisao_media = np.mean(resultados_validacao_cruzada)

        # Atualizar a barra de progresso
        pbar.update(1)
        pbar.set_description(f'Precisão Média: {precisao_media * 100:.2f}%')
        

#### Precisão do modelo

# Substitua 'target' pelo nome real da coluna que contém seus rótulos
X = df.drop('target', axis=1)
y = df['target']

# Dividir os dados em conjuntos de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo GBM
modelo_gbm = GradientBoostingClassifier()
modelo_gbm.fit(X_treino, y_treino)

# Fazer previsões nos dados de teste
previsoes = modelo_gbm.predict(X_teste)

# Avaliar a precisão do modelo
precisao = accuracy_score(y_teste, previsoes)

#### Escalonamento dos dados

# Criando um objeto de scaler
scaler = StandardScaler()

# Aplicando o scaler aos dados de treino e teste
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# Criar e treinar o modelo GBM com dados escalados
modelo_gbm_scaled = GradientBoostingClassifier()
modelo_gbm_scaled.fit(X_treino_scaled, y_treino)

# Fazer previsões nos dados de teste
previsoes_scaled = modelo_gbm_scaled.predict(X_teste_scaled)

# Avaliar a precisão do modelo
precisao_scaled = accuracy_score(y_teste, previsoes_scaled)
print(f'Precisão do modelo com Feature Scaling: {precisao_scaled * 100:.2f}%')

##########################################################################
####################### TREINANDO O MODELO - GBM #########################
##########################################################################


#### Acuraria do modelo ao longo do treinamento

# Substitua 'df' pelo seu DataFrame
X = df.drop('target', axis=1)
y = df['target']

# Dividir os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo
modelo_gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.01, max_depth=3)

# Dividir o conjunto de treinamento para treino e validação
X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, test_size=0.2, random_state=42)

# Listas para armazenar a acurácia no treino e na validação
acuracia_treino = []
acuracia_val = []

# Configurar a barra de progresso
pbar = tqdm(total=200, desc='Treinando Modelo', position=0, leave=True)

# Treinar o modelo e avaliar a acurácia em cada iteração
for i in range(1, 201):  # 200 iterações
    modelo_gbm.n_estimators = i
    modelo_gbm.fit(X_treino, y_treino)

    previsoes_treino = modelo_gbm.predict(X_treino)
    acuracia_treino.append(np.mean(previsoes_treino == y_treino))

    # Avaliar a acurácia no conjunto de validação
    previsoes_val = modelo_gbm.predict(X_val)
    acuracia_val_i = np.mean(previsoes_val == y_val)
    acuracia_val.append(acuracia_val_i)

    # Atualizar a barra de progresso
    pbar.update(1)

# Fechar a barra de progresso
pbar.close()

# Encontrar o número de iterações com melhor acurácia no conjunto de validação
melhor_iteracao = np.argmax(acuracia_val)

# Plotar a acurácia no treino e na validação
plt.figure(figsize=(10, 6))
plt.plot(acuracia_treino, label='Treino')
plt.plot(acuracia_val, label='Validação')
plt.scatter(melhor_iteracao, acuracia_val[melhor_iteracao], color='red', marker='o', label='Melhor Iteração')
plt.title('Acurácia ao Longo do Treinamento')
plt.xlabel('Número de Iterações')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

#### Curva de aprendizado do GBM

# Substitua 'df' pelo seu DataFrame
X = df.drop('target', axis=1)
y = df['target']

# Dividir os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo GBM
modelo_gbm = GradientBoostingClassifier()

# Treinar o modelo
modelo_gbm.fit(X_treino, y_treino)

# Lista para armazenar as acurácias de treino em cada iteração
acuracia_treino = []

# Treinar o modelo e avaliar a acurácia em cada iteração
for i, previsoes_treino in enumerate(modelo_gbm.staged_predict(X_treino)):
    acuracia_treino.append(np.mean(previsoes_treino == y_treino))

# Plotar a curva de aprendizado
plt.plot(range(1, len(acuracia_treino) + 1), acuracia_treino, label='Acurácia Treino', marker='o')
plt.xlabel('Número de Interações')
plt.ylabel('Acurácia')
plt.title('Curva de Aprendizado do Gradient Boosting Classifier')
plt.legend()
plt.show()