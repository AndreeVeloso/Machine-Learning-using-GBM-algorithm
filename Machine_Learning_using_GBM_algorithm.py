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
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm



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

###### Conversao para dados do mesmo tipo ######

# Lista de colunas a serem convertidas para float
colunas_para_converter = ['feature0', 'feature2', 'feature3', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14']

# Convertendo os valores das colunas para float
for coluna in colunas_para_converter:
    df[coluna] = pd.to_numeric(df[coluna], errors='coerce')

# Convertendo as colunas para float e verificando os tipos de dados
df = df.astype(float)
print(df.dtypes)
df = df.apply(pd.to_numeric, errors='coerce')



###### Tratamento de Duplicatas ######

# Verificando duplicatas em no dataframe e imprimindo as linhas duplicadas
duplicatas_totais = df.duplicated()
linhas_duplicadas = df[duplicatas_totais]
print("Linhas duplicadas:")
print(linhas_duplicadas)

# Verificando duplicatas em colunas de cada uma das features e imprindo na tela
colunas_para_verificar = ['feature0', 'feature2', 'feature3', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14']
duplicatas_colunas = df.duplicated(subset=colunas_para_verificar)
linhas_duplicadas_colunas = df[duplicatas_colunas]
print("Linhas duplicadas nas colunas específicas:")
print(linhas_duplicadas_colunas)

# Eliminando as linhas duplicadas e imprimindo dataframe apos a remocao
df_sem_duplicatas = df.drop_duplicates()
print("DataFrame após a remoção de todas as linhas duplicadas:")
print(df_sem_duplicatas)


###### Verificando colunas com possiveis dados faltantes ######

# Verificando colunas com dados faltantes e imprimindo
colunas_com_faltantes = df.columns[df.isnull().any()]
print("Colunas com dados faltantes:")
print(colunas_com_faltantes)

# Removendo colunas com dados faltantes e imprimindo
df_sem_faltantes = df.dropna(axis=1)
print("DataFrame após a remoção de colunas com dados faltantes:")
print(df_sem_faltantes)

df.head()

###### Plotando a matriz de correlacao com a variavel target(Não Classe e Classe) ######

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


###### Plotando o mapa de calor para a correlacao das variaveis ###### 

# Calculando a matriz de correlação e configurando o tamanho da figura
matriz_correlacao = df.corr()
plt.figure(figsize=(12, 10))

# Criando um mapa de calor - biblioteca seaborn***
sns.heatmap(matriz_correlacao, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title('Matriz de Correlação entre Variáveis')
plt.show()


##########################################################################
########## CRIACAO DO MODELO GBM PARA ITERAÇÃO DAS FEATURES ##############
##########################################################################


# Repassando a variavel do dataset para dataframe
dataframe = df

# Substituindo 'target' pelo nome da coluna que contém os rótulos
X = dataframe.drop('target', axis=1)
y = dataframe['target']

# Criando o modelo GBM - configs padrão
modelo_gbm = GradientBoostingClassifier()
modelo_gbm.fit(X, y)

# Correlacionando importância das características
importancias = modelo_gbm.feature_importances_

# DataFrame para visualização
df_importancias = pd.DataFrame({'Feature': X.columns, 'Importância': importancias})

# Ordenar o DataFrame por importância em ordem decrescente
df_importancias = df_importancias.sort_values(by='Importância', ascending=False)

# Visualizar a importância das características
plt.figure(figsize=(10, 7))
sns.barplot(x='Importância', y='Feature', data=df_importancias, palette='viridis')
plt.title('Importância das características - GBM')
plt.show()


##########################################################################
##################### TREINANDO O 1o MODELO - GBM ########################
##########################################################################

# Removendo coluna 'target' para treinamento do modelo e passando para as coluna y os rotulos
X = df.drop('target', axis=1)
y = df['target']

# Dividindo os dados em conjuntos de treinamento e teste - 70/30
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# Iniciando o modelo
modelo_gbm = GradientBoostingClassifier()

# Treinando o modelo
modelo_gbm.fit(X_treino, y_treino)

# Previsões do modelo
previsoes = modelo_gbm.predict(X_teste)

# Calculo das metricas
report = classification_report(y_teste, previsoes)
print(report)

# Gerando a Matriz de confusão
matriz_confusao = confusion_matrix(y_teste, previsoes)

# Plotando a matriz de confusão - biblioteca seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt="d", cmap="Blues", xticklabels=['Não Fraude', 'Fraude'], yticklabels=['Não Fraude', 'Fraude'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()


##########################################################################
##################### TREINANDO O 2o MODELO - GBM ########################
##########################################################################

# Ajuste na variavel parametros para a montagem da grade
parametros = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.25, 0.5],
    'max_depth': [3, 4, 5]
}

# Criando o modelo GBM
modelo_gbm = GradientBoostingClassifier()

# Montagem da grade de parâmetros
grid = ParameterGrid(parametros)


# Barra de progresso do treino
total_combinacoes = len(grid)
with tqdm(total=total_combinacoes) as pbar:
    for params in grid:
        modelo_gbm.set_params(**params)
        resultados_validacao_cruzada = cross_val_score(modelo_gbm, X_treino, y_treino, cv=5, scoring='accuracy')
        precisao_media = np.mean(resultados_validacao_cruzada)

        # Atualizar a barra de progresso
        pbar.update(1)
        pbar.set_description(f'Precisão Média: {precisao_media * 100:.2f}%')
        

###### Precisão do modelo ######

# Removendo coluna 'target' para treinamento do modelo e passando para a coluna y os rotulos
X = df.drop('target', axis=1)
y = df['target']

# Dividindo os dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o modelo e treinando
modelo_gbm = GradientBoostingClassifier()
modelo_gbm.fit(X_treino, y_treino)

# Previsões nos dados de teste
previsoes = modelo_gbm.predict(X_teste)

# Avaliação do modelo pela precisão
precisao = accuracy_score(y_teste, previsoes)


##########################################################################
##################### TREINANDO O 3o MODELO - GBM ########################
##########################################################################

###### Escalonamento dos dados  ######

# Criação do objeto scaler
scaler = StandardScaler()

# Aplicando o scaler aos dados de treino e teste
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# Criando e treinando o modelo GBM com dados escalados
modelo_gbm_scaled = GradientBoostingClassifier()
modelo_gbm_scaled.fit(X_treino_scaled, y_treino)

# Previsões nos dados de teste
previsoes_scaled = modelo_gbm_scaled.predict(X_teste_scaled)

# Avaliando a precisão do modelo
precisao_scaled = accuracy_score(y_teste, previsoes_scaled)
print(f'Precisão do modelo com Feature Scaling: {precisao_scaled * 100:.2f}%')


###### Acuraria do modelo ao longo do treinamento ######

# Removendo coluna 'target' para treinamento do modelo e passando para a coluna y os rotulos
X = df.drop('target', axis=1)
y = df['target']

# Dividindo os dados em conjuntos de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o modelo
modelo_gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3)

# Dividindo o conjunto de treinamento para treino e validação
X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, test_size=0.2, random_state=42)

# Listas para armazenar a acurácia no treino e na validação para as iterações
acuracia_treino = []
acuracia_val = []

# Configurando a barra de progresso
pbar = tqdm(total=200, desc='Treinando Modelo', position=0, leave=True)

# Treinando o modelo e avaliando a acurácia em cada iteração
for i in range(1, 201):  # 200 iterações
    modelo_gbm.n_estimators = i
    modelo_gbm.fit(X_treino, y_treino)

    previsoes_treino = modelo_gbm.predict(X_treino)
    acuracia_treino.append(np.mean(previsoes_treino == y_treino))

    # Avaliando a acurácia no conjunto de validação
    previsoes_val = modelo_gbm.predict(X_val)
    acuracia_val_i = np.mean(previsoes_val == y_val)
    acuracia_val.append(acuracia_val_i)

    # Atualizando a barra de progresso
    pbar.update(1)

# Fechar a barra de progresso
pbar.close()

# Encontrando o número de iterações com melhor acurácia no conjunto de validação
melhor_iteracao = np.argmax(acuracia_val)

# Plotando a acurácia no treino e na validação
plt.figure(figsize=(10, 6))
plt.plot(acuracia_treino, label='Treino')
plt.plot(acuracia_val, label='Validação')
plt.scatter(melhor_iteracao, acuracia_val[melhor_iteracao], color='red', marker='o', label='Melhor Iteração')
plt.title('Acurácia ao Longo do Treinamento')
plt.xlabel('Número de Iterações')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

###### Curva de aprendizado do GBM ######

# Removendo coluna 'target' para treinamento do modelo e passando para a coluna y os rotulos
X = df.drop('target', axis=1)
y = df['target']

# Dividindo os dados em conjuntos de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o modelo GBM
modelo_gbm = GradientBoostingClassifier()

# Treinando o modelo
modelo_gbm.fit(X_treino, y_treino)

# Lista para armazenar as acurácias de treino em cada iteração
acuracia_treino = []

# Treinando o modelo e avaliando a acurácia em cada iteração
for i, previsoes_treino in enumerate(modelo_gbm.staged_predict(X_treino)):
    acuracia_treino.append(np.mean(previsoes_treino == y_treino))

# Plotando a curva de aprendizado
plt.plot(range(1, len(acuracia_treino) + 1), acuracia_treino, label='Acurácia Treino', marker='o')
plt.xlabel('Número de Interações')
plt.ylabel('Acurácia')
plt.title('Curva de Aprendizado do Gradient Boosting Classifier')
plt.legend()
plt.show()


##########################################################################
############ TREINANDO O MODELO GBM COM RUIDO GAUSSIANO ##################
##########################################################################


# Função para adicionar ruído gaussiano a features aleatorias
def adicionar_ruido_gaussiano(df, features, desvio_padrao=0.1):
    df_ruido = df.copy()
    for feature in features:
        df_ruido[feature] += np.random.normal(0, desvio_padrao, df.shape[0])
    return df_ruido

# Selecionando 5 features aleatórias para adicionar ruído
features_com_ruido = np.random.choice(df.columns[:-1], 5, replace=False)

# Adicionando ruído nas features selecionadas
df_ruido = adicionar_ruido_gaussiano(df, features_com_ruido)

# Removendo coluna 'target' para treinamento do modelo e passando para a coluna y os rotulos
X_ruido = df_ruido.drop('target', axis=1)
y_ruido = df_ruido['target']

# Dividindo os dados ruidosos em conjuntos de treino e teste
X_treino_ruido, X_teste_ruido, y_treino_ruido, y_teste_ruido = train_test_split(X_ruido, y_ruido, test_size=0.3, random_state=42)

# Criando o modelo GBM 
modelo_gbm_ruido = GradientBoostingClassifier()

# Treinando o modelo com os dados com ruido
modelo_gbm_ruido.fit(X_treino_ruido, y_treino_ruido)

# Previsoes nos dados de teste com ruido
previsoes_ruido = modelo_gbm_ruido.predict(X_teste_ruido)

# Calculando as metricas para treinamento
report_ruidoso = classification_report(y_teste_ruido, previsoes_ruido)
print("Métricas para o segundo treinamento:")
print(report_ruidoso)

# Matriz de confusao para o segundo treinamento
matriz_confusao_ruidoso = confusion_matrix(y_teste_ruido, previsoes_ruido)

# Plotagem da Matriz de confusao - biblioteca seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_ruidoso, annot=True, fmt="d", cmap="Blues", xticklabels=['Não Fraude', 'Fraude'], yticklabels=['Não Fraude', 'Fraude'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Segundo Treinamento')
plt.show()

# Comparando as importancias das caracteristicas entre o treinamento original e o treinamento com ruido
importancias_ruido = modelo_gbm_ruido.feature_importances_
df_importancias_ruido = pd.DataFrame({'Feature': X_ruido.columns, 'Importância Ruido': importancias_ruido})
df_importancias_ruido = df_importancias_ruido.sort_values(by='Importância Ruido', ascending=False)

# Plotando importancia das caracteristicas no treinamento com ruido
plt.figure(figsize=(10, 7))
sns.barplot(x='Importância Ruidoso', y='Feature', data=df_importancias_ruido, palette='viridis')
plt.title('Importância das características - GBM com Ruído')
plt.show()

##########################################################################
########## TREINANDO O MODELO - MLP Apenas Tratamento de Dados ###########
##########################################################################

# Removendo coluna 'target' para treinamento do modelo e passando para a coluna y os rotulos
X_mlp = df.drop('target', axis=1)
y_mlp = df['target']

# Dividindo os dados em conjuntos de treino e teste
X_treino_mlp, X_teste_mlp, y_treino_mlp, y_teste_mlp = train_test_split(X_mlp, y_mlp, test_size=0.2, random_state=42)

# Criando o modelo MLP
modelo_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Treinando o modelo MLP
modelo_mlp.fit(X_treino_mlp, y_treino_mlp)

# Previsoes com o modelo MLP
previsoes_mlp = modelo_mlp.predict(X_teste_mlp)

# Metricas para o modelo MLP
report_mlp = classification_report(y_teste_mlp, previsoes_mlp)
print("Métricas para o modelo MLP:")
print(report_mlp)

# Matriz de confusao para o modelo MLP
matriz_confusao_mlp = confusion_matrix(y_teste_mlp, previsoes_mlp)

# Plotando a matriz de confusao - biblio seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_mlp, annot=True, fmt="d", cmap="Blues", xticklabels=['Não Fraude', 'Fraude'], yticklabels=['Não Fraude', 'Fraude'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Modelo MLP')
plt.show()

##################################################################
########## Ajuste de Hiperparametros para o Modelo MLP ###########
##################################################################

parametros_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'max_iter': [500, 1000, 1500]
}

modelo_mlp = MLPClassifier()

grid_mlp = ParameterGrid(parametros_mlp)

with tqdm(total=len(grid_mlp)) as pbar:
    for params in grid_mlp:
        modelo_mlp.set_params(**params)
        modelo_mlp.fit(X_treino_mlp, y_treino_mlp)
        previsoes_mlp = modelo_mlp.predict(X_teste_mlp)
        precisao_media = accuracy_score(y_teste_mlp, previsoes_mlp)

        pbar.update(1)
        pbar.set_description(f'Precisão Média: {precisao_media * 100:.2f}%')


#### Avaliacao da Acuracia ao Longo do Treinamento para o Modelo MLP
acuracia_treino_mlp = []
acuracia_val_mlp = []

total_iteracoes = 200

# Definindo X_val_mlp e y_val_mlp utilizando a mesma logica que usou para criar X_treino_mlp e y_treino_mlp - antes de entrar no loop for de iteracoes
X_treino_mlp, X_val_mlp, y_treino_mlp, y_val_mlp = train_test_split(X_treino_mlp, y_treino_mlp, test_size=0.2, random_state=42)

pbar_mlp = tqdm(total=total_iteracoes, desc='Treinando Modelo MLP', position=0, leave=True)

for i in range(1, total_iteracoes + 1):
    modelo_mlp.max_iter = i
    modelo_mlp.fit(X_treino_mlp, y_treino_mlp)

    previsoes_treino_mlp = modelo_mlp.predict(X_treino_mlp)
    acuracia_treino_mlp.append(np.mean(previsoes_treino_mlp == y_treino_mlp))

    previsoes_val_mlp = modelo_mlp.predict(X_val_mlp)
    acuracia_val_i_mlp = np.mean(previsoes_val_mlp == y_val_mlp)
    acuracia_val_mlp.append(acuracia_val_i_mlp)

    pbar_mlp.update(1)

# Fechar a barra de progresso
pbar_mlp.close()

# Encontrar o numero de iteracoes com melhor acuracia no conjunto de validacao
melhor_iteracao_mlp = np.argmax(acuracia_val_mlp)

# Plotar a acuracia no treino e na validacao
plt.figure(figsize=(10, 6))
plt.plot(acuracia_treino_mlp, label='Treino')
plt.plot(acuracia_val_mlp, label='Validação')
plt.scatter(melhor_iteracao_mlp, acuracia_val_mlp[melhor_iteracao_mlp], color='red', marker='o', label='Melhor Iteração')
plt.title('Acurácia ao Longo do Treinamento - Modelo MLP')
plt.xlabel('Número de Iterações')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

########################################################################
############# Treinamento do Modelo MLP com Feature Scaling ############
########################################################################

# Criando objeto para escalar os dados
scaler_mlp = StandardScaler()

# Aplicando os dados escalados para treino e teste
X_treino_scaled_mlp = scaler_mlp.fit_transform(X_treino_mlp)
X_teste_scaled_mlp = scaler_mlp.transform(X_teste_mlp)

# Criando e treinando o modelo
modelo_mlp_scaled = MLPClassifier()
modelo_mlp_scaled.fit(X_treino_scaled_mlp, y_treino_mlp)

# Previsoes do modelo
previsoes_scaled_mlp = modelo_mlp_scaled.predict(X_teste_scaled_mlp)

# Plotando a precisão do modelo
precisao_scaled_mlp = accuracy_score(y_teste_mlp, previsoes_scaled_mlp)
print(f'Precisão do modelo MLP com Feature Scaling: {precisao_scaled_mlp * 100:.2f}%')

########################################################################
############ Treinamento do Modelo MLP com Ruído Gaussiano #############
########################################################################

# Adicionando ruido nas features do dataframe

def adicionar_ruido_gaussiano_mlp(df, features, desvio_padrao=0.1):
    df_ruidoso_mlp = df.copy()
    for feature in features:
        df_ruidoso_mlp[feature] += np.random.normal(0, desvio_padrao, df.shape[0])
    return df_ruidoso_mlp

features_com_ruido_mlp = np.random.choice(X_mlp.columns, 5, replace=False)
df_ruidoso_mlp = adicionar_ruido_gaussiano_mlp(pd.concat([X_mlp, y_mlp], axis=1), features_com_ruido_mlp)

# Removendo coluna 'target' para treinamento do modelo e passando para a coluna y os rotulos

X_ruidoso_mlp = df_ruidoso_mlp.drop('target', axis=1)
y_ruidoso_mlp = df_ruidoso_mlp['target']

X_treino_ruidoso_mlp, X_teste_ruidoso_mlp, y_treino_ruidoso_mlp, y_teste_ruidoso_mlp = train_test_split(X_ruidoso_mlp, y_ruidoso_mlp, test_size=0.2, random_state=42)

modelo_mlp_ruidoso = MLPClassifier()
modelo_mlp_ruidoso.fit(X_treino_ruidoso_mlp, y_treino_ruidoso_mlp)

previsoes_ruidosas_mlp = modelo_mlp_ruidoso.predict(X_teste_ruidoso_mlp)
report_ruidoso_mlp = classification_report(y_teste_ruidoso_mlp, previsoes_ruidosas_mlp)
print("Métricas para o segundo treinamento do modelo MLP com ruído:")
print(report_ruidoso_mlp)

matriz_confusao_ruidoso_mlp = confusion_matrix(y_teste_ruidoso_mlp, previsoes_ruidosas_mlp)

plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_ruidoso_mlp, annot=True, fmt="d", cmap="Blues", xticklabels=['Não Fraude', 'Fraude'], yticklabels=['Não Fraude', 'Fraude'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Segundo Treinamento - Modelo MLP com Ruído')
plt.show()