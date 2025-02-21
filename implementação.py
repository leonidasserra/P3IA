import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns

""" Carrega a base de dados Car Evaluation"""
dataset = fetch_ucirepo(id=19)
df = pd.DataFrame(dataset.data.features, columns=dataset.metadata.features) 
df['class'] = dataset.data.targets

"""
Pré-processamento: Codificar atributos categóricos
Cada coluna categórica foi transformada em valores numéricos utilizando LabelEncoder,
para permitir que os modelo processe os dados corretamente.
"""
label_encoders = {}
for coluna in df.columns:
    le = LabelEncoder()
    df[coluna] = le.fit_transform(df[coluna])
    label_encoders[coluna] = le

""" Separa o dataframe em features(atributos) e classe alvo de aceitabilidade"""
X = df.drop(columns=['class'])
y = df['class']

"""
Divide o conjunto de dados em treino e teste
Aqui usamos 80% dos dados para treino e 20% para teste.
A divisão mantém a distribuição original das classes com stratify=y.
"""

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

"""
Escolhemos o modelo Decision Tree (arvore de decisão)
Criamos um classificador de árvore de decisão com um estado aleatório fixo para reprodutibilidade (random_state = 42).
"""
modelo = DecisionTreeClassifier(random_state=42)

"""
Validação cruzada k-fold
Aqui utilizamos 5 folds para avaliar a estabilidade do modelo antes do treinamento final.
"""
cv_scores = cross_val_score(modelo, X_treino, y_treino, cv=5)
print("\n" + "=" * 60)
print("Validação Cruzada")
print("=" * 60)
print(f'Média da acurácia: {cv_scores.mean():.4f}')
print(f'Desvio padrão da acurácia: {cv_scores.std():.4f}')
print(f'Acurácias em cada fold: {cv_scores}')

"""
Otimização de hiperparâmetros usando GridSearchCV
Aqui testamos diferentes valores para 'max_depth' e 'criterion' para encontrar a melhor configuração.
"""
param_grid = {'max_depth': [3, 5, 10, None], 'criterion': ['gini', 'entropy']}
grid_search = GridSearchCV(modelo, param_grid, cv=5)
grid_search.fit(X_treino, y_treino)
melhor_modelo = grid_search.best_estimator_
print("\n" + "=" * 60)
print("Otimização de Hiperparâmetros")
print("=" * 60)
print(f'Melhor hiperparâmetro: {grid_search.best_params_}')

"""
Avaliação final do modelo
Após encontrar os melhores hiperparâmetros, treinamos o modelo final e avaliamos seu desempenho.
"""
melhor_modelo.fit(X_treino, y_treino)
y_pred = melhor_modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, y_pred)
print("\n" + "=" * 60)
print("Avaliação Final")
print("=" * 60)
print(f'Acurácia no teste: {acuracia:.4f}')
print("\nRelatório de Classificação:")
print(classification_report(y_teste, y_pred))

"""
Matriz de Confusão
Mostra a quantidade de classificações corretas e incorretas para cada classe.
"""
matriz_de_confusao = confusion_matrix(y_teste, y_pred)
print("\n" + "=" * 60)
print("Matriz de Confusão")
print("=" * 60)
print(matriz_de_confusao)

"""
Visualização da Matriz de Confusão usando heatmap
Isso ajuda a identificar erros de classificação.
"""
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_de_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['class'].classes_, yticklabels=label_encoders['class'].classes_)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

"""
Importância das Features (se disponível)
Se o modelo tiver informações sobre a importância das features, exibimos um gráfico mostrando quais
variáveis têm maior influência na previsão.
"""
if hasattr(melhor_modelo, 'feature_importances_'):
    feature_importances = melhor_modelo.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("\n" + "=" * 60)
    print("Importância das Features")
    print("=" * 60)
    print(importance_df)

    # Gráfico de Importância das Features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Importância das Features')
    plt.show()
