# Tutorial MLFlow - Exemplos Práticos

## Visão Geral

Este tutorial prático guia você através de quatro exemplos progressivos que demonstram as principais funcionalidades do MLFlow.

## Preparação

Antes de começar, certifique-se de:

1. ✅ Ter instalado todas as dependências (`pip install -r requirements.txt`)
2. ✅ Ter gerado os dados (`python data/generate_data.py`)
3. ✅ Estar na raiz do projeto

## Exemplo 1: Tracking Básico

### Objetivo

Aprender o básico do MLFlow tracking: logar parâmetros, métricas e salvar modelos.

### O que você vai aprender

- Criar e configurar um experimento
- Iniciar uma run
- Logar parâmetros do modelo
- Logar métricas de avaliação
- Salvar o modelo treinado

### Executar

```bash
cd models
python 01_basic_tracking.py
```

### O que acontece

1. **Carrega os dados** de churn de clientes do CSV
2. **Divide** em treino (80%) e teste (20%)
3. **Treina** um modelo de Regressão Logística
4. **Loga no MLFlow**:
   - Parâmetros: `C`, `solver`, `max_iter`
   - Métricas: `accuracy`, `precision`, `recall`, `f1_score`
   - Informações adicionais: quantidade de amostras, features
   - Modelo treinado

### Saída esperada

```
==============================================================
EXEMPLO 1: TRACKING BÁSICO COM MLFLOW
==============================================================

📊 Experimento: Churn_Prediction_Basic

📁 Carregando dados...
   - Total de registros: 1000
   - Treino: 800 | Teste: 200
   - Features: ['idade', 'tempo_cliente_meses', ...]

🔧 Parâmetros do modelo: {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 100}

🚀 Treinando modelo...
📈 Avaliando modelo...

✓ Treinamento concluído!

📊 Métricas:
   - accuracy: 0.7850
   - precision: 0.7234
   - recall: 0.6891
   - f1_score: 0.7058

🔗 Run ID: a7b3c4d5e6f7g8h9...
```

### Visualizar na UI

1. Execute `mlflow ui` (em outro terminal)
2. Acesse `http://localhost:5000`
3. Clique no experimento "Churn_Prediction_Basic"
4. Veja os parâmetros, métricas e artefatos da run

### Pontos de atenção

- **Run Name**: Nome descritivo ajuda a identificar runs específicas
- **Parâmetros vs Métricas**: Parâmetros são inputs, métricas são outputs
- **Modelo Salvo**: O modelo fica disponível em `mlruns/.../artifacts/model`

---

## Exemplo 2: Logging de Artefatos

### Objetivo

Aprender a logar artefatos visuais e comparar diferentes modelos.

### O que você vai aprender

- Logar gráficos (confusion matrix, ROC curve)
- Logar feature importance
- Salvar predições em CSV
- Comparar múltiplos modelos

### Executar

```bash
python 02_artifacts_tracking.py
```

### O que acontece

1. **Treina dois modelos**:
   - Regressão Logística
   - Árvore de Decisão
2. **Para cada modelo, loga**:
   - Confusion Matrix (heatmap)
   - Curva ROC
   - Feature Importance
   - Arquivo CSV com predições
3. **Compara** os resultados dos dois modelos

### Saída esperada

```
==============================================================
EXEMPLO 2: LOGGING DE ARTEFATOS
==============================================================

==============================================================
🤖 Modelo: Logistic Regression
==============================================================

🚀 Treinando...

📊 Métricas:
   - accuracy: 0.7850
   - precision: 0.7234
   - recall: 0.6891
   - f1_score: 0.7058
   - roc_auc: 0.8234

📈 Gerando artefatos...
✓ Artefatos salvos com sucesso!
   - Confusion Matrix
   - ROC Curve
   - Feature Importance
   - Predictions CSV

==============================================================
🤖 Modelo: Decision Tree
==============================================================
[similar output]

==============================================================
📊 COMPARAÇÃO DE MODELOS
==============================================================
                       accuracy  precision    recall  f1_score   roc_auc
Logistic Regression    0.7850    0.7234      0.6891    0.7058    0.8234
Decision Tree          0.7650    0.6982      0.7123    0.7051    0.7891

🏆 Melhor modelo (F1-Score): Logistic Regression
```

### Visualizar na UI

1. Abra `http://localhost:5000`
2. Clique no experimento "Churn_Prediction_Artifacts"
3. Selecione uma run
4. Vá para a aba "Artifacts"
5. Visualize os gráficos clicando neles

### Pontos de atenção

- **Artefatos Visuais**: Ajudam a entender o desempenho do modelo
- **Comparação**: A tabela final facilita identificar o melhor modelo
- **CSV de Predições**: Útil para análise posterior ou auditoria

---

## Exemplo 3: Comparação de Múltiplos Experimentos

### Objetivo

Executar grid search e comparar dezenas de modelos automaticamente.

### O que você vai aprender

- Executar múltiplas runs em loop
- Grid search de hiperparâmetros
- Comparar modelos diferentes (Logistic Regression, Decision Tree, Random Forest)
- Identificar automaticamente o melhor modelo

### Executar

```bash
python 03_compare_experiments.py
```

⚠️ **Atenção**: Este script executa muitas runs (6 LR + 12 DT + 8 RF = 26 runs) e pode levar alguns minutos.

### O que acontece

1. **Regressão Logística**: Testa 6 combinações de hiperparâmetros
   - `C`: [0.1, 1.0, 10.0]
   - `solver`: ['lbfgs', 'liblinear']

2. **Árvore de Decisão**: Testa 12 combinações
   - `max_depth`: [3, 5, 7, 10]
   - `min_samples_split`: [2, 10, 20]

3. **Random Forest**: Testa 8 combinações
   - `n_estimators`: [50, 100]
   - `max_depth`: [5, 10]
   - `min_samples_split`: [2, 10]

4. **Compara** todos os resultados e identifica o melhor

### Saída esperada

```
======================================================================
EXEMPLO 3: COMPARAÇÃO DE MÚLTIPLOS EXPERIMENTOS
======================================================================

======================================================================
🔵 REGRESSÃO LOGÍSTICA - Grid Search
======================================================================
Total de combinações: 6

[1/6] Testando: {'C': 0.1, 'solver': 'lbfgs', 'max_iter': 100}
   F1-Score: 0.7012 | ROC-AUC: 0.8145
[2/6] Testando: {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 100}
   F1-Score: 0.7058 | ROC-AUC: 0.8234
...

======================================================================
📊 RESUMO DE TODOS OS EXPERIMENTOS
======================================================================

🏆 TOP 10 MODELOS (ordenados por F1-Score):

model_type          f1_score  roc_auc  accuracy  run_id
Random Forest       0.7234    0.8456   0.7950    abc123...
Random Forest       0.7198    0.8423   0.7925    def456...
Decision Tree       0.7156    0.8134   0.7850    ghi789...
...

🥇 MELHOR MODELO POR TIPO:

model_type          f1_score  roc_auc  params
Random Forest       0.7234    0.8456    {'n_estimators': 100, 'max_depth': 10, ...}
Decision Tree       0.7156    0.8134    {'max_depth': 7, 'min_samples_split': 2}
Logistic Regression 0.7058    0.8234    {'C': 1.0, 'solver': 'lbfgs', ...}

======================================================================
🎯 MELHOR MODELO GERAL
======================================================================
Tipo: Random Forest
Parâmetros: {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2}
F1-Score: 0.7234
ROC-AUC: 0.8456
Accuracy: 0.7950
Run ID: abc123...

✓ Resultados salvos em: comparison_results.csv
```

### Visualizar na UI

1. Abra `http://localhost:5000`
2. Clique no experimento "Churn_Prediction_Comparison"
3. **Compare runs**:
   - Selecione múltiplas runs (checkbox)
   - Clique em "Compare"
   - Visualize tabela comparativa e gráficos

### Dicas de visualização

- **Filtrar por modelo**: Use a barra de busca com `tags.model_type = "Random Forest"`
- **Ordenar**: Clique nos headers das colunas para ordenar por métrica
- **Gráfico de comparação**: Visualize tendências de hiperparâmetros vs métricas

### Pontos de atenção

- **Nomeação de Runs**: Cada run tem nome único com os parâmetros
- **CSV de Resultados**: Salvo para análise offline em `comparison_results.csv`
- **Escalabilidade**: Para grids grandes, considere usar ferramentas de otimização (Optuna, Hyperopt)

---

## Exemplo 4: Model Registry

### Objetivo

Aprender a usar o Model Registry para gerenciar versões e lifecycle de modelos.

### O que você vai aprender

- Registrar modelos no Model Registry
- Criar múltiplas versões
- Transicionar entre stages (Staging, Production, Archived)
- Carregar modelos registrados
- Adicionar descrições e tags

### Executar

```bash
python 04_model_registry.py
```

### O que acontece

1. **Treina Versão 1**:
   - Random Forest baseline
   - Registra no Model Registry
   - Move para stage "Staging"

2. **Treina Versão 2**:
   - Random Forest otimizado
   - Registra como nova versão
   - Move para stage "Production"
   - Arquiva versão anterior

3. **Gerencia o registro**:
   - Adiciona descrições
   - Adiciona tags
   - Lista todas as versões

4. **Carrega modelo de produção**:
   - Usa URI especial `models:/{nome}/Production`
   - Faz predições de exemplo

### Saída esperada

```
======================================================================
EXEMPLO 4: MODEL REGISTRY - GERENCIAMENTO DE MODELOS
======================================================================

📊 Experimento: Churn_Prediction_Registry
🏷️  Nome do modelo: churn_prediction_model

======================================================================
📦 VERSÃO 1: Modelo Inicial
======================================================================

🚀 Treinando novo modelo: Versão 1 - Modelo baseline
   Métricas - F1: 0.7145, ROC-AUC: 0.8312

✓ Modelo registrado como versão 1
✓ Versão 1 movida para: Staging

======================================================================
📦 VERSÃO 2: Modelo Melhorado
======================================================================

🚀 Treinando novo modelo: Versão 2 - Modelo otimizado
   Métricas - F1: 0.7234, ROC-AUC: 0.8456

✓ Modelo registrado como versão 2
✓ Versão 2 movida para: Production
✓ Versão 1 movida para: Archived

======================================================================
📋 TODAS AS VERSÕES DO MODELO
======================================================================

Versão  Stage       Run ID       Criado em
2       Production  abc123...    2024-01-15 10:30
1       Archived    def456...    2024-01-15 10:29

======================================================================
🔄 CARREGANDO MODELO DE PRODUÇÃO
======================================================================

✓ Modelo carregado: models:/churn_prediction_model/Production

📊 Exemplo de predições com modelo de produção:
   idade  tempo_cliente_meses  ...  Predição  Probabilidade Churn
   35     24                  ...  0         0.234
   52     8                   ...  1         0.789
   ...
```

### Visualizar na UI

1. Abra `http://localhost:5000`
2. Clique na aba **"Models"** (topo da página)
3. Veja o modelo "churn_prediction_model"
4. Explore:
   - Versões do modelo
   - Stage atual de cada versão
   - Descrições e tags
   - Métricas linkadas

### Operações no Model Registry (via UI)

- **Transição de Stage**: Clique em "Stage" → Selecione novo stage
- **Adicionar Descrição**: Edite a descrição da versão
- **Comparar Versões**: Selecione múltiplas versões e compare
- **Ver Run Original**: Clique no link da run para ver detalhes

### Pontos de atenção

- **Stages**: None → Staging → Production → Archived
- **Múltiplas Versões em Production**: Possível ter > 1 versão em Production
- **Carregar Modelo**: Use `models:/{nome}/{stage}` ou `models:/{nome}/{version}`
- **Auditoria**: Todas as transições ficam registradas

---

## Comandos Úteis

### Executar todos os exemplos em sequência

```bash
cd models
python 01_basic_tracking.py && \
python 02_artifacts_tracking.py && \
python 03_compare_experiments.py && \
python 04_model_registry.py
```

### Limpar dados do MLFlow

```bash
# ⚠️ CUIDADO: Remove todos os experimentos
rm -rf mlruns/
rm -rf mlartifacts/
```

### Exportar experimento

```bash
mlflow experiments csv -x 1 -o experiment_1.csv
```

### Ver informações de uma run específica

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
run = client.get_run("run_id_aqui")
print(run.data.params)
print(run.data.metrics)
```

## Exercícios Práticos

### Exercício 1: Modificar Hiperparâmetros

Edite `01_basic_tracking.py` e teste diferentes valores de `C`:

```python
params = {
    'C': 0.5,  # Teste: 0.1, 0.5, 2.0, 10.0
    'solver': 'lbfgs',
    'max_iter': 100
}
```

Compare as métricas na UI.

### Exercício 2: Adicionar Nova Métrica

No `02_artifacts_tracking.py`, adicione a métrica de especificidade:

```python
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
mlflow.log_metric("specificity", specificity)
```

### Exercício 3: Novo Tipo de Modelo

No `03_compare_experiments.py`, adicione o SVM:

```python
from sklearn.svm import SVC

svm_param_grid = {
    'C': [0.1, 1.0, 10.0],
    'kernel': ['linear', 'rbf']
}
# Implemente o loop similar aos outros modelos
```

### Exercício 4: Promover Modelo

Crie um script que:
1. Busca a run com melhor F1-score
2. Registra esse modelo no Registry
3. Promove para Production automaticamente

## Próximos Passos

Parabéns! Você completou os exemplos práticos. Continue para:

- **[05 - Model Registry](05-model-registry.md)**: Aprofunde no gerenciamento de modelos
- **[06 - Melhores Práticas](06-melhores-praticas.md)**: Aprenda dicas avançadas

---

**Dica**: Experimente modificar os códigos e ver o impacto nas métricas. O MLFlow torna fácil experimentar e comparar!
