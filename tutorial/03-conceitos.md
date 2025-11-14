# Tutorial MLFlow - Conceitos Fundamentais

## Visão Geral

Este documento detalha os conceitos fundamentais do MLFlow que você precisa entender para usar a plataforma efetivamente.

## 1. Experiments (Experimentos)

### O que são?

Um **Experiment** é uma coleção de runs relacionadas para um problema ou objetivo específico de ML. Pense nele como um "projeto" que agrupa todas as tentativas de resolver um problema.

### Quando criar um novo experimento?

- Quando você está trabalhando em um novo problema/dataset
- Quando quer separar diferentes abordagens de modelagem
- Quando quer organizar trabalhos por equipe ou fase do projeto

### Como usar?

```python
import mlflow

# Criar ou definir experimento ativo
mlflow.set_experiment("Customer_Churn_Prediction")

# Obter ID do experimento
experiment = mlflow.get_experiment_by_name("Customer_Churn_Prediction")
print(f"Experiment ID: {experiment.experiment_id}")
```

### Boas práticas

- Use nomes descritivos: `Credit_Card_Fraud_Detection` ✅ vs `exp1` ❌
- Mantenha experimentos focados em um objetivo
- Use tags para adicionar metadados aos experimentos

## 2. Runs

### O que são?

Uma **Run** representa uma única execução do seu código de ML. Cada run registra parâmetros, métricas, artefatos e metadados.

### Anatomia de uma Run

```python
with mlflow.start_run(run_name="random_forest_v1"):
    # 1. Treinar modelo
    model.fit(X_train, y_train)
    
    # 2. Logar parâmetros
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # 3. Logar métricas
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("f1_score", 0.93)
    
    # 4. Logar artefatos
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("feature_importance.png")
    
    # 5. Adicionar tags
    mlflow.set_tag("model_type", "Random Forest")
```

### Run ID

Cada run recebe um ID único:

```python
run = mlflow.active_run()
print(f"Run ID: {run.info.run_id}")
# Output: Run ID: a7b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7
```

### Nested Runs

Você pode criar runs aninhadas para organizar melhor:

```python
with mlflow.start_run(run_name="hyperparameter_tuning"):
    for lr in [0.01, 0.1, 1.0]:
        with mlflow.start_run(run_name=f"lr_{lr}", nested=True):
            # Treinar e logar
            pass
```

## 3. Parameters (Parâmetros)

### O que são?

Parâmetros são **valores de entrada** para o seu modelo - geralmente hiperparâmetros que você ajusta.

### Tipos de parâmetros

```python
# Parâmetros do modelo
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)

# Parâmetros do dataset
mlflow.log_param("train_size", 0.8)
mlflow.log_param("random_state", 42)

# Parâmetros de pré-processamento
mlflow.log_param("scaler", "StandardScaler")
mlflow.log_param("feature_selection", "SelectKBest")
```

### Logar múltiplos parâmetros

```python
params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2
}
mlflow.log_params(params)
```

### Limitações

- Parâmetros são **strings** (convertidos automaticamente)
- Não podem ser atualizados após serem logados
- Use para valores de configuração, não para dados que mudam

## 4. Metrics (Métricas)

### O que são?

Métricas são **valores de saída** que avaliam o desempenho do modelo.

### Tipos de métricas

```python
# Métricas de classificação
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metric("precision", 0.93)
mlflow.log_metric("recall", 0.91)
mlflow.log_metric("f1_score", 0.92)
mlflow.log_metric("roc_auc", 0.96)

# Métricas de regressão
mlflow.log_metric("mse", 0.05)
mlflow.log_metric("rmse", 0.22)
mlflow.log_metric("mae", 0.18)
mlflow.log_metric("r2_score", 0.87)
```

### Métricas ao longo do tempo (Step)

Útil para treinar modelos iterativos:

```python
for epoch in range(10):
    # Treinar por uma época
    train_loss = train_one_epoch()
    val_loss = validate()
    
    # Logar métricas com step
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
```

### Logar múltiplas métricas

```python
metrics = {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.91
}
mlflow.log_metrics(metrics)
```

## 5. Artifacts (Artefatos)

### O que são?

Artefatos são **arquivos** gerados durante a run: modelos, gráficos, datasets, etc.

### Tipos comuns de artefatos

#### Modelos

```python
# Scikit-learn
mlflow.sklearn.log_model(model, "model")

# TensorFlow
mlflow.tensorflow.log_model(model, "model")

# PyTorch
mlflow.pytorch.log_model(model, "model")

# Modelo genérico
mlflow.pyfunc.log_model(artifact_path="model", python_model=custom_model)
```

#### Gráficos/Figuras

```python
import matplotlib.pyplot as plt

# Opção 1: Logar figura matplotlib diretamente
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
mlflow.log_figure(fig, "line_plot.png")

# Opção 2: Salvar e logar arquivo
plt.savefig("histogram.png")
mlflow.log_artifact("histogram.png")
```

#### Arquivos CSV/JSON

```python
# Salvar predições
predictions_df.to_csv("predictions.csv", index=False)
mlflow.log_artifact("predictions.csv")

# Salvar configurações
import json
config = {"param1": "value1"}
with open("config.json", "w") as f:
    json.dump(config, f)
mlflow.log_artifact("config.json")
```

#### Diretórios

```python
# Logar diretório inteiro
mlflow.log_artifacts("output_dir/", artifact_path="outputs")
```

### Estrutura de artefatos

```python
# Organizar artefatos em pastas
mlflow.log_artifact("model.pkl", artifact_path="models")
mlflow.log_artifact("plot.png", artifact_path="visualizations")
mlflow.log_artifact("data.csv", artifact_path="data")
```

## 6. Tags

### O que são?

Tags são **metadados chave-valor** para organizar e filtrar runs.

### Uso comum

```python
# Tags customizadas
mlflow.set_tag("model_type", "Random Forest")
mlflow.set_tag("version", "v1.0")
mlflow.set_tag("environment", "production")
mlflow.set_tag("developer", "João Silva")

# Tags automáticas do MLFlow
# - mlflow.source.name (nome do arquivo)
# - mlflow.source.type (tipo de source)
# - mlflow.user (usuário que executou)
```

### Buscar runs por tags

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="tags.model_type = 'Random Forest'"
)
```

## 7. Tracking URI

### O que é?

O **Tracking URI** define onde o MLFlow armazena os dados.

### Modos de armazenamento

#### Local (Padrão)

```python
# Salva em ./mlruns
mlflow.set_tracking_uri("file:./mlruns")
```

#### SQLite

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

#### Servidor Remoto

```python
mlflow.set_tracking_uri("http://localhost:5000")
```

#### PostgreSQL

```python
mlflow.set_tracking_uri(
    "postgresql://user:password@localhost:5432/mlflow"
)
```

## 8. Backend Store vs Artifact Store

### Backend Store

Armazena **metadados**:
- Informações de experimentos e runs
- Parâmetros
- Métricas
- Tags

**Opções**: File system, SQLAlchemy (SQLite, PostgreSQL, MySQL)

### Artifact Store

Armazena **arquivos grandes**:
- Modelos
- Gráficos
- Datasets
- Arquivos customizados

**Opções**: Local file system, S3, Azure Blob Storage, Google Cloud Storage, HDFS

### Configuração

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://my-bucket/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

## 9. Autologging

### O que é?

O MLFlow pode **automaticamente logar** parâmetros, métricas e modelos para frameworks populares.

### Frameworks suportados

```python
# Scikit-learn
mlflow.sklearn.autolog()

# TensorFlow/Keras
mlflow.tensorflow.autolog()

# PyTorch
mlflow.pytorch.autolog()

# XGBoost
mlflow.xgboost.autolog()

# LightGBM
mlflow.lightgbm.autolog()
```

### Exemplo

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier

mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    # Parâmetros, métricas e modelo logados automaticamente!
```

## 10. MLflow Client

### O que é?

API Python para interagir programaticamente com o MLFlow.

### Usos comuns

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Listar experimentos
experiments = client.search_experiments()

# Buscar runs
runs = client.search_runs(experiment_ids=["1"])

# Obter detalhes de uma run
run = client.get_run("run_id")

# Obter métricas
metrics = client.get_metric_history("run_id", "accuracy")

# Baixar artefatos
client.download_artifacts("run_id", "model", dst_path="./")
```

## Resumo dos Conceitos

| Conceito | O que é | Exemplo |
|----------|---------|---------|
| **Experiment** | Coleção de runs relacionadas | "Customer_Churn_Prediction" |
| **Run** | Execução única de código ML | Uma tentativa de treinar um modelo |
| **Parameter** | Valor de entrada (hiperparâmetro) | `n_estimators=100` |
| **Metric** | Valor de saída (desempenho) | `accuracy=0.95` |
| **Artifact** | Arquivo gerado | Modelo, gráfico, CSV |
| **Tag** | Metadado chave-valor | `model_type="Random Forest"` |
| **Tracking URI** | Local de armazenamento | `http://localhost:5000` |

## Próximos Passos

Agora que você entende os conceitos fundamentais:

- **[04 - Exemplos Práticos](04-exemplos-praticos.md)**: Veja os conceitos em ação
- **[05 - Model Registry](05-model-registry.md)**: Aprenda sobre gerenciamento de modelos
- **[06 - Melhores Práticas](06-melhores-praticas.md)**: Dicas para usar MLFlow efetivamente

---

**Dica**: Mantenha este documento como referência enquanto executa os exemplos práticos!
