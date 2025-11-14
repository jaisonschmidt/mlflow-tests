# Tutorial MLFlow - Melhores Práticas

## Introdução

Este documento reúne melhores práticas, dicas e padrões recomendados para usar o MLFlow efetivamente em projetos de Machine Learning.

## 1. Organização de Experimentos

### Estrutura de Nomenclatura

#### ✅ Boas Práticas

```python
# Nomes descritivos e hierárquicos
mlflow.set_experiment("/projects/customer-analytics/churn-prediction")
mlflow.set_experiment("/models/nlp/sentiment-analysis-bert")
mlflow.set_experiment("/research/deep-learning/cnn-experiments")

# Com prefixo de time
mlflow.set_experiment("/data-science-team/production/fraud-detection")
mlflow.set_experiment("/ml-team/research/recommendation-system")
```

#### ❌ Evitar

```python
# Nomes genéricos
mlflow.set_experiment("test")
mlflow.set_experiment("model1")
mlflow.set_experiment("exp")
```

### Organização por Projeto

```
Experimentos/
├── /customer-churn/
│   ├── baseline-models
│   ├── feature-engineering
│   ├── hyperparameter-tuning
│   └── production-candidates
├── /fraud-detection/
│   ├── exploratory
│   ├── model-comparison
│   └── final-models
└── /recommendation-system/
    ├── collaborative-filtering
    ├── content-based
    └── hybrid-approaches
```

## 2. Naming Conventions

### Run Names

```python
# ✅ Descritivo com informações chave
with mlflow.start_run(run_name="RF_n100_depth10_2024-01-15"):
    pass

with mlflow.start_run(run_name="BERT_lr0.001_batch32_epoch10"):
    pass

# ❌ Genérico
with mlflow.start_run(run_name="run1"):
    pass
```

### Tags Padronizadas

```python
# Tags organizacionais
mlflow.set_tag("team", "data-science")
mlflow.set_tag("project", "customer-churn")
mlflow.set_tag("environment", "production")
mlflow.set_tag("model_type", "random_forest")

# Tags de processo
mlflow.set_tag("git_commit", "abc123def456")
mlflow.set_tag("developer", "joao.silva")
mlflow.set_tag("purpose", "hyperparameter_tuning")
mlflow.set_tag("dataset_version", "v2.1")

# Tags de status
mlflow.set_tag("status", "validated")
mlflow.set_tag("approved_for_production", "true")
```

## 3. Logging de Parâmetros e Métricas

### O que Logar

#### Parâmetros Essenciais

```python
# Hiperparâmetros do modelo
mlflow.log_params({
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.01,
    "random_state": 42
})

# Configuração de dados
mlflow.log_params({
    "train_size": 0.8,
    "test_size": 0.2,
    "cv_folds": 5,
    "stratify": True
})

# Pré-processamento
mlflow.log_params({
    "scaler": "StandardScaler",
    "handle_missing": "median",
    "feature_selection": "SelectKBest_k10"
})

# Ambiente
mlflow.log_params({
    "python_version": "3.9.7",
    "sklearn_version": "1.0.2",
    "n_jobs": -1
})
```

#### Métricas Completas

```python
# Métricas de classificação
mlflow.log_metrics({
    # Básicas
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    
    # Avançadas
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "log_loss": log_loss,
    
    # Por classe (se multiclass)
    "precision_class_0": precision_0,
    "recall_class_1": recall_1,
    
    # Treino vs Teste
    "train_accuracy": train_acc,
    "test_accuracy": test_acc,
    "overfitting_score": train_acc - test_acc
})

# Métricas de tempo
mlflow.log_metrics({
    "training_time_seconds": training_time,
    "prediction_time_ms": prediction_time,
    "model_size_mb": model_size
})
```

### Métricas Progressivas

```python
# Para treinamento iterativo (ex: Deep Learning)
for epoch in range(num_epochs):
    train_loss, val_loss = train_epoch()
    
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": current_lr
    }, step=epoch)
```

## 4. Gestão de Artefatos

### Organização de Artefatos

```python
# Estrutura organizada
with mlflow.start_run():
    # Modelos
    mlflow.sklearn.log_model(model, "models/final_model")
    
    # Gráficos por categoria
    mlflow.log_artifact("confusion_matrix.png", "visualizations/matrices")
    mlflow.log_artifact("roc_curve.png", "visualizations/curves")
    mlflow.log_artifact("feature_importance.png", "visualizations/analysis")
    
    # Dados processados
    mlflow.log_artifact("predictions.csv", "predictions")
    mlflow.log_artifact("feature_importance.csv", "data")
    
    # Configurações
    mlflow.log_artifact("model_config.json", "configs")
    mlflow.log_artifact("preprocessing_pipeline.pkl", "pipelines")
```

### Salvando Artefatos Complexos

```python
import tempfile
import json
import pickle

with mlflow.start_run():
    # JSON para configs
    config = {"param1": "value1", "param2": "value2"}
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = f"{tmp_dir}/config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(config_path, "configs")
    
    # Pickle para objetos Python
    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_path = f"{tmp_dir}/scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_path, "preprocessors")
    
    # Texto para logs
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = f"{tmp_dir}/training_log.txt"
        with open(log_path, 'w') as f:
            f.write("Training completed successfully\n")
            f.write(f"Final accuracy: {accuracy}\n")
        mlflow.log_artifact(log_path, "logs")
```

## 5. Reproducibilidade

### Fixar Seeds

```python
import random
import numpy as np
import mlflow

def set_seeds(seed=42):
    """Fixa todas as seeds para reproducibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    # Para PyTorch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # Para TensorFlow
    # tf.random.set_seed(seed)

with mlflow.start_run():
    set_seeds(42)
    mlflow.log_param("random_state", 42)
    
    # Treinar modelo
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
```

### Logar Versões de Dependências

```python
import sklearn
import pandas as pd
import numpy as np
import sys

with mlflow.start_run():
    # Versões de bibliotecas
    mlflow.log_param("python_version", sys.version)
    mlflow.log_param("sklearn_version", sklearn.__version__)
    mlflow.log_param("pandas_version", pd.__version__)
    mlflow.log_param("numpy_version", np.__version__)
    
    # Ou salvar requirements.txt
    import subprocess
    with tempfile.TemporaryDirectory() as tmp_dir:
        req_path = f"{tmp_dir}/requirements.txt"
        subprocess.run(["pip", "freeze"], stdout=open(req_path, 'w'))
        mlflow.log_artifact(req_path, "environment")
```

### Logar Código

```python
import mlflow
import os

with mlflow.start_run():
    # Logar script principal
    mlflow.log_artifact(__file__, "code")
    
    # Logar módulos customizados
    for module_file in ["utils.py", "preprocessing.py", "models.py"]:
        if os.path.exists(module_file):
            mlflow.log_artifact(module_file, "code")
```

### Git Integration

```python
import subprocess

def get_git_info():
    """Obtém informações do Git."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()
        
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        ).decode('ascii').strip()
        
        return commit, branch
    except:
        return None, None

with mlflow.start_run():
    commit, branch = get_git_info()
    if commit:
        mlflow.set_tag("git_commit", commit)
        mlflow.set_tag("git_branch", branch)
```

## 6. Performance e Otimização

### Batch Logging

```python
# ❌ Lento: Muitas chamadas individuais
for param_name, param_value in params.items():
    mlflow.log_param(param_name, param_value)

for metric_name, metric_value in metrics.items():
    mlflow.log_metric(metric_name, metric_value)

# ✅ Rápido: Batch logging
mlflow.log_params(params)
mlflow.log_metrics(metrics)
```

### Autologging

```python
# Usar autolog quando possível para economizar código
import mlflow.sklearn

mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True
)

with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    # Parâmetros, métricas e modelo logados automaticamente!
```

### Limitando Tamanho de Artefatos

```python
# Salvar apenas amostra de predições
predictions_sample = predictions_df.sample(n=1000, random_state=42)
predictions_sample.to_csv("predictions_sample.csv", index=False)
mlflow.log_artifact("predictions_sample.csv")

# Comprimir arquivos grandes
import gzip
import shutil

with gzip.open('large_file.csv.gz', 'wb') as f_out:
    with open('large_file.csv', 'rb') as f_in:
        shutil.copyfileobj(f_in, f_out)
mlflow.log_artifact('large_file.csv.gz')
```

## 7. Desenvolvimento vs Produção

### Ambiente de Desenvolvimento

```python
# dev_config.py
MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "/dev/customer-churn"
LOG_LEVEL = "DEBUG"
AUTOLOG = True
```

### Ambiente de Produção

```python
# prod_config.py
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
EXPERIMENT_NAME = "/prod/customer-churn"
LOG_LEVEL = "INFO"
AUTOLOG = False  # Mais controle
```

### Código Unificado

```python
import os
from config import dev_config, prod_config

# Escolher config baseado em variável de ambiente
ENV = os.getenv("ENVIRONMENT", "dev")
config = prod_config if ENV == "prod" else dev_config

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.EXPERIMENT_NAME)

if config.AUTOLOG:
    mlflow.sklearn.autolog()
```

## 8. Tratamento de Erros

### Garantir Finalização de Runs

```python
import mlflow

run = mlflow.start_run()
try:
    # Treinar modelo
    model.fit(X_train, y_train)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
except Exception as e:
    # Logar erro
    mlflow.set_tag("error", str(e))
    mlflow.set_tag("status", "failed")
    raise
    
finally:
    # Garantir que run seja finalizada
    mlflow.end_run()
```

### Context Manager (Recomendado)

```python
# ✅ Melhor: usa context manager
with mlflow.start_run():
    try:
        model.fit(X_train, y_train)
        mlflow.log_metric("accuracy", accuracy)
    except Exception as e:
        mlflow.set_tag("error", str(e))
        raise
```

## 9. Busca e Análise de Experimentos

### Buscar Runs Específicas

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Buscar runs com alta accuracy
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.85",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

# Buscar por tag
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="tags.model_type = 'Random Forest' and tags.status = 'validated'"
)

# Buscar por parâmetro
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="params.n_estimators = '100'"
)
```

### Análise Programática

```python
import pandas as pd

# Converter runs para DataFrame
runs_data = []
for run in runs:
    run_data = {
        'run_id': run.info.run_id,
        'run_name': run.data.tags.get('mlflow.runName', ''),
        **run.data.params,
        **run.data.metrics
    }
    runs_data.append(run_data)

df = pd.DataFrame(runs_data)

# Análise
print("Melhor accuracy:", df['accuracy'].max())
print("Média de F1-Score:", df['f1_score'].mean())

# Correlação entre parâmetros e métricas
correlation = df[['n_estimators', 'max_depth', 'accuracy']].corr()
print(correlation)
```

## 10. Segurança e Governança

### Controle de Acesso

```python
# Usar variáveis de ambiente para credenciais
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
```

### Auditoria

```python
with mlflow.start_run():
    # Logar quem executou
    mlflow.set_tag("executed_by", os.getenv("USER"))
    mlflow.set_tag("execution_time", datetime.now().isoformat())
    mlflow.set_tag("purpose", "production_training")
    
    # Logar aprovação
    mlflow.set_tag("approved_by", "supervisor@empresa.com")
    mlflow.set_tag("approval_date", "2024-01-15")
```

## 11. Checklist de Boas Práticas

### Antes de Treinar

- [ ] Fixar random seeds para reproducibilidade
- [ ] Definir experimento com nome descritivo
- [ ] Configurar tags organizacionais (team, project)
- [ ] Verificar tracking URI está configurado

### Durante o Treinamento

- [ ] Usar context manager (`with mlflow.start_run()`)
- [ ] Logar todos os hiperparâmetros relevantes
- [ ] Logar métricas de treino e validação
- [ ] Logar tempo de treinamento
- [ ] Adicionar tags de identificação (model_type, version)

### Após o Treinamento

- [ ] Salvar modelo treinado
- [ ] Logar artefatos relevantes (gráficos, predições)
- [ ] Adicionar descrição ao run
- [ ] Comparar com runs anteriores
- [ ] Documentar decisões importantes via tags

### Antes de Produção

- [ ] Validar métricas em holdout set
- [ ] Registrar modelo no Model Registry
- [ ] Adicionar descrição detalhada
- [ ] Testar carregamento do modelo
- [ ] Documentar processo de validação
- [ ] Obter aprovação necessária

## 12. Templates Úteis

### Template de Run Completa

```python
import mlflow
import mlflow.sklearn
from datetime import datetime

def train_and_log_model(X_train, y_train, X_test, y_test, params, tags=None):
    """Template para treinar e logar modelo com MLFlow."""
    
    with mlflow.start_run(run_name=f"model_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        # Tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        # Info de execução
        mlflow.set_tag("executed_by", os.getenv("USER"))
        mlflow.set_tag("execution_time", datetime.now().isoformat())
        
        # Parâmetros do modelo
        mlflow.log_params(params)
        
        # Parâmetros do dataset
        mlflow.log_params({
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1]
        })
        
        # Treinar
        start_time = time.time()
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Avaliar
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "training_time": training_time
        }
        
        mlflow.log_metrics(metrics)
        
        # Artefatos
        # Confusion Matrix
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()
        
        # Modelo
        mlflow.sklearn.log_model(model, "model")
        
        return model, metrics

# Uso
model, metrics = train_and_log_model(
    X_train, y_train, X_test, y_test,
    params={"n_estimators": 100, "max_depth": 10},
    tags={"team": "data-science", "project": "churn"}
)
```

## Recursos Adicionais

- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLFlow Examples](https://github.com/mlflow/mlflow/tree/master/examples)
- [MLFlow Best Practices (Databricks)](https://docs.databricks.com/mlflow/best-practices.html)

---

**Próximo**: Volte ao [README.md](../README.md) para visão geral completa do projeto.
