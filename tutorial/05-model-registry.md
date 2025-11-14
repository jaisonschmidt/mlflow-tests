# Tutorial MLFlow - Model Registry Detalhado

## Introdução

O **MLFlow Model Registry** é um repositório centralizado para gerenciar o ciclo de vida completo de modelos de Machine Learning, desde o desenvolvimento até a produção.

## Por que usar Model Registry?

### Problemas que resolve

- ❌ **Sem versionamento**: Difícil saber qual versão do modelo está em produção
- ❌ **Falta de rastreabilidade**: Não sabe como o modelo foi treinado
- ❌ **Colaboração difícil**: Equipe não sabe qual modelo usar
- ❌ **Deploy manual**: Processo propenso a erros
- ❌ **Sem governança**: Qualquer um pode colocar modelo em produção

### Benefícios

- ✅ **Versionamento automático**: Cada registro cria nova versão
- ✅ **Rastreabilidade completa**: Link para run original com todos os detalhes
- ✅ **Workflow colaborativo**: Stages facilitam colaboração
- ✅ **Deploy simplificado**: Carregue modelos por stage ou versão
- ✅ **Governança**: Controle sobre transições de stage
- ✅ **Auditoria**: Histórico completo de mudanças

## Conceitos Fundamentais

### 1. Registered Model

Um **Registered Model** é um modelo nomeado no registro que pode ter múltiplas versões.

```python
# Registrar modelo durante logging
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="customer_churn_predictor"
)

# Ou registrar modelo existente
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.create_registered_model(
    name="customer_churn_predictor",
    description="Modelo para predição de churn de clientes"
)
```

### 2. Model Version

Cada vez que você registra um modelo, uma nova versão é criada automaticamente.

```python
# Primeira vez - cria versão 1
mlflow.sklearn.log_model(model, "model", registered_model_name="my_model")

# Segunda vez - cria versão 2
mlflow.sklearn.log_model(model2, "model", registered_model_name="my_model")

# Terceira vez - cria versão 3
mlflow.sklearn.log_model(model3, "model", registered_model_name="my_model")
```

### 3. Model Stages

Cada versão pode estar em um **stage** do ciclo de vida:

| Stage | Descrição | Quando usar |
|-------|-----------|-------------|
| **None** | Sem stage definido | Versão recém-criada |
| **Staging** | Em validação | Modelo sendo testado antes de produção |
| **Production** | Em produção | Modelo servindo predições em produção |
| **Archived** | Arquivado | Versões antigas não mais utilizadas |

```python
# Transicionar versão para Production
client.transition_model_version_stage(
    name="customer_churn_predictor",
    version=2,
    stage="Production"
)
```

## Workflow Completo

### Passo 1: Treinar e Registrar Modelo

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Configurar experimento
mlflow.set_experiment("churn_prediction")

with mlflow.start_run(run_name="random_forest_v1"):
    # Treinar modelo
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Avaliar
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Registrar no Model Registry
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="churn_predictor"
    )
```

### Passo 2: Adicionar Metadados

```python
from mlflow.tracking import MlflowClient
import time

client = MlflowClient()

# Aguardar registro completar
time.sleep(1)

# Obter versão mais recente
latest_version = client.get_latest_versions("churn_predictor")[0].version

# Adicionar descrição
client.update_model_version(
    name="churn_predictor",
    version=latest_version,
    description="Random Forest com 100 árvores. "
                "Accuracy: 85% em dataset de teste. "
                "Treinado em 2024-01-15."
)

# Adicionar tags
client.set_model_version_tag(
    name="churn_predictor",
    version=latest_version,
    key="validation_status",
    value="pending"
)

client.set_model_version_tag(
    name="churn_predictor",
    version=latest_version,
    key="framework",
    value="sklearn"
)
```

### Passo 3: Validar e Promover

```python
# Transicionar para Staging para validação
client.transition_model_version_stage(
    name="churn_predictor",
    version=latest_version,
    stage="Staging",
    archive_existing_versions=False
)

# Após validação bem-sucedida, promover para Production
client.transition_model_version_stage(
    name="churn_predictor",
    version=latest_version,
    stage="Production",
    archive_existing_versions=True  # Arquiva versões anteriores em Production
)

# Atualizar tag
client.set_model_version_tag(
    name="churn_predictor",
    version=latest_version,
    key="validation_status",
    value="approved"
)
```

### Passo 4: Usar em Produção

```python
import mlflow.pyfunc

# Carregar modelo em produção
model_uri = "models:/churn_predictor/Production"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Fazer predições
predictions = loaded_model.predict(new_data)
```

## Operações Avançadas

### Buscar Modelos

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Listar todos os modelos registrados
all_models = client.search_registered_models()
for model in all_models:
    print(f"Nome: {model.name}")
    print(f"Criado: {model.creation_timestamp}")
    
# Obter detalhes de um modelo específico
model = client.get_registered_model("churn_predictor")
print(f"Descrição: {model.description}")
print(f"Latest versions: {model.latest_versions}")

# Buscar versões específicas
versions = client.search_model_versions("name='churn_predictor'")
for v in versions:
    print(f"Versão {v.version} - Stage: {v.current_stage}")
```

### Comparar Versões

```python
# Obter todas as versões de um modelo
versions = client.search_model_versions("name='churn_predictor'")

# Comparar métricas
for version in versions:
    run_id = version.run_id
    run = client.get_run(run_id)
    metrics = run.data.metrics
    print(f"Versão {version.version}:")
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"  F1-Score: {metrics.get('f1_score', 'N/A')}")
    print(f"  Stage: {version.current_stage}")
```

### Carregar Versão Específica

```python
# Por stage
model_prod = mlflow.pyfunc.load_model("models:/churn_predictor/Production")
model_stg = mlflow.pyfunc.load_model("models:/churn_predictor/Staging")

# Por número de versão
model_v1 = mlflow.pyfunc.load_model("models:/churn_predictor/1")
model_v2 = mlflow.pyfunc.load_model("models:/churn_predictor/2")

# Última versão
latest_versions = client.get_latest_versions("churn_predictor")
latest_version_number = latest_versions[0].version
model_latest = mlflow.pyfunc.load_model(f"models:/churn_predictor/{latest_version_number}")
```

### Deletar Versões/Modelos

```python
# Deletar uma versão específica
client.delete_model_version(
    name="churn_predictor",
    version=1
)

# Deletar modelo completo (todas as versões)
client.delete_registered_model(name="churn_predictor")
```

### Renomear Modelo

```python
# Renomear modelo registrado
client.rename_registered_model(
    name="churn_predictor",
    new_name="customer_churn_model_v2"
)
```

## Webhooks e Notificações

### Criar Webhook

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Webhook para notificar quando modelo vai para Production
client.create_model_version_tag(
    name="churn_predictor",
    version="1",
    key="notify_on_production",
    value="team@example.com"
)
```

### Webhooks via API REST

```bash
# Criar webhook que chama URL quando stage muda
curl -X POST http://localhost:5000/api/2.0/mlflow/registry-webhooks/create \
  -H "Content-Type: application/json" \
  -d '{
    "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
    "description": "Notificar equipe quando modelo vai para produção",
    "model_name": "churn_predictor",
    "http_url_spec": {
      "url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    }
  }'
```

## Integração com CI/CD

### Exemplo: GitHub Actions

```yaml
name: Deploy ML Model

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install mlflow scikit-learn
      
      - name: Train and register model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python train_model.py
      
      - name: Promote to production
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python scripts/promote_model.py
```

### Script de Promoção Automática

```python
# scripts/promote_model.py
from mlflow.tracking import MlflowClient
import sys

def promote_best_model():
    client = MlflowClient()
    
    # Buscar última versão
    latest_versions = client.get_latest_versions("churn_predictor", stages=["None"])
    
    if not latest_versions:
        print("Nenhuma versão encontrada")
        sys.exit(1)
    
    latest_version = latest_versions[0]
    run_id = latest_version.run_id
    
    # Verificar se métricas atendem critérios
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0)
    f1_score = run.data.metrics.get("f1_score", 0)
    
    if accuracy >= 0.80 and f1_score >= 0.75:
        # Promover para Staging
        client.transition_model_version_stage(
            name="churn_predictor",
            version=latest_version.version,
            stage="Staging"
        )
        print(f"✓ Versão {latest_version.version} promovida para Staging")
        
        # Após testes, promover para Production
        # (adicionar lógica de testes aqui)
        client.transition_model_version_stage(
            name="churn_predictor",
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"✓ Versão {latest_version.version} promovida para Production")
    else:
        print(f"✗ Métricas não atendem critérios mínimos")
        print(f"  Accuracy: {accuracy} (mín: 0.80)")
        print(f"  F1-Score: {f1_score} (mín: 0.75)")
        sys.exit(1)

if __name__ == "__main__":
    promote_best_model()
```

## Servindo Modelos

### Opção 1: MLFlow Models Serve

```bash
# Servir modelo via REST API
mlflow models serve \
  -m "models:/churn_predictor/Production" \
  -p 5001 \
  --no-conda
```

Fazer predições:
```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["idade", "tempo_cliente_meses", "valor_mensal", ...],
      "data": [[35, 24, 150.0, 3, 4, 2, 1, 1]]
    }
  }'
```

### Opção 2: Azure ML

```python
from azureml.core import Workspace, Model

# Conectar ao Azure ML
ws = Workspace.from_config()

# Registrar modelo do MLFlow no Azure
model_uri = "models:/churn_predictor/Production"
model = Model.register(
    workspace=ws,
    model_path=model_uri,
    model_name="churn_predictor"
)
```

### Opção 3: AWS SageMaker

```python
import mlflow.sagemaker as mfs

# Deploy para SageMaker
mfs.deploy(
    model_uri="models:/churn_predictor/Production",
    app_name="churn-predictor",
    execution_role_arn="arn:aws:iam::YOUR_ROLE",
    image_url="YOUR_ECR_IMAGE_URL"
)
```

## Boas Práticas

### 1. Nomenclatura

```python
# ✅ Bom: Nome descritivo e consistente
"customer_churn_predictor_v2"
"fraud_detection_random_forest"
"sentiment_analysis_bert"

# ❌ Ruim: Nome genérico ou confuso
"model1"
"test"
"my_model"
```

### 2. Descrições Detalhadas

```python
client.update_model_version(
    name="churn_predictor",
    version=3,
    description="""
    Random Forest Classifier para predição de churn de clientes.
    
    **Performance:**
    - Accuracy: 87.5%
    - F1-Score: 85.2%
    - ROC-AUC: 91.3%
    
    **Dataset:**
    - 10,000 clientes
    - Período: Jan-Dez 2023
    - Features: 15 variáveis comportamentais
    
    **Hiperparâmetros:**
    - n_estimators: 200
    - max_depth: 15
    - min_samples_split: 10
    
    **Aprovação:**
    - Validado por: Data Science Team
    - Data: 2024-01-15
    - Aprovador: João Silva
    """
)
```

### 3. Tags Organizacionais

```python
# Tags úteis
client.set_model_version_tag(name, version, "team", "data-science")
client.set_model_version_tag(name, version, "project", "churn-reduction")
client.set_model_version_tag(name, version, "framework", "sklearn")
client.set_model_version_tag(name, version, "deployed_date", "2024-01-15")
client.set_model_version_tag(name, version, "validation_dataset", "holdout_2024_Q1")
client.set_model_version_tag(name, version, "approved_by", "joao.silva@empresa.com")
```

### 4. Workflow de Stages

```
┌─────────────────────────────────────────────────────┐
│  1. Treinar modelo → Registrar (Stage: None)       │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  2. Validação inicial → Mover para Staging          │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  3. Testes A/B, QA → Aprovar ou rejeitar            │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  4. Aprovado → Mover para Production                │
│     (Arquivar versão anterior)                      │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  5. Monitorar performance em produção               │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  6. Nova versão disponível → Mover para Archived    │
└─────────────────────────────────────────────────────┘
```

### 5. Múltiplas Versões em Production

```python
# Manter versões A/B em produção
client.transition_model_version_stage(
    name="churn_predictor",
    version=2,
    stage="Production",
    archive_existing_versions=False  # Não arquivar versão anterior
)

# Agora versões 1 e 2 estão ambas em Production
# Útil para A/B testing

# Carregar versão específica
model_a = mlflow.pyfunc.load_model("models:/churn_predictor/1")
model_b = mlflow.pyfunc.load_model("models:/churn_predictor/2")
```

## Próximos Passos

- **[06 - Melhores Práticas](06-melhores-praticas.md)**: Aprenda mais dicas avançadas
- **[Exemplo 4](04-exemplos-praticos.md#exemplo-4-model-registry)**: Execute o exemplo prático de Model Registry

---

**Recursos Adicionais:**
- [MLFlow Model Registry Docs](https://mlflow.org/docs/latest/model-registry.html)
- [MLFlow Registry Webhooks](https://mlflow.org/docs/latest/registry-webhooks.html)
