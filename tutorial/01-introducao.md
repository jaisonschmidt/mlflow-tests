# Tutorial MLFlow - Introdução

## O que é MLFlow?

MLFlow é uma plataforma open-source desenvolvida pela Databricks para gerenciar o ciclo de vida completo de projetos de Machine Learning. Ele foi criado para resolver problemas comuns enfrentados por cientistas de dados e engenheiros de ML, como:

- **Rastreamento de experimentos**: Dificuldade em acompanhar diferentes versões de modelos e seus resultados
- **Reprodutibilidade**: Desafio de reproduzir experimentos e resultados
- **Deployment**: Complexidade em colocar modelos em produção
- **Gerenciamento de modelos**: Falta de centralização para versionar e organizar modelos

## Principais Componentes do MLFlow

O MLFlow é composto por quatro componentes principais:

### 1. MLFlow Tracking 📊

O componente de tracking permite registrar e consultar experimentos, incluindo:
- Código (versão do código-fonte)
- Parâmetros (hiperparâmetros do modelo)
- Métricas (accuracy, precision, recall, etc.)
- Artefatos (modelos, gráficos, arquivos)

**Exemplo de uso:**
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pkl")
```

### 2. MLFlow Projects 📦

Define projetos de ML de forma reproduzível usando:
- Formato padrão para empacotar código
- Especificação de dependências
- API para executar projetos

### 3. MLFlow Models 🤖

Fornece formato padrão para empacotar modelos que podem ser usados em diferentes plataformas:
- Salvamento consistente de modelos
- Suporte para múltiplos frameworks (scikit-learn, TensorFlow, PyTorch, etc.)
- Deploy facilitado

### 4. MLFlow Registry 🗄️

Sistema centralizado para gerenciar o ciclo de vida de modelos:
- Versionamento de modelos
- Transição de stages (Development → Staging → Production)
- Anotações e descrições
- Controle de acesso

## Por que usar MLFlow?

### Benefícios Principais

1. **Organização**: Mantenha todos os seus experimentos organizados em um único lugar
2. **Comparação**: Compare facilmente diferentes runs e identifique o melhor modelo
3. **Reprodutibilidade**: Registre tudo necessário para reproduzir um experimento
4. **Colaboração**: Compartilhe experimentos e modelos com sua equipe
5. **Deployment**: Simplifique o processo de colocar modelos em produção
6. **Framework Agnóstico**: Funciona com qualquer biblioteca de ML

### Casos de Uso Comuns

- **Experimentação Rápida**: Teste rapidamente diferentes hiperparâmetros e algoritmos
- **A/B Testing**: Compare versões de modelos em produção
- **Auditoria**: Mantenha histórico completo de todos os modelos treinados
- **Governança**: Controle quais modelos estão em produção e suas versões

## Arquitetura do MLFlow

```
┌─────────────────────────────────────────────────────────┐
│                    MLFlow UI (Interface Web)             │
│               http://localhost:5000                      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  MLFlow Tracking Server                  │
│  - Gerencia runs, experimentos e métricas               │
│  - API REST para logging e queries                      │
└─────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌──────────────────────┐      ┌──────────────────────┐
│   Backend Store      │      │   Artifact Store     │
│  (Metadata)          │      │   (Arquivos)         │
│                      │      │                      │
│  - Runs              │      │  - Modelos           │
│  - Parâmetros        │      │  - Gráficos          │
│  - Métricas          │      │  - Datasets          │
│  - Tags              │      │  - Arquivos CSV      │
│                      │      │                      │
│  SQLite / PostgreSQL │      │  Local / S3 / Azure  │
└──────────────────────┘      └──────────────────────┘
```

## Conceitos Fundamentais

### Experiment (Experimento)
Um experimento agrupa runs relacionadas. Por exemplo, você pode ter um experimento chamado "Customer Churn Prediction" que contém todas as tentativas de treinar modelos para esse problema.

```python
mlflow.set_experiment("Customer_Churn_Prediction")
```

### Run
Uma run representa uma única execução do seu código de ML. Cada run registra:
- Parâmetros de entrada
- Métricas de saída
- Versão do código
- Artefatos gerados

```python
with mlflow.start_run():
    # Seu código aqui
    pass
```

### Parameters (Parâmetros)
Valores de entrada para o seu modelo (hiperparâmetros):
```python
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)
```

### Metrics (Métricas)
Valores de saída que avaliam o desempenho do modelo:
```python
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metric("f1_score", 0.93)
```

### Artifacts (Artefatos)
Arquivos gerados durante a run (modelos, gráficos, datasets):
```python
mlflow.log_artifact("confusion_matrix.png")
mlflow.sklearn.log_model(model, "model")
```

### Tags
Metadados adicionais para organizar e filtrar runs:
```python
mlflow.set_tag("model_type", "random_forest")
mlflow.set_tag("version", "v1.0")
```

## Fluxo de Trabalho Típico

1. **Setup**: Configurar experimento e iniciar tracking
2. **Train**: Treinar modelo e logar parâmetros
3. **Evaluate**: Avaliar modelo e logar métricas
4. **Log**: Salvar artefatos (modelo, gráficos)
5. **Compare**: Comparar diferentes runs na UI
6. **Register**: Registrar melhor modelo no Model Registry
7. **Deploy**: Colocar modelo em produção

## Próximos Passos

Agora que você entende os conceitos básicos do MLFlow, continue para:

- **[02 - Instalação](02-instalacao.md)**: Configure o ambiente e instale as dependências
- **[03 - Conceitos](03-conceitos.md)**: Aprofunde-se nos conceitos do MLFlow
- **[04 - Exemplos Práticos](04-exemplos-praticos.md)**: Execute exemplos hands-on

---

**Recursos Adicionais:**
- [Documentação Oficial do MLFlow](https://mlflow.org/docs/latest/index.html)
- [MLFlow GitHub Repository](https://github.com/mlflow/mlflow)
- [MLFlow Tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)
