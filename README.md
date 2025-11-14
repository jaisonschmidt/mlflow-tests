# Tutorial Completo de MLFlow

<p align="center">
	<img src="https://img.shields.io/badge/MLFlow-2.9+-blue" alt="MLFlow">
	<img src="https://img.shields.io/badge/Python-3.8+-green" alt="Python">
	<img src="https://img.shields.io/badge/scikit--learn-1.3+-orange" alt="scikit-learn">
	<img src="https://img.shields.io/badge/Status-Completo-success" alt="Status">
</p>

Tutorial prático e completo de MLFlow em português, desde conceitos básicos até Model Registry e deployment. Aprenda a fazer tracking de experimentos de Machine Learning usando um caso prático de predição de churn de clientes.

---

## 📚 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Exemplos Práticos](#exemplos-práticos)
- [Documentação](#documentação)
- [Recursos Adicionais](#recursos-adicionais)

---

## 🎯 Sobre o Projeto

Este repositório contém um tutorial completo de **MLFlow**, uma plataforma open-source para gerenciar o ciclo de vida completo de projetos de Machine Learning.

### O que você vai aprender:

- ✅ **Conceitos fundamentais** do MLFlow (Experiments, Runs, Parameters, Metrics)
- ✅ **Tracking básico** de modelos e experimentos
- ✅ **Logging de artefatos** (gráficos, confusion matrix, ROC curves)
- ✅ **Comparação de experimentos** com grid search
- ✅ **Model Registry** para versionamento e gerenciamento de modelos
- ✅ **Melhores práticas** para projetos de ML

### Caso de Uso:

**Predição de Churn de Clientes** - Um problema de classificação binária usando dados fictícios de clientes de uma empresa, com objetivo de prever quais clientes têm maior probabilidade de cancelar o serviço.

---

## 📁 Estrutura do Projeto

```
mlflow-tests/
├── data/                          # Dados do projeto
│   ├── generate_data.py          # Script para gerar dados fictícios
│   └── customer_churn.csv        # Dataset gerado (após executar)
├── models/                        # Scripts de treinamento
│   ├── 01_basic_tracking.py      # Exemplo 1: Tracking básico
│   ├── 02_artifacts_tracking.py  # Exemplo 2: Logging de artefatos
│   ├── 03_compare_experiments.py # Exemplo 3: Comparação de experimentos
│   └── 04_model_registry.py      # Exemplo 4: Model Registry
├── notebooks/                     # Notebooks Jupyter
│   └── mlflow_tutorial.ipynb     # Tutorial interativo completo
├── tutorial/                      # Documentação detalhada
│   ├── 01-introducao.md          # Introdução ao MLFlow
│   ├── 02-instalacao.md          # Guia de instalação
│   ├── 03-conceitos.md           # Conceitos fundamentais
│   ├── 04-exemplos-praticos.md   # Guia dos exemplos práticos
│   ├── 05-model-registry.md      # Model Registry detalhado
│   └── 06-melhores-praticas.md   # Melhores práticas e dicas
├── utils/                         # Utilitários (futuro)
├── mlruns/                        # Dados do MLFlow (auto-gerado)
├── .gitignore                     # Arquivos ignorados pelo Git
├── requirements.txt               # Dependências Python
└── README.md                      # Este arquivo
```

---

## 🔧 Pré-requisitos

- **Python 3.8 ou superior**
- **pip** (gerenciador de pacotes Python)
- **Git** (opcional, para clonar o repositório)

Verificar versões:
```bash
python --version
pip --version
```

---

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/jaisonschmidt/mlflow-tests.git
cd mlflow-tests
```

### 2. Crie um ambiente virtual (recomendado)

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Gere os dados de exemplo

```bash
python data/generate_data.py
```

**Saída esperada:**
```
✓ Dados gerados com sucesso!
✓ Arquivo salvo em: data/customer_churn.csv

Resumo dos dados:
	- Total de clientes: 1000
	- Taxa de churn: 35.2%
```

---

## 🚀 Como Usar

### Iniciar a Interface Web do MLFlow

```bash
mlflow ui
```

Acesse no navegador: **http://localhost:5000**

### Executar os Exemplos

#### Exemplo 1: Tracking Básico
```bash
python models/01_basic_tracking.py
```

Demonstra tracking básico com parâmetros, métricas e salvamento de modelo.

#### Exemplo 2: Logging de Artefatos
```bash
python models/02_artifacts_tracking.py
```

Demonstra como logar gráficos (confusion matrix, ROC curve, feature importance) e comparar modelos.

#### Exemplo 3: Comparação de Experimentos
```bash
python models/03_compare_experiments.py
```

Executa grid search com múltiplos modelos e compara resultados (⚠️ pode levar alguns minutos).

#### Exemplo 4: Model Registry
```bash
python models/04_model_registry.py
```

Demonstra registro, versionamento e gerenciamento de modelos no Model Registry.

### Executar o Notebook Interativo

```bash
jupyter notebook notebooks/mlflow_tutorial.ipynb
```

O notebook contém todos os exemplos com explicações detalhadas e exercícios práticos.

---

## 📖 Exemplos Práticos

### Exemplo 1: Tracking Básico

```python
import mlflow
from sklearn.linear_model import LogisticRegression

mlflow.set_experiment("Churn_Prediction")

with mlflow.start_run(run_name="logistic_regression"):
		# Treinar modelo
		model = LogisticRegression(C=1.0, max_iter=100)
		model.fit(X_train, y_train)
    
		# Logar parâmetros
		mlflow.log_param("C", 1.0)
		mlflow.log_param("max_iter", 100)
    
		# Logar métricas
		accuracy = model.score(X_test, y_test)
		mlflow.log_metric("accuracy", accuracy)
    
		# Salvar modelo
		mlflow.sklearn.log_model(model, "model")
```

### Exemplo 2: Logging de Artefatos

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

with mlflow.start_run():
		# ... treinar modelo ...
    
		# Logar confusion matrix
		fig, ax = plt.subplots()
		cm = confusion_matrix(y_test, y_pred)
		sns.heatmap(cm, annot=True, fmt='d', ax=ax)
		mlflow.log_figure(fig, "confusion_matrix.png")
```

### Exemplo 3: Model Registry

```python
# Registrar modelo
mlflow.sklearn.log_model(
		model,
		"model",
		registered_model_name="churn_predictor"
)

# Transicionar para Production
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
		name="churn_predictor",
		version=1,
		stage="Production"
)

# Carregar modelo de produção
model = mlflow.pyfunc.load_model("models:/churn_predictor/Production")
predictions = model.predict(new_data)
```

---

## 📚 Documentação

Documentação completa em português disponível na pasta `tutorial/`:

1. **[Introdução](tutorial/01-introducao.md)** - O que é MLFlow, conceitos e arquitetura
2. **[Instalação](tutorial/02-instalacao.md)** - Guia passo a passo de instalação e setup
3. **[Conceitos](tutorial/03-conceitos.md)** - Experiments, Runs, Parameters, Metrics, Artifacts
4. **[Exemplos Práticos](tutorial/04-exemplos-praticos.md)** - Guia detalhado de cada exemplo
5. **[Model Registry](tutorial/05-model-registry.md)** - Versionamento e gerenciamento de modelos
6. **[Melhores Práticas](tutorial/06-melhores-praticas.md)** - Dicas e padrões recomendados

---

## 🎓 Conceitos Principais

### Experiments
Agrupa runs relacionadas para um problema específico de ML.

### Runs
Representa uma única execução do código de ML, registrando parâmetros, métricas e artefatos.

### Parameters
Valores de entrada para o modelo (hiperparâmetros).

### Metrics
Valores de saída que avaliam o desempenho do modelo.

### Artifacts
Arquivos gerados durante a run (modelos, gráficos, datasets).

### Model Registry
Sistema centralizado para gerenciar versões e lifecycle de modelos.

---

## 🔥 Features

- ✅ **4 exemplos progressivos** de uso do MLFlow
- ✅ **Notebook interativo** com exercícios práticos
- ✅ **Documentação completa** em português
- ✅ **Dataset fictício** gerado automaticamente
- ✅ **Visualizações** (confusion matrix, ROC curve, feature importance)
- ✅ **Comparação de modelos** (Logistic Regression, Decision Tree, Random Forest)
- ✅ **Model Registry** com versionamento
- ✅ **Melhores práticas** e padrões recomendados

---

## 🛠️ Tecnologias Utilizadas

- **MLFlow 2.9+** - Platform de tracking de ML
- **Python 3.8+** - Linguagem de programação
- **Scikit-learn 1.3+** - Algoritmos de ML
- **Pandas** - Manipulação de dados
- **Matplotlib & Seaborn** - Visualizações
- **Jupyter** - Notebooks interativos

---

## 📊 Métricas e Visualizações

Os exemplos incluem tracking de:

- **Métricas de Classificação**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizações**: Confusion Matrix, ROC Curve, Feature Importance
- **Comparações**: Múltiplos modelos e hiperparâmetros

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fork o projeto
2. Criar uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abrir um Pull Request

---

## 📝 Licença

Este projeto é open-source e está disponível sob a licença MIT.

---

## 🌐 Recursos Adicionais

- [Documentação Oficial do MLFlow](https://mlflow.org/docs/latest/index.html)
- [MLFlow GitHub Repository](https://github.com/mlflow/mlflow)
- [MLFlow Tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

## 👨‍💻 Autor

**Jaison Schmidt**
- GitHub: [@jaisonschmidt](https://github.com/jaisonschmidt)

---

## 📞 Suporte

Se tiver dúvidas ou problemas:

1. Consulte a [documentação](tutorial/)
2. Verifique os [exemplos práticos](tutorial/04-exemplos-praticos.md)
3. Abra uma [issue](https://github.com/jaisonschmidt/mlflow-tests/issues)

---

<p align="center">
	<strong>⭐ Se este tutorial foi útil, considere dar uma estrela no repositório! ⭐</strong>
</p>

<p align="center">
	Feito com ❤️ para a comunidade de Data Science
</p>