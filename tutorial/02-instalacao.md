# Tutorial MLFlow - Instalação e Setup

## Pré-requisitos

Antes de começar, certifique-se de ter instalado:

- **Python 3.8+**: Versão mínima recomendada
- **pip**: Gerenciador de pacotes do Python
- **Git**: Para clonar o repositório (opcional)

Verificar versões:
```bash
python --version
pip --version
git --version
```

## Instalação

### 1. Clone o Repositório (ou baixe os arquivos)

```bash
git clone https://github.com/seu-usuario/mlflow-tests.git
cd mlflow-tests
```

### 2. Crie um Ambiente Virtual (Recomendado)

É uma boa prática usar um ambiente virtual para isolar as dependências do projeto:

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

Você saberá que o ambiente está ativo quando ver `(venv)` no início do prompt.

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

Isso instalará todos os pacotes necessários:
- `mlflow` - Plataforma de tracking
- `pandas` - Manipulação de dados
- `scikit-learn` - Algoritmos de ML
- `matplotlib` - Visualização de dados
- `seaborn` - Visualização estatística
- `numpy` - Computação numérica
- `jupyter` - Notebooks interativos (opcional)

### 4. Verifique a Instalação

```bash
mlflow --version
```

Você deve ver algo como: `mlflow, version 2.9.0`

## Estrutura do Projeto

Após a instalação, seu projeto deve ter a seguinte estrutura:

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
├── notebooks/                     # Notebooks Jupyter (opcional)
│   └── mlflow_tutorial.ipynb     # Tutorial interativo
├── tutorial/                      # Documentação
│   ├── 01-introducao.md
│   ├── 02-instalacao.md
│   ├── 03-conceitos.md
│   ├── 04-exemplos-praticos.md
│   ├── 05-model-registry.md
│   └── 06-melhores-praticas.md
├── utils/                         # Utilitários (futuro)
├── mlruns/                        # Dados do MLFlow (auto-gerado)
├── .gitignore                     # Arquivos ignorados pelo Git
├── requirements.txt               # Dependências Python
└── README.md                      # Documentação principal
```

## Configuração Inicial

### 1. Gere os Dados de Exemplo

Antes de executar os exemplos, você precisa gerar o dataset:

```bash
cd data
python generate_data.py
```

Saída esperada:
```
Gerando dados fictícios de clientes...

✓ Dados gerados com sucesso!
✓ Arquivo salvo em: customer_churn.csv

Resumo dos dados:
  - Total de clientes: 1000
  - Taxa de churn: 35.2%
  - Clientes com churn: 352
  - Clientes sem churn: 648
```

### 2. Verifique os Dados Gerados

```bash
head -n 5 customer_churn.csv
```

Você deve ver as primeiras linhas do CSV com colunas como:
- `cliente_id`
- `idade`
- `tempo_cliente_meses`
- `valor_mensal`
- `chamadas_suporte`
- `satisfacao`
- `num_produtos`
- `tem_cartao`
- `membro_ativo`
- `churn` (target)

### 3. Configure o MLFlow Tracking URI (Opcional)

Por padrão, o MLFlow salva os dados localmente em `./mlruns`. Se desejar usar um servidor remoto:

```bash
export MLFLOW_TRACKING_URI=http://seu-servidor:5000
```

Para este tutorial, usaremos o modo local (padrão).

## Testando a Instalação

Execute um exemplo simples para testar:

```bash
cd models
python 01_basic_tracking.py
```

Se tudo estiver configurado corretamente, você verá:
```
==============================================================
EXEMPLO 1: TRACKING BÁSICO COM MLFLOW
==============================================================

📊 Experimento: Churn_Prediction_Basic

📁 Carregando dados...
   - Total de registros: 1000
   - Treino: 800 | Teste: 200
   ...
```

## Iniciando a Interface Web do MLFlow

A interface web do MLFlow permite visualizar e comparar seus experimentos:

```bash
mlflow ui
```

Depois acesse no navegador:
```
http://localhost:5000
```

**Dica**: Para usar uma porta diferente:
```bash
mlflow ui --port 8080
```

### Principais Recursos da UI

1. **Experiments**: Lista todos os experimentos
2. **Runs**: Visualize todas as runs de um experimento
3. **Compare**: Compare múltiplas runs lado a lado
4. **Charts**: Visualize métricas em gráficos
5. **Models**: Acesse o Model Registry

## Solução de Problemas Comuns

### Erro: "mlflow: command not found"

**Solução**: Certifique-se de que o ambiente virtual está ativado e o MLFlow foi instalado:
```bash
source venv/bin/activate  # Linux/Mac
pip install mlflow
```

### Erro: "No module named 'sklearn'"

**Solução**: Instale o scikit-learn:
```bash
pip install scikit-learn
```

### Erro: "Permission denied" ao executar scripts

**Solução Linux/Mac**: Dê permissão de execução:
```bash
chmod +x models/*.py
```

### Porta 5000 já está em uso

**Solução**: Use uma porta diferente:
```bash
mlflow ui --port 5001
```

### ImportError no Windows

**Solução**: Certifique-se de estar usando Python 3.8+ e reinstale as dependências:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Configurações Avançadas (Opcional)

### Backend Store Remoto

Para usar PostgreSQL como backend:

```bash
mlflow server \
    --backend-store-uri postgresql://user:password@localhost/mlflow \
    --default-artifact-root s3://my-bucket/mlflow-artifacts \
    --host 0.0.0.0
```

### Artifact Store Remoto

Para usar S3 para artefatos:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my_experiment")
```

Configure variáveis de ambiente:
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
```

## Próximos Passos

Agora que você tem tudo instalado e configurado:

1. ✅ Ambiente configurado
2. ✅ Dados gerados
3. ✅ MLFlow instalado e testado

Continue para:
- **[03 - Conceitos](03-conceitos.md)**: Aprenda os conceitos fundamentais
- **[04 - Exemplos Práticos](04-exemplos-praticos.md)**: Execute os exemplos práticos

---

**Dica**: Mantenha o MLFlow UI aberto em uma aba do navegador enquanto executa os exemplos para visualizar os resultados em tempo real!
