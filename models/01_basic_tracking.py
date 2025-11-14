"""
Exemplo 1: Tracking Básico com MLFlow

Este script demonstra o uso básico do MLFlow para tracking de experimentos:
- Criação de um experimento
- Logging de parâmetros
- Logging de métricas
- Salvamento do modelo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from pathlib import Path


def load_data():
    """Carrega os dados de churn de clientes."""
    data_path = Path(__file__).parent.parent / 'data' / 'customer_churn.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: {data_path}\n"
            "Execute primeiro: python data/generate_data.py"
        )
    
    df = pd.read_csv(data_path)
    return df


def prepare_data(df):
    """Prepara os dados para treinamento."""
    # Separar features e target
    X = df.drop(['cliente_id', 'churn'], axis=1)
    y = df['churn']
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, C=1.0, solver='lbfgs', max_iter=100):
    """Treina um modelo de Regressão Logística."""
    model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Avalia o modelo e retorna as métricas."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    return metrics


def main():
    """Função principal do experimento."""
    
    print("=" * 60)
    print("EXEMPLO 1: TRACKING BÁSICO COM MLFLOW")
    print("=" * 60)
    
    # Configurar o experimento MLFlow
    experiment_name = "Churn_Prediction_Basic"
    mlflow.set_experiment(experiment_name)
    
    print(f"\n📊 Experimento: {experiment_name}")
    
    # Carregar e preparar dados
    print("\n📁 Carregando dados...")
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    print(f"   - Total de registros: {len(df)}")
    print(f"   - Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"   - Features: {list(X_train.columns)}")
    
    # Parâmetros do modelo
    params = {
        'C': 1.0,
        'solver': 'lbfgs',
        'max_iter': 100
    }
    
    print(f"\n🔧 Parâmetros do modelo: {params}")
    
    # Iniciar run do MLFlow
    with mlflow.start_run(run_name="logistic_regression_baseline"):
        
        print("\n🚀 Treinando modelo...")
        
        # Treinar modelo
        model = train_model(X_train, y_train, **params)
        
        # Avaliar modelo
        print("📈 Avaliando modelo...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Logar parâmetros no MLFlow
        mlflow.log_params(params)
        
        # Logar métricas no MLFlow
        mlflow.log_metrics(metrics)
        
        # Logar informações adicionais
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Salvar o modelo no MLFlow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=None  # Por enquanto, não registrar no Model Registry
        )
        
        # Exibir resultados
        print("\n✓ Treinamento concluído!")
        print("\n📊 Métricas:")
        for metric_name, metric_value in metrics.items():
            print(f"   - {metric_name}: {metric_value:.4f}")
        
        # Informações sobre o run
        run = mlflow.active_run()
        print(f"\n🔗 Run ID: {run.info.run_id}")
        print(f"🔗 Artifact URI: {run.info.artifact_uri}")
    
    print("\n" + "=" * 60)
    print("Para visualizar os resultados, execute:")
    print("  mlflow ui")
    print("e acesse: http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
