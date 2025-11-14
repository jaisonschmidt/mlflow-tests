"""
Exemplo 3: Comparação de Múltiplos Experimentos

Este script demonstra como executar múltiplas runs com diferentes hiperparâmetros
e comparar os resultados:
- Grid de hiperparâmetros
- Múltiplas runs em um mesmo experimento
- Comparação automática de resultados
- Identificação do melhor modelo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import mlflow
import mlflow.sklearn
from pathlib import Path
from itertools import product


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
    X = df.drop(['cliente_id', 'churn'], axis=1)
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Avalia o modelo e retorna as métricas."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    return metrics


def train_logistic_regression(X_train, X_test, y_train, y_test, params):
    """Treina e avalia Regressão Logística com parâmetros específicos."""
    
    run_name = f"LR_C={params['C']}_solver={params['solver']}"
    
    with mlflow.start_run(run_name=run_name):
        # Treinar modelo
        model = LogisticRegression(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Avaliar
        metrics = evaluate_model(model, X_test, y_test)
        
        # Logar
        mlflow.log_params(params)
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        return metrics, mlflow.active_run().info.run_id


def train_decision_tree(X_train, X_test, y_train, y_test, params):
    """Treina e avalia Árvore de Decisão com parâmetros específicos."""
    
    run_name = f"DT_maxdepth={params['max_depth']}_minsamples={params['min_samples_split']}"
    
    with mlflow.start_run(run_name=run_name):
        # Treinar modelo
        model = DecisionTreeClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Avaliar
        metrics = evaluate_model(model, X_test, y_test)
        
        # Logar
        mlflow.log_params(params)
        mlflow.log_param("model_type", "Decision Tree")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        return metrics, mlflow.active_run().info.run_id


def train_random_forest(X_train, X_test, y_train, y_test, params):
    """Treina e avalia Random Forest com parâmetros específicos."""
    
    run_name = f"RF_nestimators={params['n_estimators']}_maxdepth={params['max_depth']}"
    
    with mlflow.start_run(run_name=run_name):
        # Treinar modelo
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Avaliar
        metrics = evaluate_model(model, X_test, y_test)
        
        # Logar
        mlflow.log_params(params)
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        return metrics, mlflow.active_run().info.run_id


def main():
    """Função principal do experimento."""
    
    print("=" * 70)
    print("EXEMPLO 3: COMPARAÇÃO DE MÚLTIPLOS EXPERIMENTOS")
    print("=" * 70)
    
    # Configurar o experimento MLFlow
    experiment_name = "Churn_Prediction_Comparison"
    mlflow.set_experiment(experiment_name)
    
    print(f"\n📊 Experimento: {experiment_name}")
    
    # Carregar e preparar dados
    print("\n📁 Carregando dados...")
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    print(f"   - Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    # Armazenar resultados
    all_results = []
    
    # ========== REGRESSÃO LOGÍSTICA ==========
    print("\n" + "=" * 70)
    print("🔵 REGRESSÃO LOGÍSTICA - Grid Search")
    print("=" * 70)
    
    lr_param_grid = {
        'C': [0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [100]
    }
    
    lr_combinations = [dict(zip(lr_param_grid.keys(), v)) 
                       for v in product(*lr_param_grid.values())]
    
    print(f"Total de combinações: {len(lr_combinations)}")
    
    for i, params in enumerate(lr_combinations, 1):
        print(f"\n[{i}/{len(lr_combinations)}] Testando: {params}")
        metrics, run_id = train_logistic_regression(
            X_train, X_test, y_train, y_test, params
        )
        all_results.append({
            'model_type': 'Logistic Regression',
            'params': str(params),
            'run_id': run_id,
            **metrics
        })
        print(f"   F1-Score: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # ========== ÁRVORE DE DECISÃO ==========
    print("\n" + "=" * 70)
    print("🌳 ÁRVORE DE DECISÃO - Grid Search")
    print("=" * 70)
    
    dt_param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 10, 20]
    }
    
    dt_combinations = [dict(zip(dt_param_grid.keys(), v)) 
                       for v in product(*dt_param_grid.values())]
    
    print(f"Total de combinações: {len(dt_combinations)}")
    
    for i, params in enumerate(dt_combinations, 1):
        print(f"\n[{i}/{len(dt_combinations)}] Testando: {params}")
        metrics, run_id = train_decision_tree(
            X_train, X_test, y_train, y_test, params
        )
        all_results.append({
            'model_type': 'Decision Tree',
            'params': str(params),
            'run_id': run_id,
            **metrics
        })
        print(f"   F1-Score: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # ========== RANDOM FOREST ==========
    print("\n" + "=" * 70)
    print("🌲 RANDOM FOREST - Grid Search")
    print("=" * 70)
    
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 10]
    }
    
    rf_combinations = [dict(zip(rf_param_grid.keys(), v)) 
                       for v in product(*rf_param_grid.values())]
    
    print(f"Total de combinações: {len(rf_combinations)}")
    
    for i, params in enumerate(rf_combinations, 1):
        print(f"\n[{i}/{len(rf_combinations)}] Testando: {params}")
        metrics, run_id = train_random_forest(
            X_train, X_test, y_train, y_test, params
        )
        all_results.append({
            'model_type': 'Random Forest',
            'params': str(params),
            'run_id': run_id,
            **metrics
        })
        print(f"   F1-Score: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # ========== RESUMO FINAL ==========
    print("\n" + "=" * 70)
    print("📊 RESUMO DE TODOS OS EXPERIMENTOS")
    print("=" * 70)
    
    results_df = pd.DataFrame(all_results)
    
    # Ordenar por F1-Score
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    # Top 10 modelos
    print("\n🏆 TOP 10 MODELOS (ordenados por F1-Score):\n")
    top_10 = results_df.head(10)[['model_type', 'f1_score', 'roc_auc', 'accuracy', 'run_id']]
    print(top_10.to_string(index=False))
    
    # Melhor modelo por tipo
    print("\n🥇 MELHOR MODELO POR TIPO:\n")
    best_by_type = results_df.loc[results_df.groupby('model_type')['f1_score'].idxmax()]
    print(best_by_type[['model_type', 'f1_score', 'roc_auc', 'params']].to_string(index=False))
    
    # Melhor modelo geral
    best_overall = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("🎯 MELHOR MODELO GERAL")
    print("=" * 70)
    print(f"Tipo: {best_overall['model_type']}")
    print(f"Parâmetros: {best_overall['params']}")
    print(f"F1-Score: {best_overall['f1_score']:.4f}")
    print(f"ROC-AUC: {best_overall['roc_auc']:.4f}")
    print(f"Accuracy: {best_overall['accuracy']:.4f}")
    print(f"Run ID: {best_overall['run_id']}")
    
    # Salvar resultados
    output_file = Path(__file__).parent.parent / 'comparison_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Resultados salvos em: {output_file}")
    
    print("\n" + "=" * 70)
    print("Para visualizar e comparar todos os experimentos:")
    print("  mlflow ui")
    print("e acesse: http://localhost:5000")
    print("\nDica: Use a interface do MLFlow para:")
    print("  - Comparar runs lado a lado")
    print("  - Visualizar gráficos de métricas")
    print("  - Filtrar por parâmetros específicos")
    print("=" * 70)


if __name__ == "__main__":
    main()
