"""
Exemplo 2: Logging de Artefatos (Gráficos e Arquivos)

Este script demonstra como fazer tracking de artefatos no MLFlow:
- Comparação entre dois modelos diferentes
- Logging de confusion matrix
- Logging de curva ROC
- Logging de feature importance
- Logging de arquivo CSV com predições
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
import mlflow
import mlflow.sklearn
from pathlib import Path
import tempfile


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


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Cria um gráfico de matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title)
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predito')
    plt.tight_layout()
    
    return plt.gcf()


def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    """Cria um gráfico da curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """Cria um gráfico de importância das features."""
    if hasattr(model, 'coef_'):
        # Regressão Logística - usar coeficientes
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        # Árvore de Decisão - usar importâncias
        importance = model.feature_importances_
    else:
        return None
    
    # Criar DataFrame para facilitar visualização
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_imp_df['feature'], feature_imp_df['importance'])
    plt.xlabel('Importância')
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()


def save_predictions(X_test, y_test, y_pred, y_proba, filename="predictions.csv"):
    """Salva as predições em um arquivo CSV."""
    predictions_df = X_test.copy()
    predictions_df['true_churn'] = y_test.values
    predictions_df['predicted_churn'] = y_pred
    predictions_df['churn_probability'] = y_proba
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / filename
        predictions_df.to_csv(filepath, index=False)
        return filepath


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Treina, avalia e faz tracking de um modelo."""
    
    print(f"\n{'='*60}")
    print(f"🤖 Modelo: {model_name}")
    print(f"{'='*60}")
    
    with mlflow.start_run(run_name=model_name):
        
        # Treinar modelo
        print("🚀 Treinando...")
        model.fit(X_train, y_train)
        
        # Fazer predições
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Logar parâmetros
        mlflow.log_params(model.get_params())
        
        # Logar métricas
        mlflow.log_metrics(metrics)
        
        print("\n📊 Métricas:")
        for metric_name, metric_value in metrics.items():
            print(f"   - {metric_name}: {metric_value:.4f}")
        
        # Criar e logar Confusion Matrix
        print("\n📈 Gerando artefatos...")
        fig_cm = plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {model_name}")
        mlflow.log_figure(fig_cm, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
        plt.close(fig_cm)
        
        # Criar e logar ROC Curve
        fig_roc = plot_roc_curve(y_test, y_proba, f"ROC Curve - {model_name}")
        mlflow.log_figure(fig_roc, f"roc_curve_{model_name.lower().replace(' ', '_')}.png")
        plt.close(fig_roc)
        
        # Criar e logar Feature Importance
        fig_imp = plot_feature_importance(model, X_train.columns, f"Feature Importance - {model_name}")
        if fig_imp:
            mlflow.log_figure(fig_imp, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
            plt.close(fig_imp)
        
        # Salvar e logar predições
        pred_file = save_predictions(X_test, y_test, y_pred, y_proba)
        mlflow.log_artifact(pred_file, "predictions")
        
        # Salvar modelo
        mlflow.sklearn.log_model(model, "model")
        
        print("✓ Artefatos salvos com sucesso!")
        print(f"   - Confusion Matrix")
        print(f"   - ROC Curve")
        print(f"   - Feature Importance")
        print(f"   - Predictions CSV")
        
        return metrics


def main():
    """Função principal do experimento."""
    
    print("=" * 60)
    print("EXEMPLO 2: LOGGING DE ARTEFATOS")
    print("=" * 60)
    
    # Configurar o experimento MLFlow
    experiment_name = "Churn_Prediction_Artifacts"
    mlflow.set_experiment(experiment_name)
    
    print(f"\n📊 Experimento: {experiment_name}")
    
    # Carregar e preparar dados
    print("\n📁 Carregando dados...")
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    print(f"   - Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    # Modelo 1: Regressão Logística
    lr_model = LogisticRegression(C=1.0, max_iter=100, random_state=42)
    lr_metrics = train_and_evaluate_model(
        lr_model, "Logistic Regression", 
        X_train, X_test, y_train, y_test
    )
    
    # Modelo 2: Árvore de Decisão
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_metrics = train_and_evaluate_model(
        dt_model, "Decision Tree", 
        X_train, X_test, y_train, y_test
    )
    
    # Comparar modelos
    print("\n" + "=" * 60)
    print("📊 COMPARAÇÃO DE MODELOS")
    print("=" * 60)
    
    comparison_df = pd.DataFrame({
        'Logistic Regression': lr_metrics,
        'Decision Tree': dt_metrics
    }).T
    
    print(comparison_df.to_string())
    
    # Determinar melhor modelo
    best_model = comparison_df['f1_score'].idxmax()
    print(f"\n🏆 Melhor modelo (F1-Score): {best_model}")
    
    print("\n" + "=" * 60)
    print("Para visualizar os artefatos, execute:")
    print("  mlflow ui")
    print("e acesse: http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
