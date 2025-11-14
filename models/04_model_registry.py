"""
Exemplo 4: Model Registry - Registro e Gerenciamento de Modelos

Este script demonstra como usar o MLFlow Model Registry:
- Registro de modelos no registry
- Versionamento de modelos
- Transição entre stages (None -> Staging -> Production)
- Carregamento de modelos registrados
- Atualização de descrições e tags
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from pathlib import Path
import time


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


def train_and_register_model(X_train, X_test, y_train, y_test, 
                             model_name, version_description):
    """Treina um modelo e registra no Model Registry."""
    
    print(f"\n🚀 Treinando novo modelo: {version_description}")
    
    with mlflow.start_run(run_name=version_description):
        # Configurar parâmetros
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'random_state': 42
        }
        
        # Treinar modelo
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Avaliar
        metrics = evaluate_model(model, X_test, y_test)
        
        # Logar parâmetros e métricas
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        print(f"   Métricas - F1: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Registrar modelo no Model Registry
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=model_name
        )
        
        run_id = mlflow.active_run().info.run_id
        
    return run_id, metrics


def get_latest_model_version(client, model_name):
    """Obtém a versão mais recente de um modelo."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            return max([int(v.version) for v in versions])
        return None
    except:
        return None


def main():
    """Função principal do experimento."""
    
    print("=" * 70)
    print("EXEMPLO 4: MODEL REGISTRY - GERENCIAMENTO DE MODELOS")
    print("=" * 70)
    
    # Nome do modelo no registry
    model_name = "churn_prediction_model"
    
    # Configurar o experimento MLFlow
    experiment_name = "Churn_Prediction_Registry"
    mlflow.set_experiment(experiment_name)
    
    # Inicializar client do MLFlow
    client = MlflowClient()
    
    print(f"\n📊 Experimento: {experiment_name}")
    print(f"🏷️  Nome do modelo: {model_name}")
    
    # Carregar e preparar dados
    print("\n📁 Carregando dados...")
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # ========== VERSÃO 1: MODELO INICIAL ==========
    print("\n" + "=" * 70)
    print("📦 VERSÃO 1: Modelo Inicial")
    print("=" * 70)
    
    run_id_v1, metrics_v1 = train_and_register_model(
        X_train, X_test, y_train, y_test,
        model_name,
        "Versão 1 - Modelo baseline"
    )
    
    # Obter número da versão
    time.sleep(1)  # Aguardar registro
    version_1 = get_latest_model_version(client, model_name)
    
    if version_1:
        # Adicionar descrição e tags
        client.update_model_version(
            name=model_name,
            version=version_1,
            description="Modelo baseline de Random Forest para predição de churn. "
                       "Primeira versão do modelo com hiperparâmetros padrão."
        )
        
        client.set_model_version_tag(
            name=model_name,
            version=version_1,
            key="stage",
            value="baseline"
        )
        
        print(f"\n✓ Modelo registrado como versão {version_1}")
        
        # Transicionar para Staging
        client.transition_model_version_stage(
            name=model_name,
            version=version_1,
            stage="Staging"
        )
        print(f"✓ Versão {version_1} movida para: Staging")
    
    # ========== VERSÃO 2: MODELO MELHORADO ==========
    print("\n" + "=" * 70)
    print("📦 VERSÃO 2: Modelo Melhorado")
    print("=" * 70)
    
    # Simular melhoria: treinar com mais dados ou diferentes parâmetros
    # (neste exemplo, vamos apenas criar uma segunda versão para demonstração)
    run_id_v2, metrics_v2 = train_and_register_model(
        X_train, X_test, y_train, y_test,
        model_name,
        "Versão 2 - Modelo otimizado"
    )
    
    time.sleep(1)
    version_2 = get_latest_model_version(client, model_name)
    
    if version_2:
        # Adicionar descrição
        client.update_model_version(
            name=model_name,
            version=version_2,
            description="Versão melhorada do modelo com hiperparâmetros otimizados. "
                       "Aprovada para produção após validação em staging."
        )
        
        client.set_model_version_tag(
            name=model_name,
            version=version_2,
            key="stage",
            value="production_candidate"
        )
        
        print(f"\n✓ Modelo registrado como versão {version_2}")
        
        # Transicionar para Production
        client.transition_model_version_stage(
            name=model_name,
            version=version_2,
            stage="Production"
        )
        print(f"✓ Versão {version_2} movida para: Production")
        
        # Arquivar versão anterior
        if version_1:
            client.transition_model_version_stage(
                name=model_name,
                version=version_1,
                stage="Archived"
            )
            print(f"✓ Versão {version_1} movida para: Archived")
    
    # ========== VISUALIZAR TODAS AS VERSÕES ==========
    print("\n" + "=" * 70)
    print("📋 TODAS AS VERSÕES DO MODELO")
    print("=" * 70)
    
    all_versions = client.search_model_versions(f"name='{model_name}'")
    
    if all_versions:
        versions_data = []
        for version in all_versions:
            versions_data.append({
                'Versão': version.version,
                'Stage': version.current_stage,
                'Run ID': version.run_id[:8] + "...",
                'Criado em': pd.to_datetime(version.creation_timestamp, unit='ms').strftime('%Y-%m-%d %H:%M')
            })
        
        versions_df = pd.DataFrame(versions_data)
        versions_df = versions_df.sort_values('Versão', ascending=False)
        print("\n" + versions_df.to_string(index=False))
    
    # ========== CARREGAR MODELO DE PRODUÇÃO ==========
    print("\n" + "=" * 70)
    print("🔄 CARREGANDO MODELO DE PRODUÇÃO")
    print("=" * 70)
    
    # Carregar modelo do stage "Production"
    model_uri = f"models:/{model_name}/Production"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    
    print(f"\n✓ Modelo carregado: {model_uri}")
    
    # Fazer predição de exemplo
    sample_data = X_test.head(5)
    predictions = loaded_model.predict(sample_data)
    probabilities = loaded_model.predict_proba(sample_data)[:, 1]
    
    print("\n📊 Exemplo de predições com modelo de produção:")
    results = sample_data.copy()
    results['Predição'] = predictions
    results['Probabilidade Churn'] = probabilities.round(3)
    print(results.to_string(index=False))
    
    # ========== INFORMAÇÕES DO MODELO REGISTRADO ==========
    print("\n" + "=" * 70)
    print("ℹ️  INFORMAÇÕES DO MODELO REGISTRADO")
    print("=" * 70)
    
    try:
        model_details = client.get_registered_model(model_name)
        print(f"\nNome: {model_details.name}")
        print(f"Criado em: {pd.to_datetime(model_details.creation_timestamp, unit='ms')}")
        print(f"Última atualização: {pd.to_datetime(model_details.last_updated_timestamp, unit='ms')}")
        print(f"Descrição: {model_details.description if model_details.description else 'N/A'}")
    except Exception as e:
        print(f"Erro ao obter detalhes: {e}")
    
    # ========== COMPARAÇÃO DE MÉTRICAS ==========
    print("\n" + "=" * 70)
    print("📊 COMPARAÇÃO DE MÉTRICAS ENTRE VERSÕES")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        'Versão 1': metrics_v1,
        'Versão 2': metrics_v2,
        'Diferença': {k: metrics_v2[k] - metrics_v1[k] for k in metrics_v1.keys()}
    }).T
    
    print("\n" + comparison.to_string())
    
    print("\n" + "=" * 70)
    print("✅ PRÓXIMOS PASSOS")
    print("=" * 70)
    print("\n1. Visualize o Model Registry:")
    print("   mlflow ui")
    print("   Acesse: http://localhost:5000")
    print("\n2. No MLFlow UI, vá para a aba 'Models' para:")
    print("   - Ver todas as versões do modelo")
    print("   - Comparar métricas entre versões")
    print("   - Gerenciar stages (Staging, Production, Archived)")
    print("   - Adicionar descrições e tags")
    print("\n3. Para usar o modelo em produção:")
    print(f"   model = mlflow.sklearn.load_model('models:/{model_name}/Production')")
    print("=" * 70)


if __name__ == "__main__":
    main()
