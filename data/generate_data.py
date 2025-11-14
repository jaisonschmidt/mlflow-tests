"""
Script para gerar dataset fictício de churn de clientes.

Este script cria um arquivo CSV com dados fictícios simulando
comportamento de clientes e se eles cancelaram o serviço (churn).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configurar seed para reprodutibilidade
np.random.seed(42)

def generate_customer_data(n_samples=1000):
    """
    Gera dados fictícios de clientes.
    
    Parameters:
    -----------
    n_samples : int
        Número de amostras a serem geradas
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com os dados gerados
    """
    
    data = {
        # ID do cliente
        'cliente_id': range(1, n_samples + 1),
        
        # Idade (18 a 70 anos)
        'idade': np.random.randint(18, 71, n_samples),
        
        # Tempo como cliente (0 a 120 meses / 10 anos)
        'tempo_cliente_meses': np.random.randint(0, 121, n_samples),
        
        # Valor gasto mensalmente (R$ 50 a R$ 500)
        'valor_mensal': np.random.uniform(50, 500, n_samples).round(2),
        
        # Número de chamadas ao suporte (0 a 15)
        'chamadas_suporte': np.random.randint(0, 16, n_samples),
        
        # Satisfação do cliente (1 a 5)
        'satisfacao': np.random.randint(1, 6, n_samples),
        
        # Número de produtos contratados (1 a 4)
        'num_produtos': np.random.randint(1, 5, n_samples),
        
        # Tem cartão de crédito cadastrado (0 ou 1)
        'tem_cartao': np.random.randint(0, 2, n_samples),
        
        # É membro ativo (0 ou 1)
        'membro_ativo': np.random.randint(0, 2, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Gerar target (churn) com base em regras lógicas para criar padrões
    # Clientes têm maior probabilidade de churn se:
    # - Satisfação baixa (1-2)
    # - Muitas chamadas ao suporte (>8)
    # - Pouco tempo como cliente (<12 meses)
    # - Não é membro ativo
    
    churn_probability = np.zeros(n_samples)
    
    # Aumentar probabilidade baseado em satisfação
    churn_probability += (6 - df['satisfacao']) * 0.15
    
    # Aumentar probabilidade baseado em chamadas ao suporte
    churn_probability += (df['chamadas_suporte'] / 15) * 0.25
    
    # Aumentar probabilidade para clientes novos
    churn_probability += np.where(df['tempo_cliente_meses'] < 12, 0.3, 0)
    
    # Diminuir probabilidade para membros ativos
    churn_probability -= df['membro_ativo'] * 0.2
    
    # Diminuir probabilidade para clientes com muitos produtos
    churn_probability -= (df['num_produtos'] / 4) * 0.15
    
    # Normalizar probabilidades entre 0 e 1
    churn_probability = np.clip(churn_probability, 0, 1)
    
    # Gerar churn baseado nas probabilidades
    df['churn'] = np.random.binomial(1, churn_probability)
    
    return df


def main():
    """Função principal para gerar e salvar os dados."""
    
    print("Gerando dados fictícios de clientes...")
    
    # Gerar dados
    df = generate_customer_data(n_samples=1000)
    
    # Criar diretório se não existir
    output_dir = Path(__file__).parent
    output_file = output_dir / 'customer_churn.csv'
    
    # Salvar CSV
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Dados gerados com sucesso!")
    print(f"✓ Arquivo salvo em: {output_file}")
    print(f"\nResumo dos dados:")
    print(f"  - Total de clientes: {len(df)}")
    print(f"  - Taxa de churn: {df['churn'].mean():.1%}")
    print(f"  - Clientes com churn: {df['churn'].sum()}")
    print(f"  - Clientes sem churn: {len(df) - df['churn'].sum()}")
    print(f"\nPrimeiras linhas:")
    print(df.head())
    print(f"\nEstatísticas descritivas:")
    print(df.describe())


if __name__ == "__main__":
    main()
