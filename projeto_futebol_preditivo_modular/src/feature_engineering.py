# src/feature_engineering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def engineer_features(df):
    """
    Cria features avançadas a partir do DataFrame pré-processado.
    - Extrai ano, mês, dia da semana da data.
    - Cria 'is_home_game', 'goal_difference', 'total_goals'.
    """
    if df is None:
        print("DataFrame de entrada é None. Não é possível engenheirar features.")
        return None

    print("\n--- Iniciando Engenharia de Features Avançada ---")

    # Converter a coluna 'date' para o formato datetime
    df['date'] = pd.to_datetime(df['date'])

    # Criando features baseadas na data
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek # Segunda-feira=0, Domingo=6

    # Criando feature de vantagem de jogar em casa (1 se não é neutro, 0 caso contrário)
    df['is_home_game'] = df.apply(lambda row: 1 if row['neutral'] == False else 0, axis=1)

    # Feature: Diferença de gols
    df['goal_difference'] = df['home_score'] - df['away_score']

    # Feature: Total de gols no jogo
    df['total_goals'] = df['home_score'] + df['away_score']

    print("Engenharia de features concluída.")
    return df

def analyze_correlation(df):
    """
    Realiza a Análise Exploratória de Dados (EDA) e análise de correlação.
    """
    if df is None:
        print("DataFrame de entrada é None. Não é possível analisar correlação.")
        return

    print("\n--- Análise Exploratória de Dados (EDA) e Correlação ---")

    # Distribuição dos resultados
    plt.figure(figsize=(8, 6))
    sns.countplot(x='result', data=df, palette='viridis', hue='result')
    plt.title('Distribuição dos Resultados dos Jogos')
    plt.xlabel('Resultado')
    plt.ylabel('Número de Jogos')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Distribuição dos gols do time da casa
    plt.figure(figsize=(10, 6))
    sns.histplot(df['home_score'], bins=range(0, df['home_score'].max() + 2), kde=True, color='skyblue')
    plt.title('Distribuição dos Gols Marcados pelo Time da Casa')
    plt.xlabel('Gols Marcados')
    plt.ylabel('Frequência')
    plt.xticks(range(0, df['home_score'].max() + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Distribuição dos gols do time visitante
    plt.figure(figsize=(10, 6))
    sns.histplot(df['away_score'], bins=range(0, df['away_score'].max() + 2), kde=True, color='lightcoral')
    plt.title('Distribuição dos Gols Marcados pelo Time Visitante')
    plt.xlabel('Gols Marcados')
    plt.ylabel('Frequência')
    plt.xticks(range(0, df['away_score'].max() + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Comparação de gols médios por tipo de resultado
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='result', y='home_score', data=df, palette='pastel')
    plt.title('Gols do Time da Casa por Resultado do Jogo')
    plt.xlabel('Resultado do Jogo')
    plt.ylabel('Gols do Time da Casa')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    plt.figure(figsize=(12, 7))
    sns.boxplot(x='result', y='away_score', data=df, palette='pastel')
    plt.title('Gols do Time Visitante por Resultado do Jogo')
    plt.xlabel('Resultado do Jogo')
    plt.ylabel('Gols do Time Visitante')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Para a análise de correlação, precisamos converter a variável 'result' em numérica.
    result_mapping = {'Home Win': 1, 'Draw': 0, 'Away Win': -1}
    df['result_numeric'] = df['result'].map(result_mapping)

    # Selecionando features numéricas para a matriz de correlação
    numerical_features = ['home_score', 'away_score', 'goal_difference', 'total_goals', 'year', 'month', 'day_of_week', 'is_home_game', 'result_numeric']
    correlation_matrix = df[numerical_features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Matriz de Correlação das Features Numéricas')
    plt.show()

    print("\nCorrelação das features com o resultado numérico (result_numeric):")
    print(correlation_matrix['result_numeric'].sort_values(ascending=False))

    # Testes de Hipótese (Exemplo: Mando de Campo)
    print("\n--- Testes de Hipótese: Mando de Campo ---")
    t_stat_goals, p_value_goals = ttest_ind(df['home_score'], df['away_score'])

    print(f"\nMédia de Gols do Time da Casa: {df['home_score'].mean():.2f}")
    print(f"Média de Gols do Time Visitante: {df['away_score'].mean():.2f}")
    print(f"Estatística T (comparação de gols): {t_stat_goals:.2f}")
    print(f"Valor P (comparação de gols): {p_value_goals:.3f}")

    if p_value_goals < 0.05:
        print("Há uma diferença estatisticamente significativa na média de gols entre times da casa e visitantes.")
    else:
        print("Não há uma diferença estatisticamente significativa na média de gols entre times da casa e visitantes.")

    print("\nAnálise de correlação e EDA concluídas.")

if __name__ == '__main__':
    # Exemplo de uso
    from src.data_ingestion import load_raw_data, save_data
    from src.data_preprocessing import preprocess_data
    from src.config import ANALYZED_DATA_PATH, CLEANED_DATA_PATH

    df_raw = load_raw_data(from_url=False)
    if df_raw is not None:
        df_cleaned = preprocess_data(df_raw.copy())
        if df_cleaned is not None:
            df_analyzed = engineer_features(df_cleaned.copy())
            if df_analyzed is not None:
                analyze_correlation(df_analyzed.copy())
                save_data(df_analyzed, ANALYZED_DATA_PATH)
