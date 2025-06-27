# src/data_preprocessing.py

import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Realiza as etapas iniciais de limpeza e pré-processamento dos dados.
    - Trata valores ausentes (se houver).
    - Cria a variável alvo 'result'.
    """
    if df is None:
        print("DataFrame de entrada é None. Não é possível pré-processar.")
        return None

    print("\n--- Iniciando Pré-processamento de Dados ---")
    print("Verificando valores ausentes antes do pré-processamento:")
    print(df.isnull().sum())

    # Converter colunas de score para numérico, tratando possíveis erros
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype(int)
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype(int)

    # Tratar valores ausentes (ex: preencher com 0 ou remover linhas)
    # Para este dataset, scores não devem ter NaN, mas é uma boa prática.
    # Outras colunas como 'city' ou 'country' podem ter NaNs, mas OneHotEncoder lida com 'handle_unknown'.
    # df.dropna(subset=['home_score', 'away_score'], inplace=True) # Exemplo: remover se scores forem NaN

    # Criar a variável 'result' (variável alvo)
    def get_match_result(row):
        if row['home_score'] > row['away_score']:
            return 'Home Win'
        elif row['home_score'] < row['away_score']:
            return 'Away Win'
        else:
            return 'Draw'

    df['result'] = df.apply(get_match_result, axis=1)
    print("\nVariável 'result' criada com sucesso!")
    print("Contagem de cada tipo de resultado:")
    print(df['result'].value_counts())

    print("Pré-processamento de dados concluído.")
    return df

if __name__ == '__main__':
    # Exemplo de uso (requer um DataFrame de entrada)
    from src.data_ingestion import load_raw_data, save_data
    from src.config import CLEANED_DATA_PATH

    df_raw = load_raw_data(from_url=False)
    if df_raw is not None:
        df_cleaned = preprocess_data(df_raw.copy()) # Passa uma cópia para não modificar o original
        if df_cleaned is not None:
            print("\nPrimeiras 5 linhas do DataFrame pré-processado:")
            print(df_cleaned.head())
            save_data(df_cleaned, CLEANED_DATA_PATH)
