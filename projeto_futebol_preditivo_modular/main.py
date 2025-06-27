# main.py

import os
import pandas as pd

# Importa as funções de cada módulo
from src.data_ingestion import load_raw_data, save_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features, analyze_correlation
from src.model_training import train_models, save_model
from src.model_evaluation import evaluate_model, interpret_model
from src.monitoring_and_insights import load_model, simulate_new_data, monitor_and_insight
from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH, ANALYZED_DATA_PATH, FEATURES, TARGET

def run_dmaic_project():
    """
    Orquestra a execução de todas as fases do projeto DMAIC.
    """
    print("--- Iniciando Projeto de Análise Preditiva no Futebol (DMAIC) ---")

    # --- Fase 1: DEFINE (Definir o Problema e o Objetivo do Projeto) ---
    print("\n### Fase 1: DEFINE (Definir o Problema e o Objetivo do Projeto) ###")
    print("Esta fase foi definida conceitualmente e documentada no README.md.")
    print("Objetivo: Aumentar a acurácia da previsão de resultados de jogos de futebol em 15% e identificar os 3 principais fatores estatísticos.")
    
    # --- Fase 2: MEASURE (Medir o Desempenho Atual e Coletar Dados) ---
    print("\n### Fase 2: MEASURE (Medir o Desempenho Atual e Coletar Dados) ###")
    # Carregar dados brutos (tente do local primeiro, se não, da URL)
    df_raw = load_raw_data(from_url=False)
    if df_raw is None:
        print("Falha ao carregar dados brutos. Encerrando o projeto.")
        return

    # Pré-processar os dados
    df_cleaned = preprocess_data(df_raw.copy())
    if df_cleaned is None:
        print("Falha no pré-processamento dos dados. Encerrando o projeto.")
        return
    save_data(df_cleaned, CLEANED_DATA_PATH)

    # --- Fase 3: ANALYZE (Analisar as Causas-Raiz e Desenvolver Hipóteses) ---
    print("\n### Fase 3: ANALYZE (Analisar as Causas-Raiz e Desenvolver Hipóteses) ###")
    # Engenharia de features
    df_analyzed = engineer_features(df_cleaned.copy())
    if df_analyzed is None:
        print("Falha na engenharia de features. Encerrando o projeto.")
        return
    
    # Análise de correlação e EDA
    analyze_correlation(df_analyzed.copy()) # Passa uma cópia para evitar modificações
    save_data(df_analyzed, ANALYZED_DATA_PATH)

    # --- Fase 4: IMPROVE (Melhorar e Implementar Soluções/Modelos) ---
    print("\n### Fase 4: IMPROVE (Melhorar e Implementar Soluções/Modelos) ###")
    # Treinar e selecionar o melhor modelo
    best_model, best_model_name, X_test_df, y_test_df = train_models(df_analyzed.copy())
    if best_model is None:
        print("Falha no treinamento/seleção do modelo. Encerrando o projeto.")
        return
    
    # Salvar o melhor modelo
    save_model(best_model, best_model_name)

    # Avaliar o modelo e interpretar (se aplicável)
    evaluate_model(best_model, X_test_df, y_test_df)
    
    # Para interpretação, passamos o X e Y originais para que o preprocessor possa ser re-utilizado
    # e obter os nomes das features transformadas corretamente.
    interpret_model(best_model, best_model_name, df_analyzed[FEATURES], df_analyzed[TARGET])


    # --- Fase 5: CONTROL (Controlar e Sustentar as Melhorias) ---
    print("\n### Fase 5: CONTROL (Controlar e Sustentar as Melhorias) ###")
    # Carregar o modelo salvo (para simular um novo ciclo de monitoramento)
    loaded_model, loaded_model_name = load_model()
    if loaded_model is None:
        print("Falha ao carregar o modelo para monitoramento. Encerrando o projeto.")
        return

    # Simular novos dados e monitorar
    df_new_games = simulate_new_data(df_analyzed.copy())
    monitor_and_insight(loaded_model, loaded_model_name, df_new_games)

    print("\n--- Projeto de Análise Preditiva no Futebol (DMAIC) Concluído! ---")

if __name__ == '__main__':
    run_dmaic_project()
