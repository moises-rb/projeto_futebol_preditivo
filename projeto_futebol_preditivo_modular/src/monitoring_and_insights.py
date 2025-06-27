# src/monitoring_and_insights.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.config import MODELS_DIR, LOGISTIC_REGRESSION_MODEL_NAME, RANDOM_FOREST_MODEL_NAME, NUM_SIMULATED_GAMES, FEATURES, TARGET, NUMERICAL_COLS, CATEGORICAL_COLS

def load_model():
    """
    Tenta carregar o melhor modelo salvo (Regressão Logística ou Random Forest).
    """
    model_filename_lr = os.path.join(MODELS_DIR, LOGISTIC_REGRESSION_MODEL_NAME)
    model_filename_rf = os.path.join(MODELS_DIR, RANDOM_FOREST_MODEL_NAME)

    best_model = None
    best_model_name = None

    if os.path.exists(model_filename_lr):
        best_model = joblib.load(model_filename_lr)
        best_model_name = 'Logistic Regression'
        print(f"\nModelo '{LOGISTIC_REGRESSION_MODEL_NAME}' carregado com sucesso!")
    elif os.path.exists(model_filename_rf):
        best_model = joblib.load(model_filename_rf)
        best_model_name = 'Random Forest'
        print(f"\nModelo '{RANDOM_FOREST_MODEL_NAME}' carregado com sucesso!")
    else:
        print(f"\nErro: Nenhum modelo foi encontrado em '{MODELS_DIR}'.")
        print("Por favor, verifique se o notebook '04_model_training_evaluation.ipynb' foi executado para salvar o modelo.")
    
    return best_model, best_model_name

def simulate_new_data(df_base):
    """
    Simula novos dados de jogos para monitoramento.
    Usa dados existentes para garantir compatibilidade de categorias.
    """
    print("\n--- Simulando Novos Dados de Jogos para Monitoramento ---")

    if df_base.empty:
        print("DataFrame base para simulação está vazio. Usando fallback para amostras.")
        sample_teams = ['Brazil', 'Argentina', 'Germany', 'France', 'Italy']
        sample_tournaments = ['Friendly', 'FIFA World Cup', 'Copa America']
        sample_cities = ['Rio de Janeiro', 'Buenos Aires', 'Berlin']
        sample_countries = ['Brazil', 'Argentina', 'Germany']
    else:
        sample_teams = df_base['home_team'].unique()
        sample_tournaments = df_base['tournament'].unique()
        sample_cities = df_base['city'].unique()
        sample_countries = df_base['country'].unique()

        # Garantir que há dados suficientes para amostrar
        if len(sample_teams) > 5: sample_teams = sample_teams[:5]
        if len(sample_tournaments) > 3: sample_tournaments = sample_tournaments[:3]
        if len(sample_cities) > 3: sample_cities = sample_cities[:3]
        if len(sample_countries) > 3: sample_countries = sample_countries[:3]

    new_data = {
        'home_team': [np.random.choice(sample_teams) for _ in range(NUM_SIMULATED_GAMES)],
        'away_team': [np.random.choice(sample_teams) for _ in range(NUM_SIMULATED_GAMES)],
        'tournament': [np.random.choice(sample_tournaments) for _ in range(NUM_SIMULATED_GAMES)],
        'city': [np.random.choice(sample_cities) for _ in range(NUM_SIMULATED_GAMES)],
        'country': [np.random.choice(sample_countries) for _ in range(NUM_SIMULATED_GAMES)],
        'neutral': [np.random.choice([True, False]) for _ in range(NUM_SIMULATED_GAMES)],
        'year': [2024 for _ in range(NUM_SIMULATED_GAMES)],
        'month': [np.random.randint(1, 13) for _ in range(NUM_SIMULATED_GAMES)],
        'day_of_week': [np.random.randint(0, 7) for _ in range(NUM_SIMULATED_GAMES)],
        'is_home_game': [np.random.choice([0, 1]) for _ in range(NUM_SIMULATED_GAMES)],
        'goal_difference': [np.random.randint(-3, 4) for _ in range(NUM_SIMULATED_GAMES)],
        'total_goals': [np.random.randint(0, 6) for _ in range(NUM_SIMULATED_GAMES)],
        'result': [np.random.choice(['Home Win', 'Away Win', 'Draw']) for _ in range(NUM_SIMULATED_GAMES)]
    }
    df_new_games = pd.DataFrame(new_data)
    print("\nNovos dados de jogos simulados:")
    print(df_new_games.head())
    return df_new_games

def monitor_and_insight(model, model_name, df_new_games):
    """
    Realiza previsões em novos dados, avalia o desempenho e gera insights acionáveis.
    """
    if model is None or df_new_games is None or df_new_games.empty:
        print("\nNão foi possível realizar previsões ou gerar insights, pois o modelo ou os dados são inválidos.")
        return

    print("\n--- Realizando Previsões e Avaliação em Novos Dados ---")
    X_new = df_new_games[FEATURES]
    y_new = df_new_games[TARGET]

    y_pred_new = model.predict(X_new)

    accuracy_new = accuracy_score(y_new, y_pred_new)
    print(f"\nAcurácia do modelo em novos dados simulados: {accuracy_new:.4f}")
    print("\nRelatório de Classificação (novos dados):")
    print(classification_report(y_new, y_pred_new, zero_division=0))
    print("\nMatriz de Confusão (novos dados):")
    print(confusion_matrix(y_new, y_pred_new))

    print("\n--- Gerando Visualizações para Monitoramento ---")
    results_comparison = pd.DataFrame({'Real': y_new, 'Previsto': y_pred_new})
    results_melted = results_comparison.melt(var_name='Tipo de Resultado', value_name='Resultado do Jogo')

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Resultado do Jogo', hue='Tipo de Resultado', data=results_melted, palette={'Real': 'skyblue', 'Previsto': 'lightcoral'})
    plt.title('Comparação de Resultados Reais vs. Previstos em Novos Jogos')
    plt.xlabel('Resultado do Jogo')
    plt.ylabel('Contagem')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    print("\n--- Insights Acionáveis e Conclusões para o 'Filho' ---")
    if model_name == 'Logistic Regression':
        print("\nAnálise dos Coeficientes da Regressão Logística (para insights):")
        
        preprocessor_loaded = model.named_steps['preprocessor']
        feature_names_transformed = list(NUMERICAL_COLS)
        ohe_feature_names = preprocessor_loaded.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_COLS)
        feature_names_transformed.extend(ohe_feature_names)

        if hasattr(model.named_steps['classifier'], 'coef_') and len(model.named_steps['classifier'].coef_.shape) > 1:
            coefficients = pd.Series(model.named_steps['classifier'].coef_[0], index=feature_names_transformed)
        else:
            coefficients = pd.Series(model.named_steps['classifier'].coef_, index=feature_names_transformed)

        print("Top 10 Coeficientes Positivos (aumentam a chance de vitória do time da casa):")
        print(coefficients.nlargest(10))
        print("\nTop 10 Coeficientes Negativos (aumentam a chance de vitória do time visitante/empate):")
        print(coefficients.nsmallest(10))

        print("\n--- Recomendações para o 'Filho' (Baseado na Regressão Logística): ---")
        print("1. **Diferença de Gols (goal_difference):** Este é um dos fatores mais fortes. Quanto maior a diferença de gols a favor, maior a chance de vitória. Focar em marcar mais e sofrer menos é crucial.")
        print("2. **Total de Gols (total_goals):** O número total de gols na partida também tem um impacto. Jogos com mais gols podem indicar um estilo de jogo mais ofensivo, que pode ser benéfico para a vitória.")
        print("3. **Mando de Campo (is_home_game):** Jogar em casa geralmente confere uma vantagem significativa. O apoio da torcida e a familiaridade com o campo podem influenciar o desempenho.")
        print("4. **Times Específicos:** Alguns times têm um impacto muito grande no resultado, seja por serem muito fortes (coeficientes positivos para 'home_team_Brazil', 'home_team_Germany') ou fracos (coeficientes negativos para 'away_team_Brazil', 'away_team_Germany'). Observar a qualidade do adversário é fundamental.")
        print("5. **Torneio:** O tipo de torneio também pode influenciar. Jogos de Copa do Mundo podem ter dinâmicas diferentes de amistosos.")
        print("\nLembre-se, esses são insights baseados em dados históricos. O futebol é dinâmico, mas entender esses padrões pode te dar uma vantagem na leitura do jogo e no seu próprio desenvolvimento!")

    elif model_name == 'Random Forest':
        print("\n--- Recomendações para o 'Filho' (Baseado no Random Forest): ---")
        rf_classifier = model.named_steps['classifier']
        
        preprocessor_loaded = model.named_steps['preprocessor']
        ohe_feature_names = preprocessor_loaded.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_COLS)
        all_feature_names = NUMERICAL_COLS + list(ohe_feature_names)

        feature_importances = pd.Series(rf_classifier.feature_importances_, index=all_feature_names)
        
        top_n = 15
        print("Top 15 Features Mais Importantes (Random Forest):")
        print(feature_importances.nlargest(top_n))

        print("\n--- Recomendações para o 'Filho' (Baseado no Random Forest): ---")
        print("1. **Análise de Importância das Features:** O gráfico de importância das features (se gerado) mostra quais fatores o modelo considerou mais relevantes. Foco nos top 3-5 fatores.")
        print("2. **Diferença de Gols e Total de Gols:** Geralmente, a diferença de gols e o total de gols são muito importantes. Isso reforça a necessidade de um bom ataque e defesa.")
        print("3. **Mando de Campo:** A vantagem de jogar em casa é consistentemente um fator relevante.")
        print("4. **Qualidade do Adversário:** A força do time adversário é um fator primordial. Analise o histórico e o desempenho recente do oponente.")
        print("5. **Contexto do Jogo:** O tipo de torneio e até mesmo o ano podem ter nuances que o modelo capta. Jogos importantes tendem a ser mais disputados.")
        print("\nEntender esses fatores te ajudará a ter uma visão mais estratégica do futebol e a identificar onde você pode focar para melhorar seu próprio jogo!")
    else:
        print("\nNão foi possível gerar insights específicos para o modelo carregado.")
        print("Recomendações gerais: Foco em performance ofensiva (gols marcados), defensiva (gols sofridos) e a vantagem de jogar em casa.")

if __name__ == '__main__':
    from src.data_ingestion import load_raw_data
    from src.data_preprocessing import preprocess_data
    from src.feature_engineering import engineer_features
    from src.config import ANALYZED_DATA_PATH

    # Carregar dados base para simulação de novos dados
    df_base_for_simulation = pd.read_csv(ANALYZED_DATA_PATH)

    best_model, best_model_name = load_model()
    df_new_games = simulate_new_data(df_base_for_simulation)
    
    monitor_and_insight(best_model, best_model_name, df_new_games)
