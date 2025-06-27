# src/config.py

import os

# Define o diretório base do projeto, assumindo que 'src' está na raiz
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Caminhos dos dados
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'results.csv')
CLEANED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_data.csv')
ANALYZED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'analyzed_data.csv')

# Caminho para salvar e carregar modelos
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGISTIC_REGRESSION_MODEL_NAME = 'logistic_regression_model.joblib'
RANDOM_FOREST_MODEL_NAME = 'random_forest_model.joblib'

# URL do dataset bruto no GitHub (se preferir carregar diretamente)
# Substitua 'SeuUsuario' e 'projeto_futebol_preditivo' pelo seu usuário e nome do repositório
GITHUB_RAW_DATA_URL = 'https://raw.githubusercontent.com/moises-rb/projeto_futebol_preditivo/main/02_measure/data/raw/results.csv'

# Colunas de features e alvo
FEATURES = ['home_team', 'away_team', 'tournament', 'city', 'country',
            'neutral', 'year', 'month', 'day_of_week', 'is_home_game',
            'goal_difference', 'total_goals']
TARGET = 'result'

# Colunas para pré-processamento
NUMERICAL_COLS = ['year', 'month', 'day_of_week', 'goal_difference', 'total_goals']
CATEGORICAL_COLS = ['home_team', 'away_team', 'tournament', 'city', 'country', 'neutral', 'is_home_game']

# Parâmetros de simulação para novos dados
NUM_SIMULATED_GAMES = 5
