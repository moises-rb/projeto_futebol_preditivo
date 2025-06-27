# src/model_training.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.config import FEATURES, TARGET, NUMERICAL_COLS, CATEGORICAL_COLS, MODELS_DIR

def get_preprocessor():
    """
    Retorna um ColumnTransformer para pré-processamento de dados.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS)
        ])
    return preprocessor

def train_models(df):
    """
    Prepara os dados, treina e avalia modelos de Machine Learning.
    Retorna o melhor modelo treinado e seu nome.
    """
    if df is None:
        print("DataFrame de entrada é None. Não é possível treinar modelos.")
        return None, None

    print("\n--- Treinando Modelos de Machine Learning ---")

    X = df[FEATURES]
    y = df[TARGET]

    # Divisão do dataset em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")

    preprocessor = get_preprocessor()

    # Modelo 1: Regressão Logística
    pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))])

    print("\nTreinando Regressão Logística...")
    pipeline_lr.fit(X_train, y_train)
    print("Regressão Logística treinada!")

    # Modelo 2: Random Forest Classifier
    pipeline_rf = Pipeline(steps=[('preprocessor', get_preprocessor()), # Nova instância do preprocessor para evitar side effects
                                  ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))])

    print("\nTreinando Random Forest...")
    pipeline_rf.fit(X_train, y_train)
    print("Random Forest treinada!")

    models = {
        'Logistic Regression': pipeline_lr,
        'Random Forest': pipeline_rf
    }

    best_model_name = None
    best_accuracy = 0
    best_model = None

    # Avaliação simples para selecionar o melhor modelo (detalhes em model_evaluation)
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n--- {name} ---")
        print(f"Acurácia: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model
    
    print(f"\n--- Melhor Modelo Selecionado: {best_model_name} com Acurácia de {best_accuracy:.4f} ---")
    
    return best_model, best_model_name, X_test, y_test # Retorna X_test e y_test para avaliação detalhada

def save_model(model, model_name, path=MODELS_DIR):
    """
    Salva o modelo treinado em um arquivo .joblib.
    """
    if model is None:
        print("Modelo é None. Não é possível salvar.")
        return

    os.makedirs(path, exist_ok=True) # Cria o diretório se não existir
    filename = os.path.join(path, f'{model_name.replace(" ", "_").lower()}_model.joblib')
    joblib.dump(model, filename)
    print(f"Modelo '{model_name}' salvo em: {filename}")

if __name__ == '__main__':
    from src.data_ingestion import load_raw_data
    from src.data_preprocessing import preprocess_data
    from src.feature_engineering import engineer_features
    from src.config import ANALYZED_DATA_PATH

    # Carregar dados processados
    df = pd.read_csv(ANALYZED_DATA_PATH)
    
    best_model, best_model_name, X_test_df, y_test_df = train_models(df.copy())
    if best_model is not None:
        save_model(best_model, best_model_name)
