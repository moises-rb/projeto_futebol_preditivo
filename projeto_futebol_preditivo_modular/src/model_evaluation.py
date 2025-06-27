# src/model_evaluation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.config import FEATURES, TARGET, NUMERICAL_COLS, CATEGORICAL_COLS
from src.model_training import get_preprocessor # Para obter o preprocessor para nomes de features

def evaluate_model(model, X_test, y_test):
    """
    Avalia o desempenho do modelo em um conjunto de teste.
    Imprime métricas de classificação e gera uma matriz de confusão.
    """
    if model is None:
        print("Modelo é None. Não é possível avaliar.")
        return

    print("\n--- Avaliando o Desempenho do Modelo ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    # Plotar Matriz de Confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.show()

def interpret_model(model, model_name, X_train_original, y_train_original):
    """
    Interpreta o modelo para identificar as features mais importantes.
    Gera gráficos de importância de features para Random Forest ou coeficientes para Regressão Logística.
    """
    if model is None:
        print("Modelo é None. Não é possível interpretar.")
        return

    print(f"\n--- Interpretação do Modelo ({model_name}) ---")

    # Re-fit do preprocessor no X_train_original para obter todos os nomes das features transformadas
    # Isso é necessário para garantir que o preprocessor do modelo tenha todas as categorias
    # vistas no treino, para mapear corretamente os coeficientes/importâncias.
    preprocessor_loaded = model.named_steps['preprocessor']
    
    # Nomes das colunas numéricas
    numerical_cols_for_coef = NUMERICAL_COLS
    # Nomes das colunas categóricas após OneHotEncoding
    categorical_cols_for_coef = CATEGORICAL_COLS

    ohe_feature_names = preprocessor_loaded.named_transformers_['cat'].get_feature_names_out(categorical_cols_for_coef)
    all_feature_names = numerical_cols_for_coef + list(ohe_feature_names)


    if model_name == 'Logistic Regression':
        print("\nAnálise dos Coeficientes da Regressão Logística (para insights):")
        
        # Coeficientes do classificador (primeira linha para classificação multiclasse)
        if hasattr(model.named_steps['classifier'], 'coef_') and len(model.named_steps['classifier'].coef_.shape) > 1:
            coefficients = pd.Series(model.named_steps['classifier'].coef_[0], index=all_feature_names)
        else: # Para binário ou se coef_ for 1D
            coefficients = pd.Series(model.named_steps['classifier'].coef_, index=all_feature_names)

        # Exibir os coeficientes mais impactantes (positivos e negativos)
        print("Top 10 Coeficientes Positivos (aumentam a chance de vitória do time da casa):")
        print(coefficients.nlargest(10))
        print("\nTop 10 Coeficientes Negativos (aumentam a chance de vitória do time visitante/empate):")
        print(coefficients.nsmallest(10))

        plt.figure(figsize=(12, 8))
        coefficients.nlargest(10).plot(kind='barh', color='skyblue')
        plt.title('Top 10 Coeficientes Positivos (Regressão Logística)')
        plt.xlabel('Valor do Coeficiente')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))
        coefficients.nsmallest(10).plot(kind='barh', color='lightcoral')
        plt.title('Top 10 Coeficientes Negativos (Regressão Logística)')
        plt.xlabel('Valor do Coeficiente')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()


    elif model_name == 'Random Forest':
        print("\n--- Importância das Features (Random Forest) ---")
        rf_classifier = model.named_steps['classifier']
        
        feature_importances = pd.Series(rf_classifier.feature_importances_, index=all_feature_names)
        
        top_n = 15
        print(feature_importances.nlargest(top_n))

        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_importances.nlargest(top_n).values, y=feature_importances.nlargest(top_n).index, palette='viridis')
        plt.title(f'Top {top_n} Features Mais Importantes (Random Forest)')
        plt.xlabel('Importância')
        plt.ylabel('Feature')
        plt.show()
    else:
        print("Interpretação de features não implementada para este tipo de modelo.")

if __name__ == '__main__':
    from src.data_ingestion import load_raw_data
    from src.data_preprocessing import preprocess_data
    from src.feature_engineering import engineer_features
    from src.model_training import train_models
    from src.config import ANALYZED_DATA_PATH

    # Carregar dados processados
    df = pd.read_csv(ANALYZED_DATA_PATH)
    
    # Re-treinar para obter X_test e y_test
    best_model, best_model_name, X_test_df, y_test_df = train_models(df.copy())
    
    if best_model is not None:
        evaluate_model(best_model, X_test_df, y_test_df)
        interpret_model(best_model, best_model_name, df[FEATURES], df[TARGET]) # Passa X e y completos para o preprocessor
