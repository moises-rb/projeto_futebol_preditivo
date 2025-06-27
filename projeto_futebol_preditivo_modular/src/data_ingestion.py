import pandas as pd
import os
from src.config import RAW_DATA_PATH, GITHUB_RAW_DATA_URL

def load_raw_data(from_url=False):
    """
    Carrega o dataset bruto de resultados de futebol.
    Pode carregar de uma URL do GitHub ou de um caminho de arquivo local.
    """
    if from_url:
        print(f"Tentando carregar dados da URL: {GITHUB_RAW_DATA_URL}")
        try:
            df = pd.read_csv(GITHUB_RAW_DATA_URL)
            print("Dataset carregado com sucesso da URL!")
            return df
        except Exception as e:
            print(f"Erro ao carregar dados da URL: {e}")
            print("Tentando carregar do caminho local como fallback...")
            return _load_local_raw_data()
    else:
        return _load_local_raw_data()

def _load_local_raw_data():
    """
    Função auxiliar para carregar o dataset bruto de um caminho local.
    """
    print(f"Tentando carregar dados do caminho local: {RAW_DATA_PATH}")
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print("Dataset carregado com sucesso do caminho local!")
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em '{RAW_DATA_PATH}'.")
        print("Por favor, certifique-se de que 'results.csv' está em 'data/raw/'.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o arquivo local: {e}")
        return None

def save_data(df, path):
    """
    Salva um DataFrame em um arquivo CSV, criando o diretório se não existir.
    """
    output_dir = os.path.dirname(path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório '{output_dir}' criado com sucesso.")
    
    df.to_csv(path, index=False)
    print(f"Dados salvos com sucesso em: {path}")

if __name__ == '__main__':
    # Exemplo de uso
    df_raw = load_raw_data(from_url=False) # Tente carregar do local primeiro
    if df_raw is not None:
        print("\nPrimeiras 5 linhas do DataFrame bruto:")
        print(df_raw.head())
        print(f"\nNúmero de linhas: {len(df_raw)}")
        # Você pode salvar o df_raw para simular a próxima etapa se quiser
        # save_data(df_raw, RAW_DATA_PATH.replace('raw', 'processed'))