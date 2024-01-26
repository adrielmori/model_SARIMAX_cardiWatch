# data_processing/data_processing.py
import pandas as pd
import json
from typing import Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def process_json(json_data: Dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona os dados do JSON ao DataFrame existente.

    Parâmetros:
    - json_data: Dados no formato JSON.
    - df: DataFrame existente.

    Retorna:
    - df: DataFrame atualizado.
    """
    data_dict = json.loads(json_data)
    new_data = pd.DataFrame([data_dict])
    df = pd.concat([df, new_data], ignore_index=True)
    return df


def dataset_preprocessing(dataset: pd.DataFrame = None) -> pd.DataFrame:
    """
    Realiza a normalização padrão (StandardScaler) em colunas específicas do DataFrame.

    Parâmetros:
    - dataset: DataFrame a ser processado.

    Retorna:
    - dataset: DataFrame com colunas normalizadas.
    """
    if dataset is None:
        pass

    colunas_a_normalizar = [
        coluna for coluna in dataset.columns if coluna not in ["date", "body_weight"]
    ]

    scaler = StandardScaler()
    dataset[colunas_a_normalizar] = scaler.fit_transform(dataset[colunas_a_normalizar])

    return dataset


def dataset_preprocessing_minMax(dataset: pd.DataFrame = None) -> pd.DataFrame:
    """
    Realiza a normalização Min-Max em colunas específicas do DataFrame.

    Parâmetros:
    - dataset: DataFrame a ser processado.

    Retorna:
    - dataset: DataFrame com colunas normalizadas.
    """
    if dataset is None:
        pass

    colunas_a_normalizar = [
        coluna for coluna in dataset.columns if coluna not in ["date", "body_weight"]
    ]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset[colunas_a_normalizar] = scaler.fit_transform(dataset[colunas_a_normalizar])

    return dataset


def dataset_processing(dataset: pd.DataFrame, calories: Dict, start: str = None) -> pd.DataFrame:
    """
    Processa o DataFrame, renomeando colunas, adicionando a coluna 'date', 'calories' e reorganizando as colunas.

    Parâmetros:
    - dataset: DataFrame a ser processado.
    - calories: Dicionário de calorias por dia da semana.

    Retorna:
    - dataset: DataFrame processado.
    """
    colunas = [
        "steps",
        "median_heart_rate",
        "sleep_point",
        "time_sleep",
        "calories_consumed",
        "calories_burn",
        "stress",
        "body_weight",
    ]

    # Trocar o nome das colunas
    df_temp = pd.DataFrame()
    dataset = dataset.rename(columns=dict(zip(dataset.columns, colunas)))

    # Criar coluna 'date'
    dataset["date"] = pd.date_range(start=start, periods=len(dataset), freq="D")
    dataset["day_of_week"] = dataset["date"].dt.day_name()
    dataset["calories"] = dataset["day_of_week"].map(calories)

    dataset = dataset.drop(columns=["day_of_week"])
    colunas = ["date", "calories"] + [
        coluna for coluna in dataset if coluna not in ["date", "calories"]
    ]
    dataset = dataset[colunas]

    return dataset
