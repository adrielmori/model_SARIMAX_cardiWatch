# data_processing/data_processing.py
from datetime import datetime, timedelta
import pandas as pd
import json
import sys
import numpy as np
from typing import Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from dataloader.dataloader import DataLoader
import numpy as np

# Defina a semente
np.random.seed(25)


class DataSet:
    def __init__(self, dataset: Dict[str, pd.DataFrame]):
        self.dataset = dataset

    def filter_dataframes(self, start: int, end: int) -> None:
        """
        Filtra os DataFrames no DataLoader para incluir apenas as linhas onde 'day' está entre 'start' e 'end'.

        Parâmetros:
        - start (int): O primeiro dia do intervalo.
        - end (int): O último dia do intervalo.
        """
        for key in self.dataset.keys():
            df = self.dataset[key]
            self.dataset[key] = df[(df["day"] >= start) & (df["day"] <= end)]

    def bootstrap_fill(self, start: int, end: int) -> None:
        """
        Preenche os dias faltantes nos DataFrames no DataLoader usando a técnica de bootstrapping.

        Parâmetros:
        - start (int): O primeiro dia do intervalo.
        - end (int): O último dia do intervalo.
        """
        full_range = pd.DataFrame(range(start, end + 1), columns=["day"])

        for key in self.dataset.keys():
            df = self.dataset[key]
            merged_df = pd.merge(full_range, df, on="day", how="left")
            missing_rows = merged_df.isnull().any(axis=1)

            for column in df.columns:
                if column != "day":
                    bootstrap_sample = (
                        df[column]
                        .dropna()
                        .sample(n=missing_rows.sum(), replace=True)
                        .values
                    )
                    merged_df.loc[missing_rows, column] = bootstrap_sample

            self.dataset[key] = merged_df

    def process_dataset(self, start: int = None, end: int = None) -> None:
        """
        Processa o DataLoader, filtrando e preenchendo os DataFrames conforme necessário.

        Parâmetros:
        - start (int, opcional): O primeiro dia do intervalo. Se None, não filtra os DataFrames.
        - end (int, opcional): O último dia do intervalo. Se None, não filtra os DataFrames.
        """
        if start is not None and end is not None:
            self.filter_dataframes(start, end)
            self.bootstrap_fill(start, end)


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
        coluna for coluna in dataset.columns if coluna not in ["day", "weight"]
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
        coluna for coluna in dataset.columns if coluna not in ["day", "weight"]
    ]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset[colunas_a_normalizar] = scaler.fit_transform(dataset[colunas_a_normalizar])

    return dataset


def convert_index_to_date(dataloader):
    """
    Converte o índice numérico dos DataFrames no DataLoader para o formato de data.

    Parâmetros:
    - dataloader: O DataLoader contendo os DataFrames.
    """
    # Obtém a data atual
    today = datetime.today().date()

    df = dataloader
    start_date = today - pd.DateOffset(days=int(df.index[-1]))

    # Converte o índice numérico para data
    df.index = pd.to_datetime(
        df.index.map(lambda x: start_date + pd.DateOffset(days=x))
    )

    return df


def aggregate_dataframes(dataloader):
    for key in dataloader.dataset.keys():
        df = dataloader.dataset[key]

        # Agrupa por 'day' e calcula a média das outras colunas
        df = df.groupby("day").mean().reset_index()

        dataloader.dataset[key] = df


def getDataloader(data_json: json = None):
    print("GERANDO O DATALOADER: \n\n", data_json, "\n")

    return DataLoader(data_json)
