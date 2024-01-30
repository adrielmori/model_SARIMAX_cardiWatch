import pandas as pd
import socket
import sys
from module.utils import (
    DataSet,
    dataset_preprocessing,
    getDataloader,
    aggregate_dataframes,
    convert_index_to_date,
)
from module.model_learning import SARIMAXPredictor
from module.conection import MqttClient
from dataloader.dataloader import DataLoader

from datetime import datetime, timedelta

import paho.mqtt.client as mqtt
import json
from pprint import pprint


def pipeline_process(dataloader: DataLoader, CONFIG_MODEL) -> json:
    # Pré-processamento e organização do DataFrame
    DataSet(dataloader.dataset).process_dataset(
        start=dataloader.horizon_data["initial"],
        end=dataloader.horizon_data["ended"],
    )
    aggregate_dataframes(dataloader)
    dataset = pd.concat(
        [df.set_index("day") for df in dataloader.dataset.values()], axis=1
    )

    dataset = convert_index_to_date(dataset_preprocessing(dataset))
    dataset.reset_index(level=0, inplace=True)
    dataset.rename(columns={"day": "date"}, inplace=True)

    print(dataloader.dataset["weights"])

    # Exemplo de uso
    sarimax_predictor = SARIMAXPredictor(
        order=CONFIG_MODEL["order"],
        seasonal_order=CONFIG_MODEL["seasonal_order"],
        target=CONFIG_MODEL["target"],
    )

    # Treinamento do modelo com todo o conjunto de dados
    sarimax_predictor.fit(dataset)
    pprint(sarimax_predictor.results.summary())

    dataset_temp = dataset
    dataset = dataset.set_index("date")

    horzon_proj = dataloader.weeks * 6 + 2

    # Previsão para os próximos dataloader.weeks * 6 dias
    future_dates = pd.date_range(
        start=dataset_temp["date"].max(), periods=int(horzon_proj)
    )[1:]
    future_df = pd.DataFrame(future_dates, columns=["date"])
    future_dataset = pd.DataFrame(
        index=future_dates[1:],
        columns=dataset.columns,
    )

    future_df = pd.concat([dataset, future_dataset], axis=0)
    test_exog = dataset.drop(columns=[str(CONFIG_MODEL["target"])]).tail(
        int(horzon_proj) - 1
    )
    preds = sarimax_predictor.results.predict(
        start=len(dataset), end=len(future_df), dinamic=True, exog=test_exog
    )

    preds = pd.DataFrame({"day": future_dates, "weight": preds})
    preds["day"] = pd.to_datetime(preds["day"])

    # Defina a data de início como o valor em dataloader.horizon_data["ended"]
    start_date = pd.to_datetime(dataloader.horizon_data["ended"])

    # Subtraia a data de início de cada data na coluna 'day' e converta para dias
    preds["day"] = (preds["day"] - start_date).dt.days

    # Suponha que 'df' é seu DataFrame
    dict_data = preds.to_dict("records")

    # Crie um dicionário final com a chave 'weights'
    request_to_mqtt = {"weights": dict_data}

    # Escreva o dicionário em um arquivo JSON
    with open("request_data.json", "w") as f:
        json.dump(request_to_mqtt, f)

    return request_to_mqtt


CONFIG_MODEL = {
    "model": "SARIMAX",
    "order": (1, 1, 3),
    "seasonal_order": (1, 1, 2, 7),
    "target": "weight",
}

if __name__ == "__main__":
    broker_address = "34.198.232.62"
    broker_port = 1883
    topic_names = ["cardiwatch", "messager"]

    mqtt_client = MqttClient(
        broker_address, broker_port, topic_names=topic_names, save=True
    )

    mqtt_client.broker_verify()
    mqtt_client.connect()
    mqtt_client.loop_forever()

    request_to_mqtt = pipeline_process(
        getDataloader(mqtt_client.data_json), CONFIG_MODEL
    )
    mqtt_client.publish("cardiwatch_request", json.dumps(request_to_mqtt, default=str))

    # with open("data.json", "r") as arquivo:
    #     dados_json = json.load(arquivo)
    # request_to_mqtt = pipeline_process(getDataloader(dados_json), CONFIG_MODEL)
