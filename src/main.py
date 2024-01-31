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
from module.conection import *
from dataloader.dataloader import DataLoader

from datetime import datetime, timedelta

import paho.mqtt.client as mqtt
import json
from pprint import pprint


def process_dataset(dataloader: DataLoader):
    DataSet(dataloader.dataset).process_dataset(
        start=dataloader.horizon_data["initial"],
        end=dataloader.horizon_data["ended"],
    )
    aggregate_dataframes(dataloader)
    dataset = pd.concat(
        [df.set_index("day") for df in dataloader.dataset.values()], axis=1
    )
    return dataset


def preprocess_dataset(dataset):
    dataset = convert_index_to_date(dataset_preprocessing(dataset))
    dataset.reset_index(level=0, inplace=True)
    dataset.rename(columns={"day": "date"}, inplace=True)
    return dataset


def train_model(dataset, CONFIG_MODEL):
    sarimax_predictor = SARIMAXPredictor(
        order=CONFIG_MODEL["order"],
        seasonal_order=CONFIG_MODEL["seasonal_order"],
        trend=CONFIG_MODEL["trend"],
        target=CONFIG_MODEL["target"],
    )
    sarimax_predictor.fit(dataset)
    pprint(sarimax_predictor.results.summary())
    return sarimax_predictor


def predict_future(dataloader, dataset, sarimax_predictor, CONFIG_MODEL):
    dataset.index = pd.to_datetime(dataset.index)

    horizon_proj = dataloader.weeks * 6 + 2
    future_dates = pd.date_range(start=dataset.index.max(), periods=int(horizon_proj))[
        1:
    ]

    future_df = pd.DataFrame(future_dates, columns=["date"])
    future_dataset = pd.DataFrame(
        index=future_dates[1:],
        columns=dataset.columns,
    )
    future_df = pd.concat([dataset, future_dataset], axis=0)
    test_exog = dataset.drop(columns=[str(CONFIG_MODEL["target"])]).tail(
        int(horizon_proj) - 1
    )
    preds = sarimax_predictor.results.predict(
        start=len(dataset), end=len(future_df), dynamic=True, exog=test_exog
    )
    return preds, future_dates


def format_predictions(preds, future_dates, dataloader):
    preds = pd.DataFrame({"day": future_dates, "weight": preds})
    preds["day"] = pd.to_datetime(preds["day"])
    start_date = pd.to_datetime(dataloader.horizon_data["ended"])
    preds["day"] = (preds["day"] - start_date).dt.days + 1
    return preds


def write_to_json(preds):
    dict_data = preds.to_dict("records")
    request_to_mqtt = {"weights": dict_data}
    with open("request_data.json", "w") as f:
        json.dump(request_to_mqtt, f)
    return request_to_mqtt


def pipeline_process(dataloader: DataLoader, CONFIG_MODEL) -> json:
    dataset = process_dataset(dataloader)
    dataset = preprocess_dataset(dataset)
    sarimax_predictor = train_model(dataset, CONFIG_MODEL)
    dataset = dataset.set_index("date")
    preds, future_dates = predict_future(
        dataloader, dataset, sarimax_predictor, CONFIG_MODEL
    )
    preds = format_predictions(preds, future_dates, dataloader)
    request_to_mqtt = write_to_json(preds)
    return request_to_mqtt


CONFIG_MODEL = {
    "model": "SARIMAX",
    "order": (1, 1, 4),
    "seasonal_order": (1, 1, 5, 7),
    "trend": (0, 1, 1),
    "target": "weight",
}

if __name__ == "__main__":
    broker_address = "34.198.232.62"
    broker_port = 1883
    topic_names = ["cardiwatch", "messager"]

    mqtt_client = (
        MqttClient(  ## Logic Implementation to connect and Genereted the Digitial Twin
            broker_address, broker_port, topic_names=topic_names, save=True
        )
    )
    mqtt_client.broker_verify()
    mqtt_client.connect()
    mqtt_client.loop_forever()

    # with open("data.json", "r") as arquivo:
    #     data_json = json.load(arquivo)
    # request_to_mqtt = pipeline_process(getDataloader(data_json), CONFIG_MODEL)
    # print(request_to_mqtt)
