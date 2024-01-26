import pandas as pd
from module.utils import dataset_preprocessing, dataset_processing
from module.model_learning import SARIMAXPredictor


def preprocess_bodyfat_column(dataset):
    # Organizar DataFrame manipulando a coluna 'BodyFat'
    dataset = dataset.sort_values(by=["BodyFat"], ascending=False)
    dataset = dataset.iloc[:, :-7]

    if "BodyFat" in dataset.columns:
        body_fat_col = dataset["BodyFat"]
        dataset.drop(columns=["BodyFat"], inplace=True)
        dataset["BodyFat"] = body_fat_col
    else:
        print("A coluna 'BodyFat' não está presente no DataFrame.")

    return dataset


def pipeline_process(dataset, calories, weeks_data, inital_data):
    target = "body_weight"

    dataset = preprocess_bodyfat_column(dataset)

    # Pré-processamento e organização do DataFrame
    dataset = dataset_preprocessing(
        dataset_processing(dataset=dataset, calories=calories, start=inital_data)
    )

    # Separação entre treino e teste
    last_week_start = dataset["date"].max() - pd.DateOffset(days=weeks_data * 6)
    train_set = dataset[dataset["date"] < last_week_start]
    test_set = dataset[dataset["date"] >= last_week_start]

    print(last_week_start, test_set)

    # Exemplo de uso
    sarimax_predictor = SARIMAXPredictor(
        order=(1, 1, 1), seasonal_order=(1, 2, 2, 7), target=target
    )

    # Treinamento do modelo
    sarimax_predictor.fit(train_set)

    return sarimax_predictor.forecast(test_set)


if __name__ == "__main__":
    broker_address = "seu_broker"
    broker_port = 1883
    topic_name = ["dataset1", "calories"]

    # mqtt_client = MqttClient(broker_address, broker_port, topic_name=topic_name)
    # mqtt_client.connect()
    # mqtt_client.loop_forever()
    # dataset = process_json(jason_data=mqtt_client.data_jason)

    inital_data = "2024-01-26"

    calories = {
        "Sunday": 5000,
        "Monday": 2000,
        "Tuesday": 2000,
        "Wednesday": 2000,
        "Thursday": 2000,
        "Friday": 2500,
        "Saturday": 3200,
    }

    dataset = pd.read_csv("bodyfat.csv")

    ## Precisa pegar o Start date para passar para o model em start
    request_to_mqtt = pipeline_process(
        dataset, calories, weeks_data=1, inital_data=inital_data
    )

    print(request_to_mqtt)
