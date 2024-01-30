import sys
import json
import pandas as pd
from typing import Dict


def datasetGenerator(data_json: json = None) -> list[pd.DataFrame]:
    dataframes = dict()
    print()
    for chave, lista_dicionarios in data_json.items():
        df = pd.DataFrame(lista_dicionarios)
        dataframes[chave] = df

    return dataframes


class DataLoader:
    def __init__(self, data_json):
        self.dataset = self.extract_dataset(data_json)
        self.calories = self.extract_calories(data_json)
        self.weeks = self.extract_weeks(data_json)
        self.horizon_data = self.extract_horizon_data()

    def extract_dataset(self, data_json):
        dict_df = datasetGenerator(data_json)

        out_var = ["week_horizon", "calories"]
        list_key = list(dict_df.keys())
        if "selects_vars" in list_key:
            if "vars" in dict_df["selects_vars"]:
                out_var.extend(dict_df["selects_vars"]["vars"].tolist())

        for chave in list(dict_df.keys()):
            if chave in out_var:
                dict_df.pop(chave)

        return dict_df

    def extract_calories(self, data_json):
        week_calories = data_json.get("calories", [])
        calories_dict = {entry["weyDay"]: entry["colorie"] for entry in week_calories}

        return calories_dict

    def extract_weeks(self, data_json):
        return int(data_json.get("week_horizon", [])[0]["n_weeks"])

    def extract_horizon_data(self):
        list_data = self.dataset["weights"]["day"].tolist()

        return {"initial": list_data[1], "ended": list_data[-1]}
