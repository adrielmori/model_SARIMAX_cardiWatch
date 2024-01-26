import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAXPredictor:
    def __init__(
        self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), target: str = None
    ):
        """
        Inicializa o modelo SARIMAX.

        Parâmetros:
        - order: Ordem do modelo ARIMA (p, d, q).
        - seasonal_order: Ordem sazonal do modelo SARIMA (P, D, Q, S).
            - P (ordem autoregressiva sazonal):
                Indica o número de termos autoregressivos sazonais.
            - D (ordem de diferenciação sazonal):
                Refere-se ao número de vezes que a série temporal é
            diferenciada sazonalmente para torná-la estacionária.
            - Q (ordem média móvel sazonal):
                Indica o número de termos da média móvel sazonal.
            S (comprimento sazonal):
                Representa a periodicidade sazonal da série temporal.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.target = target

    def fit(self, train_data):
        """
        Treina o modelo SARIMAX com os dados de treino.

        Parâmetros:
        - train_data: DataFrame contendo a série temporal de treino com uma coluna de datas e outra de valores.
        """
        exog_vars = train_data.drop(columns=["date", str(self.target)])
        self.model = SARIMAX(
            train_data[str(self.target)],
            exog=exog_vars,
            order=self.order,
            seasonal_order=self.seasonal_order,
        )
        self.results = self.model.fit(maxiter=22, method="lbfgs", ol=1e-20, gtol=1e-20)

    def forecast(self, test_data):
        """
        Realiza a previsão para a semana de teste.

        Parâmetros:
        - test_data: DataFrame contendo a série temporal de teste com uma coluna de datas.

        Retorna:
        - previsoes: DataFrame contendo as previsões para a semana de teste.
        """
        exog_vars = test_data.drop(columns=["date", str(self.target)])
        start_date = test_data["date"].min()
        end_date = test_data["date"].max()
        forecast_values = self.results.get_forecast(
            steps=(end_date - start_date).days + 1, exog=exog_vars
        )
        forecast_index = pd.date_range(start=start_date, end=end_date, freq="D")
        forecast_df = pd.DataFrame(
            {"forecast": forecast_values.predicted_mean.values}, index=forecast_index
        )
        return forecast_df

    # def plot_results(self, train_data, test_data, forecast_data):
    #     """
    #     Plota os resultados do modelo.

    #     Parâmetros:
    #     - train_data: DataFrame contendo a série temporal de treino com uma coluna de datas e outra de valores.
    #     - test_data: DataFrame contendo a série temporal de teste com uma coluna de datas.
    #     - forecast_data: DataFrame contendo as previsões para a semana de teste.
    #     """
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(train_data['date'], train_data['value'], label='Treino')
    #     plt.plot(test_data['date'], test_data['value'], label='Teste')
    #     plt.plot(forecast_data.index, forecast_data['forecast'], label='Previsão', linestyle='dashed', color='red')
    #     plt.title('Previsão SARIMAX')
    #     plt.xlabel('Data')
    #     plt.ylabel('Valor')
    #     plt.legend()
    #     plt.show()
