import paho.mqtt.client as mqtt
import socket
import pandas as pd
import json


class MqttClient:
    def __init__(
        self, broker_address, broker_port, topic_names: list = None, save: bool = False
    ):
        self.client = mqtt.Client()
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.topic_names = topic_names
        self.callback_triggered = False
        self.save = save

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.data_json = {}
        self.data = {}

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))

        for topic in self.topic_names:
            client.subscribe(topic)

    def on_message(self, client, userdata, msg):
        # print(f"Recebido tópico: {msg.topic}, Mensagem: {msg.payload.decode()}")
        data = msg.payload.decode()
        self.data = data
        print("self.data: \n\n", self.data)
        self.callback_triggered = True

        try:
            self.data_json = json.loads(data)
            if self.save == True:
                with open("data.json", "w") as file:
                    json.dump(self.data_json, file, indent=2)
        except json.JSONDecodeError as e:
            print("Erro ao decodificar JSON:", e)

    def connect(self):
        self.client.connect(self.broker_address, self.broker_port, 60)

    def loop_forever(self):
        self.client.loop_forever()

    def publish(self, topic, message):
        self.client.publish(topic, message)

    def broker_verify(self):
        try:
            socket.gethostbyname(str(self.broker_address))  ## endereço Mqtt broker
            print("Conectividade de rede está OK.")
        except socket.gaierror as e:
            print(f"Erro de conectividade de rede: {e}")
