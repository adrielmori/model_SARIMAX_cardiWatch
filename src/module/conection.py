import paho.mqtt.client as mqtt


class MqttClient:
    def __init__(self, broker_address, broker_port, topic_name: list = None):
        self.client = mqtt.Client()
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.topic_name = topic_name

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.data_jason = {}

    def on_connect(self, client, userdata, flags, rc):
        print(f"Conectado com código de retorno: {rc}")

        for topic in self.topic_name:
            client.subscribe(str(topic))

    def on_message(self, client, userdata, msg):
        print(f"Recebido tópico: {msg.topic}, Mensagem: {msg.payload.decode()}")
        self.data_jason = msg.payload.decode()

    def connect(self):
        self.client.connect(self.broker_address, self.broker_port, 60)

    def loop_forever(self):
        self.client.loop_forever()

