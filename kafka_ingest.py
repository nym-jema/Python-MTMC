import json
from confluent_kafka import Consumer

def make_consumer(bootstrap, group="py-mtmc"):
    conf = {"bootstrap.servers": bootstrap, "group.id": group, "auto.offset.reset":"earliest"}
    return Consumer(conf)

def consume_loop(consumer, topic, handler):
    consumer.subscribe([topic])
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None: 
                continue
            if msg.error():
                print("Kafka error", msg.error()); continue
            
            payload = json.loads(msg.value().decode('utf-8'))
            handler(payload)
    finally:
        consumer.close()