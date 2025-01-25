from multiprocessing import Queue
from threading import Lock
from itertools import product
import json

queue = Queue()
queue_lock = Lock()

def main():
    combine()

def combine():
    replications = 10
    model_names = ['alexnet', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet101', 'vgg11', 'vgg19']
    epochs = [10, 20]
    learning_rates = [0.001, 0.0001, 0.00001]
    weight_decays = [0, 0.0001]

    combinations = list(product(model_names, epochs, learning_rates, weight_decays))

    for model_name, epochs, learning_rate, weight_decay in combinations:
        json_data = json.dumps({
            "model_name": model_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay
        }, indent=4)
        queue.put(json_data)

def processRequest(receivedJson):
    with queue_lock:
        status = receivedJson.get('status')
        if status in ['ONLINE', 'FINISHED']:
            if status == 'FINISHED':
                data = receivedJson.get('data')
                if data:
                    with open("results.txt", "a") as file:
                        file.write(json.dumps(data) + "\n")
                else:
                    print("Aviso: Nenhum dado encontrado no JSON para salvar.")
            
            if not queue.empty():
                return queue.get()
            else:
                return None

main()