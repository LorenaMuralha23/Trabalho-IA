import json
import socket
import os
from queue import Queue
from multiprocessing import Process, active_children
from cnn import CNN
import torch
from torchvision import datasets
from torchvision.transforms import v2
import time
from main import Main

# Cria uma fila
task_queue = Queue()
main = Main()

data_transforms = main.define_transforms(224, 224)
train_data, validation_data, test_data = main.read_images(data_transforms)
cnn = CNN(train_data, validation_data, test_data, 8)

def main():
    json = {
        "data": [
            {"replications": 2, "model_name": "alexnet", "epochs": 10, "learning_rate": 0.001, "weight_decay": 0},
            {"replications": 2, "model_name": "alexnet", "epochs": 10, "learning_rate": 0.001, "weight_decay": 0.0001},
            {"replications": 2, "model_name": "alexnet", "epochs": 10, "learning_rate": 0.0001, "weight_decay": 0},
            {"replications": 2, "model_name": "alexnet", "epochs": 10, "learning_rate": 0.0001, "weight_decay": 0.0001}
        ]
    }
    receiveTask(json)

# É nesse método que a mensagem do front end deve chegar.
# Ela precisa ser mapeada por {IP/Porta}.
def receiveTask(receivedJson):
    # Verifica se o campo 'data' existe no JSON e se é uma lista
    if 'data' in receivedJson and isinstance(receivedJson['data'], list):
        # Adiciona cada combinação à fila
        for combination in receivedJson['data']:
            processTask(combination)
    else:
        # Lida com o caso em que 'data' não é uma lista ou não está presente
        print("O JSON recebido não contém um campo 'data' válido.")

# É o método responsável por enviar as mensagens para o front end
def sendJson(jsonToSend):
    return jsonToSend

def createJson(status):
    try:
        # Obter o IP da interface de rede principal
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # Conecta ao Google DNS para determinar o IP
            ip_address = s.getsockname()[0]
        
        # Obter o número de núcleos
        num_cores = os.cpu_count()

        # Criar o dicionário com os dados
        createdJson = {
            "machine_id": 'worker-01',
            "ip_address": ip_address,
            "port": 5000,
            "status": status,
            "num_cores": num_cores
        }

        # Converter o dicionário para JSON
        createdJson = json.dumps(createdJson, indent=4)

        return createdJson

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=4)
    
def process_task_wrapper(cnn, repl, mn, epochs, lr, wd):
    main = Main()  # Cria uma nova instância no processo
    main.processTask(cnn, repl, mn, epochs, lr, wd)

def processTask(combination):
    repl = combination.get('replications')
    mn = combination.get('model_name')
    epochs = combination.get('epochs')
    lr = combination.get('learning_rate')
    wd = combination.get('weight_decay')
    task = Process(target=process_task_wrapper, args=(cnn, repl, mn, epochs, lr, wd))
    task.start()
    print(f"Processo iniciado: PID={task.pid}, Nome={task.name}")
    print(f"Processos ativos no momento: {len(active_children())}")

if __name__ == "__main__":
    main()