from cnn import CNN
import torch
from torchvision import datasets
from torchvision.transforms import v2
import time
import os
import json
import socket

class Main:
    def define_transforms(self, height, width):
        data_transforms = {
            'train' : v2.Compose([
                        v2.Resize((height,width)),
                        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'test'  : v2.Compose([
                        v2.Resize((height,width)),
                        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        }
        return data_transforms

    def read_images(self, data_transforms):
        # Diretório base relativo ao local do script
        base_dir = os.path.join(os.path.dirname(__file__), '../..', 'data/resumido')

        train_path = os.path.join(base_dir, 'train')
        validation_path = os.path.join(base_dir, 'validation')
        test_path = os.path.join(base_dir, 'test')

        # Verifica se os diretórios existem
        if not all(os.path.exists(path) for path in [train_path, validation_path, test_path]):
            raise FileNotFoundError(f"Um ou mais caminhos não foram encontrados no diretório base: {base_dir}")

        # Carrega os dados
        train_data = datasets.ImageFolder(train_path, transform=data_transforms['train'])
        validation_data = datasets.ImageFolder(validation_path, transform=data_transforms['test'])
        test_data = datasets.ImageFolder(test_path, transform=data_transforms['test'])

        return train_data, validation_data, test_data
    
    def createJson(self, status, acc_media, rep_max, duration):
        try:
            # Obter o IP da interface de rede principal
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))  # Conecta ao Google DNS para determinar o IP
                ip_address = s.getsockname()[0]
            
            # Criar o dicionário com os dados
            createdJson = {
                "machine_id": 'worker-01',
                "status": status,
                "acc_media": acc_media,
                "rep_max": rep_max,
                "duration": duration
            }

            # Converter o dicionário para JSON
            createdJson = json.dumps(createdJson, indent=4)

            return createdJson

        except Exception as e:
            return json.dumps({"error": str(e)}, indent=4)         

    def processTask(self, cnn, replications, model_name, epochs, learning_rate, weight_decay):
        start_time = time.time()
        acc_media, rep_max = cnn.create_and_train_cnn(model_name, epochs, learning_rate, weight_decay, replications)
        end_time = time.time()
        duration = end_time - start_time
        print(self.createJson('FINISHED', acc_media, rep_max, duration))