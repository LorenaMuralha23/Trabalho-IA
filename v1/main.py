from cnn import CNN
import torch
from torchvision import datasets
from torchvision.transforms import v2
import time

# Para gerar combinações de maneira automática
from itertools import product

import os


def define_transforms(height, width):
    data_transforms = {
        'train': v2.Compose([
            v2.Resize((height, width)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ]),
        'test': v2.Compose([
            v2.Resize((height, width)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
    }
    return data_transforms


def read_images(data_transforms):
    # Diretório base relativo ao local do script
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data/resumido')

    train_path = os.path.join(base_dir, 'train')
    validation_path = os.path.join(base_dir, 'validation')
    test_path = os.path.join(base_dir, 'test')

    # Verifica se os diretórios existem
    if not all(os.path.exists(path) for path in [train_path, validation_path, test_path]):
        raise FileNotFoundError(
            f"Um ou mais caminhos não foram encontrados no diretório base: {base_dir}")

    # Carrega os dados
    train_data = datasets.ImageFolder(
        train_path, transform=data_transforms['train'])
    validation_data = datasets.ImageFolder(
        validation_path, transform=data_transforms['test'])
    test_data = datasets.ImageFolder(
        test_path, transform=data_transforms['test'])

    return train_data, validation_data, test_data


if __name__ == '__main__':
    data_transforms = define_transforms(224, 224)
    train_data, validation_data, test_data = read_images(data_transforms)
    cnn = CNN(train_data, validation_data, test_data, 8)

    # Definir os valores dos parâmetros
    replications = 10
    model_names = ['alexnet', 'mobilenet_v3_large',
                   'mobilenet_v3_small', 'resnet18', 'resnet101', 'vgg11', 'vgg19']
    epochs = [10, 20]
    learning_rates = [0.001, 0.0001, 0.00001]
    weight_decays = [0, 0.0001]

    # Gera todas as combinações possíveis
    combinations = list(product(model_names, epochs,
                        learning_rates, weight_decays))

    i = 0

    with open("results.txt", "a") as file:
        # Looping para percorrer todas as combinações e realizar os experimentos
        for model_name, epochs, learning_rate, weight_decay in combinations:
            start_time = time.time()
            average_accuracy, better_replication = cnn.create_and_train_cnn(
                model_name, epochs, learning_rate, weight_decay, replications)
            end_time = time.time()

            duration = end_time - start_time
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            milliseconds = int((duration * 1000) % 1000)

            result_text = (
                f"Combination {i}:\n"
                f"\t{model_name} - {epochs} - {learning_rate} - {weight_decay}\n"
                f"\tAverage Accuracy: {average_accuracy}\n"
                f"\tBetter Replication: {better_replication}\n"
                f"\tExecution Time: {hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}\n\n"
            )
            print(result_text)
            i += 1

            file.write(result_text)
