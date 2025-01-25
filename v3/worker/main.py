import ray
from itertools import product
from cnn import CNN
import torch
from torchvision import datasets
from torchvision.transforms import v2
import os
import time

# Define as transformações dos dados


def define_transforms(height, width):
    data_transforms = {
        'train': v2.Compose([
            v2.Resize((height, width)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ]),
        'test': v2.Compose([
            v2.Resize((height, width)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
    }
    return data_transforms

# Função para ler as imagens


def read_images(data_transforms):
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data/resumido')
    train_path = os.path.join(base_dir, 'train')
    validation_path = os.path.join(base_dir, 'validation')
    test_path = os.path.join(base_dir, 'test')

    if not all(os.path.exists(path) for path in [train_path, validation_path, test_path]):
        raise FileNotFoundError(
            f"Um ou mais caminhos não foram encontrados no diretório base: {base_dir}")

    train_data = datasets.ImageFolder(
        train_path, transform=data_transforms['train'])
    validation_data = datasets.ImageFolder(
        validation_path, transform=data_transforms['test'])
    test_data = datasets.ImageFolder(
        test_path, transform=data_transforms['test'])

    return train_data, validation_data, test_data


# Função remota para executar uma combinação de parâmetros
@ray.remote
def execute_combination(train_data_ref, validation_data_ref, test_data_ref, batch_size, model_name, epochs, learning_rate, weight_decay, replications):
    train_data = ray.get(train_data_ref)
    validation_data = ray.get(validation_data_ref)
    test_data = ray.get(test_data_ref)

    cnn = CNN(train_data, validation_data, test_data, batch_size)
    start_time = time.time()
    average_accuracy, better_replication = cnn.create_and_train_cnn(
        model_name, epochs, learning_rate, weight_decay, replications)
    end_time = time.time()

    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    milliseconds = int((duration * 1000) % 1000)

    return (
        model_name, epochs, learning_rate, weight_decay,
        average_accuracy, better_replication,
        f"{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}"
    )


if __name__ == '__main__':
    # Inicializa o Ray
    ray.init(address="auto")

    data_transforms = define_transforms(224, 224)
    train_data, validation_data, test_data = read_images(data_transforms)

    # Armazena datasets no Ray
    train_data_ref = ray.put(train_data)
    validation_data_ref = ray.put(validation_data)
    test_data_ref = ray.put(test_data)

    # Parâmetros
    replications = 10
    model_names = ['alexnet', 'mobilenet_v3_large', 'vgg11']
    epochs = [10, 20]
    learning_rates = [0.001, 0.0001]
    weight_decays = [0, 0.0001]
    batch_size = 8

    combinations = list(product(model_names, epochs,
                                learning_rates, weight_decays))

    # Cria tarefas distribuídas
    futures = [
        execute_combination.remote(
            train_data_ref, validation_data_ref, test_data_ref, batch_size,
            model_name, epochs, learning_rate, weight_decay, replications
        )
        for model_name, epochs, learning_rate, weight_decay in combinations
    ]

    # Aguarda os resultados
    results = ray.get(futures)

    # Salva os resultados
    with open("results_ray.txt", "a") as file:
        for i, result in enumerate(results):
            model_name, epochs, learning_rate, weight_decay, acc, best_rep, exec_time = result
            result_text = (
                f"Combination {i}:\n"
                f"\t{model_name} - {epochs} - {learning_rate} - {weight_decay}\n"
                f"\tAverage Accuracy: {acc}\n"
                f"\tBetter Replication: {best_rep}\n"
                f"\tExecution Time: {exec_time}\n\n"
            )
            print(result_text)
            file.write(result_text)

    ray.shutdown()
