from models.models import GATNeuralAlgorithm, MPNNmaxNeuralAlgorithm, MPNNsumNeuralAlgorithm, MPNNmeanNeuralAlgorithm
from utils.data import generator
import torch
from torch import nn
from utils.algorithms import bfs_step
from utils.train import train_one_epoch, test, graph_accuracy

from config import CONFIG


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if CONFIG.model == "gat": model = GATNeuralAlgorithm()
    if CONFIG.model == "mpnn-sum": model = MPNNsumNeuralAlgorithm()
    if CONFIG.model == "mpnn-mean": model = MPNNmeanNeuralAlgorithm()
    if CONFIG.model == "mpnn-max": model = MPNNmaxNeuralAlgorithm()

    graph_criterion = nn.BCELoss()
    termination_criterion = nn.BCELoss()

    n_nodes,dim_grid= CONFIG.train_nodes, CONFIG.train_nodes_grid
    n_nodes_test,dim_grid_test= CONFIG.test_nodes, CONFIG.test_nodes_grid

    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG.learning_rate)

    train_items =   generator("bfs","erdos renyi",n_nodes,100) + \
                    generator("bfs","grid",dim_grid,100) + \
                    generator("bfs","ladder",n_nodes,100) + \
                    generator("bfs","tree",n_nodes,100) + \
                    generator("bfs","barabasi albert",n_nodes,100)
    val_items =     generator("bfs","erdos renyi",n_nodes,5) + \
                    generator("bfs","grid",dim_grid,5) + \
                    generator("bfs","ladder",n_nodes,5) + \
                    generator("bfs","tree",n_nodes,5) + \
                    generator("bfs","barabasi albert",n_nodes,5)
    test_items =    generator("bfs","erdos renyi",n_nodes_test,5) + \
                    generator("bfs","grid",dim_grid_test,5) + \
                    generator("bfs","ladder",n_nodes_test,5) + \
                    generator("bfs","tree",n_nodes,5) + \
                    generator("bfs","barabasi albert",n_nodes_test,5)

    print("TRAINING ----------------------------------")

    patience = CONFIG.patience
    counter = 0
    best_metric = 0
    for epoch in range(CONFIG.max_epochs):
        loss, mean_metric, last_step_metric = train_one_epoch(model, train_items, val_items, graph_criterion, termination_criterion, optimizer, epoch+1, graph_accuracy, device)
        if mean_metric > best_metric:
            best_metric = mean_metric
            counter = 0
        elif epoch>99:
            counter += 1
        
        if epoch>99 and counter>=patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
        
    print()
    print("TESTING ----------------------------------")
    _ = test(model, test_items, graph_criterion, termination_criterion, graph_accuracy, device)

if __name__ == "__main__":
    main()