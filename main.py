# -*- coding: utf-8 -*-
# @Author : Zhang
# @Email : zl16035056@163.com
# @File : main.py


import copy
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from utils.create_dataset import create_iid_clients, create_noniid_clients, check_labels
from models.client import Client
from models.server import Server
from utils.dpsgd_utils import set_epsilons, compute_noise_multiplier
from utils.budgets_accountant import BudgetsAccountant
from utils.dataloader import loader


def test(model, x_val, y_val):
    model.eval().to(args.device)
    data_loader = TensorDataset(x_val, y_val)
    data_loader = DataLoader(data_loader, batch_size=128, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for x_test, y_test in data_loader:
            x_test, y_test = x_test.to(args.device), y_test.to(args.device)

            output = model(x_test)
            loss = criterion(output, y_test)

            _, test_pred = torch.max(output, 1)

            correct = (test_pred == y_test).sum()

            test_acc += correct.item()
            test_loss += loss.item()

    print()
    return test_acc / len(y_val), test_loss / len(y_val)


def prepare_local_dataset(noniid, num_clients, y_train):
    if not noniid:
        dataset = create_iid_clients(num_clients=num_clients,
                                     num_examples=len(y_train),
                                     num_classes=10,
                                     num_examples_per_client=len(y_train) // 10,
                                     num_classes_per_client=10)
    else:
        dataset = create_noniid_clients(num_clients=num_clients,
                                        num_examples=len(y_train),
                                        num_classes=10,
                                        num_examples_per_client=len(y_train) // 10,
                                        num_classes_per_client=10)

    check_labels(10, dataset, y_train)
    return dataset


def prepare_privacy_preferences(epsfile, num_clients):
    epsilons = None
    if args.dp:
        epsilons = set_epsilons(epsfile, num_clients)
    return epsilons


def main(args):
    if args.dp:
        print('Using differential privacy!\n')
    else:
        print('No differential privacy!\n')

    # prepare local dataset
    x_train, y_train, x_test, y_test = loader(args.dataset)
    dataset = prepare_local_dataset(args.noniid, args.num_clients, y_train)

    # set privacy preference
    privacy_preferences = prepare_privacy_preferences(args.eps, args.all_clients)
    print('privacy preferences: \n', privacy_preferences)
    print()

    # set clients
    clients = []
    for i in range(args.all_clients):
        client = Client(x_train=x_train,
                        y_train=y_train,
                        dataset=dataset[i],
                        batch_size=args.batch_size,
                        dp=args.dp,
                        local_round=args.local_round,
                        grad_norm=args.grad_norm,
                        device=args.device)

        # set noise multiplier
        if args.dp:
            epsilon = privacy_preferences[i]
            noise_multiplier = compute_noise_multiplier(local_dataset_size=len(client.dataset_size),
                                                        local_batch_size=args.batch_size,
                                                        T=args.global_round * args.sample_ratio,
                                                        epsilon=epsilon,
                                                        delta=args.delta)
            print('the %d client noise multiplier is %f' % ((i+1), noise_multiplier))

            budget_accountant = BudgetsAccountant(epsilon, args.delta, noise_multiplier)
            client.set_budget_accountant(budget_accountant)

        clients.append(client)

    # set server
    server = Server(num_clients=args.num_clients, device=args.device, sample_ratio=args.sample_ratio)

    # set public client
    server.set_public_clients(privacy_preferences) if args.PFA or args.PFA_plus else None

    # init server algorithm
    server.init_alg(dp=args.dp,
                    Fedavg=args.Fedavg,
                    weiavg=args.Weiavg,
                    PFA=args.PFA,
                    PFA_plus=args.PFA_plus,
                    proj_dims=args.proj_dims,
                    lanczos_iter=args.lanczos_iter)

    # init global model
    server_model = server.init_global_model()

    # communication round
    communication_round = args.global_round // args.local_round

    accuracy_accountant = []
    privacy_accountant = []
    Vks, means = None, None
    accum_nbytes1 = 0  # before PFA_plus
    accum_nbytes2 = 0  # after PFA_plus
    accum_nbytes_list1 = []
    accum_nbytes_list2 = []

    # start communication
    for r in range(communication_round):
        print()
        print('the %d communication round \n' % (r + 1))
        # pick up candidates
        candidates = server.sample_clients(args.all_clients)

        # local update and aggregate
        for c, participant in enumerate(candidates):
            print("the %dth participant local update:" % (c+1))

            # delivery model
            clients[participant].download(copy.deepcopy(server_model))
            if Vks:
                clients[participant].set_projection(Vks, means, is_private=(participant not in server.public))

            # update
            model_state, bytes1, bytes2 = clients[participant].local_update()

            # aggregate
            server.aggregate(participant, model_state, args.Fedavg, args.PFA, args.PFA_plus)

        # load average weight
        global_model = server.update()

        # get projection information
        if args.PFA_plus:
            Vks, means = server.get_projection_info()

        # test
        test_acc, test_loss = test(global_model, x_test, y_test)
        print('current global model has test acc: %.4f  test loss: %.4f' % (test_acc, test_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--Fedavg', type=bool, default=False)
    parser.add_argument('--Weiavg', type=bool, default=False)
    parser.add_argument('--PFA', type=bool, default=False)
    parser.add_argument('--PFA_plus', type=bool, default=True)
    parser.add_argument('--proj_dims', type=int, default=1)
    parser.add_argument('--lanczos_iter', type=int, default=256)
    parser.add_argument('--global_round', type=int, default=100)
    parser.add_argument('--local_round', type=int, default=20)
    parser.add_argument('--noniid', type=bool, default=False, help='if True, use noniid data')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dp', type=bool, default=True, help='if True, use differential privacy')
    parser.add_argument('--eps', type=str, default='mixgauss1', help='epsilon file name')
    parser.add_argument('--delta', type=float, default=1e-5, help='differential privacy parameter')
    parser.add_argument('--grad_norm', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sample_ratio', type=float, default=0.8)
    parser.add_argument('--all_clients', type=int, default=10)
    args = parser.parse_args()

    main(args)
