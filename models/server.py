# -*- coding: utf-8 -*-
# @Author : Zhang
# @Email : zl16035056@163.com
# @File : server.py

import numpy as np
import torch
from models.cnn import CNN
from utils.lanczos import Lanczos


def add_weights(num_vars, model_state, agg_model_state):
    # for i in range(num_vars):
    #     if not len(agg_model_dict):
    #         a = torch.unsqueeze(model_dict[i], 0)
    #     else:
    #         b = torch.cat([agg_model_dict[i], torch.unsqueeze(model_dict[i], 0)], 0)
    return [torch.unsqueeze(model_state[i], 0)
            if not len(agg_model_state) else torch.cat([agg_model_state[i], torch.unsqueeze(model_state[i], 0)], 0) for i in range(num_vars)]


class FedAvg:
    def __init__(self):
        self.__model_state = []
        self.num_vars = None
        self.shape_vars = None

    def aggregate(self, model_state):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)
        update_model_state = [state.flatten() for state in model_state]
        self.__model_state = add_weights(self.num_vars, update_model_state, self.__model_state)

    def average(self):
        mean_updates = [torch.mean(self.__model_state[i], 0).reshape(self.shape_vars[i]) for i in range(self.num_vars)]
        self.__model_state = []
        return mean_updates


class WeiAvg:
    def __init__(self):
        pass


class WeiPFA:
    def __init__(self, proj_dims, lanczos_iter, PFA, PFAplus):
        self.proj_dims = proj_dims
        self.lanczos_iter = lanczos_iter
        self.PFA = PFA
        self.PFAplus = PFAplus

        self.__num_public = 0
        self.__public_model_state = []
        self.__public_eps = []

        self.__num_privacy = 0
        self.__privacy_model_state = []
        self.__privacy_eps = []

        self.num_vars = None
        self.shape_vars = None
        self.Vks = None
        self.means = None

    def aggregate(self, eps, model_state, is_public):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)
        update_model_state = [state.flatten() for state in model_state]

        if is_public:
            self.__num_public += 1
            self.__public_eps.append(eps)
            self.__public_model_state = add_weights(self.num_vars, update_model_state, self.__public_model_state)

        else:
            self.__num_privacy += 1
            self.__privacy_eps.append(eps)
            self.__privacy_model_state = add_weights(self.num_vars, update_model_state, self.__privacy_model_state)

    def __standardize(self, M):
        n, m = M.shape
        if m == 1:
            return M, torch.zeros(n, device='cuda')
        mean = torch.mul(M, torch.ones((m, 1), dtype=torch.float32, device='cuda')) / m

        return M - mean, mean.flatten()

    def __eigen_by_lanczos(self, mat):
        T, V = Lanczos(mat, self.lanczos_iter)

        T_evals, T_evecs = np.linalg.eig(T)

        idx = T_evals.argsort()[-1: -(self.proj_dims + 1): -1]

        Vk = np.dot(V.T, T_evecs[:, idx])
        Vk = torch.from_numpy(Vk).to(torch.float32).to('cuda').squeeze()

        return Vk

    def __projected_federated_average(self):
        if len(self.__privacy_model_state):
            privacy_weights = (torch.Tensor(self.__privacy_eps) / sum(self.__privacy_eps)).view(self.__num_privacy, 1).to('cuda')
            public_weights = (torch.Tensor(self.__public_eps) / sum(self.__public_eps)).view(self.__num_public, 1).to('cuda')

            mean_priv_model_state = [torch.sum(self.__privacy_model_state[i] * privacy_weights, 0) / torch.sum(privacy_weights) for i in range(self.num_vars)]
            mean_pub_model_state = [torch.sum(self.__public_model_state[i] * public_weights, 0) / torch.sum(public_weights) for i in range(self.num_vars)]
            mean_proj_priv_model_state = [0] * self.num_vars
            mean_model_state = [0] * self.num_vars

            for i in range(self.num_vars):
                public_model_state, mean = self.__standardize(self.__public_model_state[i].T)
                Vk = self.__eigen_by_lanczos(public_model_state.T)
                mean_proj_priv_model_state[i] = torch.mul(Vk, torch.dot(Vk.T.squeeze(), (mean_priv_model_state[i] - mean))) + mean
                mean_model_state[i] = ((mean_proj_priv_model_state[i] * sum(self.__privacy_eps) + mean_pub_model_state[i] * sum(self.__public_eps))
                                       / sum(self.__privacy_eps + self.__public_eps)).reshape(self.shape_vars[i])

            return mean_model_state

    def __projected_federated_average_plus(self, warmup):
        if len(self.__privacy_model_state):
            privacy_weights = (torch.Tensor(self.__privacy_eps) / sum(self.__privacy_eps)).view(self.__num_privacy, 1).to('cuda')
            public_weights = (torch.Tensor(self.__public_eps) / sum(self.__public_eps)).view(self.__num_public, 1).to('cuda')

            mean_priv_model_state = [torch.sum(self.__privacy_model_state[i] * privacy_weights, 0) / torch.sum(privacy_weights) for i in range(self.num_vars)]
            mean_pub_model_state = [torch.sum(self.__public_model_state[i] * public_weights, 0) / torch.sum(public_weights) for i in range(self.num_vars)]
            mean_proj_priv_model_state = [0] * self.num_vars
            mean_model_state = [0] * self.num_vars

            Vks = []
            means = []

            if warmup:
                for i in range(self.num_vars):
                    public_model_state, mean = self.__standardize(self.__public_model_state[i].T)
                    Vk = self.__eigen_by_lanczos(public_model_state.T)
                    mean_proj_priv_model_state[i] = torch.mul(Vk, torch.dot(Vk.T.squeeze(), (mean_priv_model_state[i] - mean))) + mean
                    mean_model_state[i] = ((mean_proj_priv_model_state[i] * sum(self.__privacy_eps) + mean_pub_model_state[i] * sum(self.__public_eps))
                                           / sum(self.__privacy_eps + self.__public_eps)).reshape(self.shape_vars[i])

                    Vks.append(Vk)
                    means.append(mean)
            else:
                for i in range(self.num_vars):
                    a = self.Vks[i].T.squeeze()
                    b = mean_priv_model_state[i]
                    c = torch.mul(self.Vks[i], mean_priv_model_state[i])
                    mean_proj_priv_model_state[i] = torch.mul(self.Vks[i], mean_priv_model_state[i]) + self.means[i]
                    mean_model_state[i] = ((mean_proj_priv_model_state[i] * sum(self.__privacy_eps) + mean_pub_model_state[i] * sum(self.__public_eps))
                                           / sum(self.__privacy_eps + self.__public_eps)).reshape(self.shape_vars[i])
                    public_model_state, mean = self.__standardize(self.__public_model_state[i].T)
                    Vk = self.__eigen_by_lanczos(public_model_state.T)

                    Vks.append(Vk)
                    means.append(mean)

            self.Vks = Vks
            self.means = means
            return mean_model_state

    def average(self):
        mean_updates = None

        if self.PFA:
            mean_updates = self.__projected_federated_average()
        elif self.PFAplus:
            mean_updates = self.__projected_federated_average_plus(warmup=(self.Vks is None))

        self.__num_public = 0
        self.__num_privacy = 0

        self.__public_model_state = []
        self.__privacy_model_state = []

        self.__public_eps = []
        self.__privacy_eps = []

        return mean_updates


class Server:
    def __init__(self, num_clients, sample_ratio, device):
        super(Server, self).__init__()
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.device = device

        self.model = CNN(input_dim=1, output_dim=10)
        self.state_dict_key = self.model.state_dict().keys()

        self.num_vars = None
        self.shape_vars = None
        self.__alg = None
        self.public = None
        self.__epsilons = None

    def set_public_clients(self, epsilons):
        self.__epsilons = epsilons

        sorted_eps = np.sort(epsilons)
        percent = 0.1
        threshold = sorted_eps[-int(percent * self.num_clients)]

        self.public = list(np.where(np.array(epsilons) >= threshold)[0])

    def init_global_model(self):
        return self.model

    def sample_clients(self, candidates):
        m = int(self.num_clients * self.sample_ratio)

        if candidates < m:
            return []
        else:
            participants = list(np.random.permutation(candidates))[0:m]

            if self.public is None:
                return participants

            check = 50
            while check and len(set(participants).intersection(set(self.public))) == 0:
                check -= 1
                participants = list(np.random.permutation(candidates))[0:m]

            return participants if check else []

    def init_alg(self, dp=True, Fedavg=False, weiavg=False, PFA=False, PFA_plus=False, proj_dims=None, lanczos_iter=None):

        if Fedavg or (not dp):
            self.__alg = FedAvg()
            print('\nUsing FedAvg algorithm')
        elif weiavg:
            self.__alg = WeiAvg()
            print('\nUsing PFA algorithm')
        elif PFA or PFA_plus:
            self.__alg = WeiPFA(proj_dims, lanczos_iter, PFA, PFA_plus)
            print('\nUsing PFA plus algorithm')
        else:
            raise ValueError('Select an algorithm (FedAvg/WeiAvg/PFA) to get the aggregated model!')

    def get_projection_info(self):
        return self.__alg.Vks, self.__alg.means

    def aggregate(self, participant, model_state, Fedavg=False, PFA=False, PFA_plus=False):
        if PFA or PFA_plus:
            self.__alg.aggregate(self.__epsilons[participant], model_state, is_public=True if (participant in self.public) else False)
        elif Fedavg:
            self.__alg.aggregate(model_state)
        else:
            raise ValueError('Select an algorithm (FedAvg/WeiAvg/PFA) to get the aggregated model!')

    def update(self):
        mean_state = self.__alg.average()

        mean_updates = dict(zip(self.state_dict_key, mean_state))

        self.model.load_state_dict(mean_updates)
        return self.model
