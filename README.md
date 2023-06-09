# PFA_pytorch

This is the source code of paper __"Projected Federated Averaging with Heterogeneous Differential Privacy"__ (accepted by VLDB 2022).

But the original is not this, if you need to see the original, please see https://github.com/Emory-AIMS/PFA

## Getting Started

This repository is an implementation of Pytorch version of __Federated Averaging (FedAvg), Projected Federated Averaging (PFA) and Projetced Federated Averaging Plus (PFA plus)__ algorithms.

## Install

```
Pytroch = 1.12.1
Opacus = 1.4.0
```

## Usage

* NP-FedAvg algorithm:
```
python main.py --Fedavg=True
```

* DP-FedAvg algorithm:
```
python main.py --Fedavg=True --dp=True
```

* PFA algorithm:
```
python main.py --PFA=True --dp=True
```

* PFA plus algorithm:
```
python main.py --PFA_plus=True --dp=True
```

## Acknowledgements
* [Pytorch](https://github.com/pytorch/pytorch.git)
* [Pytorch/Opacus](https://github.com/pytorch/opacus.git)


## Contributing
The first author for this job is Junxu Liu. If you have any questions about this article, please contact junxu_liu@ruc.edu.cn

If you have any questions about this code, please email me zl16035056@163.com
