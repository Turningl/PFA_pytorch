# -*- coding: utf-8 -*-
# @Author : Zhang
# @Email : zl16035056@163.com
# @File : main_utils.py


import os
import csv
import numpy as np

np.random.seed(10)

def save_progress(args, Accuracy_accountant, Budgets_accountant=None, nbytes1=None, nbytes2=None):

    save_dir = os.path.join(os.getcwd(), args.save_dir, 'result', args.dataset,
                            ('noniid' if args.noniid else 'iid'),
                            (args.eps if args.dp else 'no-dp'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = '{}{}{}{}{}'.format(args.num_clients,
                              ('-Fedavg' if args.Fedavg else ''),
                              ('-PFA' if args.PFA else ''),
                              ('-PFA_plus' if args.PFA_plus else ''),
                              args.delta)

    with open(os.path.join(save_dir, file_name + '.csv'), 'w') as file:
        writer = csv.writer(file, delimiter=',')
        if args.dp:
            writer.writerow(Budgets_accountant)
        if args.PFA_plus:
            writer.writerow(nbytes1)
            writer.writerow(nbytes2)

        writer.writerow(Accuracy_accountant)


def print_accuracy_and_loss(test_accuracy, test_loss):
    print('-------------------------------------------------------------------------------------')
    print('current global model has test acc: %.4f  test loss: %.4f' % (test_accuracy, test_loss))
    print('-------------------------------------------------------------------------------------')