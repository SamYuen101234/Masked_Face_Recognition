import torch
import torch.nn as nn
from torch import optim
from sampler import PKSampler, PKSampler2

def get_Sampler(sampler,dataset,p=15,k=20):
    if sampler == 'all':
        return PKSampler2(dataset, p=p, k=k)
    else:
        return PKSampler(dataset, p=p, k=k)

def get_Optimizer(model, optimizer_type=None, lr=1e-3, weight_decay=1e-3):
    if(optimizer_type=='sgd'):
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    elif(optimizer_type=='rmsprop'):
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif(optimizer_type=='adadelta'):
        return optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_Scheduler(optimizer, lr, scheduler_name=None):
    if(scheduler_name=='cyclic'):
        return optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-4, max_lr=lr, step_size_up=500)
    elif(scheduler_name=='cosine'):
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    elif(scheduler_name=='multistep'):
        # return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,13,30], gamma=0.3)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,20,40], gamma=0.1)
    else:
        return optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
