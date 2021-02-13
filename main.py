from dataset import customized_dataset
from model import FaceNet
from sampler import samples
from train import train, load
from test import evalulate, test
from util import get_Optimizer, get_Scheduler, get_Sampler

import torch
import pandas as pd
from torch.utils.data import DataLoader
import multiprocessing
if __name__ == "__main__":
    #######################################################################################
    #########################################config#######################################
    BATCH_SIZE=256
    #NUM_WORKERS = multiprocessing.cpu_count()
    NUM_WORKERS = multiprocessing.cpu_count()
    embedding_size = 512
    # all or None(may not iterate all data in each batch)
    sampler = None
    weight_decay = 5e-4
    lr = 5e-2
    dropout = 0.3
    # resnet, effnet or None(IncepetionResNet)
    model_name = None
    pretrain = True
    # 'arcface' or 'triplet'
    loss_fn = 'triplet'
    # global gem or None(avgerage pooling)
    pool= None
    # Cyclic or Step
    scheduler_name = 'multistep'
    # sgd or None(adam) or rmsprop
    optimizer_type = 'adadelta'
    num_epochs = 100
    eval_every = 50
    # margin for triplet loss
    margin=3
    # name to open or save the model
    name = 'triplet.pth'
    #######################################################################################
    #######################################################################################
    
    # device: cpu or cuda
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)
    
    # read scv
    df_train = pd.read_csv('train.csv')
    df_eval1 = pd.read_csv('eval_same.csv')
    df_eval2 = pd.read_csv('eval_diff.csv')
    df_test = pd.read_csv('test.csv')

    # label_to_samples
    print('Initializing sampler...')
    label_to_samples = samples(df_train)

    # dataset, sampler and dataloader
    train_dataset = customized_dataset(df_train, mode='train', label_to_samples=label_to_samples)
    eval_dataset1 = customized_dataset(df_eval1, mode='eval')
    eval_dataset2 = customized_dataset(df_eval2, mode='eval')
    test_dataset = customized_dataset(df_test, mode='test')

    train_sampler = get_Sampler(sampler, train_dataset, p=20, k=30)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=False, sampler=train_sampler)
    eval_loader1 = DataLoader(eval_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
    eval_loader2 = DataLoader(eval_dataset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
    
    # model, optimizer, scheduler
    facenet = FaceNet(model_name=model_name, pool=pool, embedding_size=embedding_size, dropout=dropout, device=device, pretrain=pretrain)
    #facenet = torch.nn.DataParallel(facenet, device_ids=[0,1,2,3]) # multi-GPU training, here shows four cuda
    optimizer = get_Optimizer(facenet, optimizer_type, lr, weight_decay) # optimizer
    scheduler = get_Scheduler(optimizer, lr, scheduler_name) # scheduler
    # load previous trained model
    if False:
        facenet, optimizer, scheduler = load('./models/'+name)

    # train
    train(facenet.to(device),train_loader,eval_loader1,eval_loader2,optimizer,scheduler,num_epochs,eval_every,margin,device,name)
    dist_threshold = evalulate(facenet, eval_loader1, eval_loader2, device, loss_fn)
    print('Distance threshold:',dist_threshold)
    test(facenet,test_loader,dist_threshold,device, loss_fn)
