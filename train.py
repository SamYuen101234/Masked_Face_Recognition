import torch
from metrics import AverageMeter
from triplet import batch_all_triplet_loss, batch_hard_triplet_loss
from tqdm import tqdm 
from test import result
import matplotlib.pyplot as plt
import numpy as np
from arcface import loss_fn

def save(save_path, model, optimizer, scheduler):
    if save_path==None:
        return
    checkpoint = { 
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
    }
    save_path = './models/' + save_path 
    torch.save(checkpoint, save_path)
    print(f'Model saved to ==> {save_path}')

def load(save_path):
    save_path = './models/' + save_path 
    checkpoint = torch.load(save_path)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
    print(f'Model loaded from <== {save_path}')
    return model, optimizer, scheduler

# train function for online triplet loss
def train(model,train_loader,valid_loader1,valid_loader2,optimizer,scheduler,num_epochs,eval_every,margin,device,name):
    IOU_list = []
    best_IOU = 1
    global_step = 0
    train_loss = AverageMeter()
    local_train_loss = AverageMeter()
    best_train_loss = float("Inf")
    best_val_loss = float("Inf")
    loss_list = []
    total_step = len(train_loader)*num_epochs
    print(f'total steps: {total_step}')
    for epoch in range(num_epochs):
        print(f'epoch {epoch+1}')
        #losses = []
        for _, data in enumerate(tqdm(train_loader)):
            model.train()
            inputs = data['image'].to(device) # inputs
            target = data['target'].to(device) # targets
            embeddings = model(inputs)
            loss= batch_hard_triplet_loss(target, embeddings, margin=margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())
            local_train_loss.update(loss.item())
            global_step += 1
            current_lr = optimizer.param_groups[0]['lr']
            ### print
            if global_step % eval_every == 0:
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f} ({:.4f}), lr: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, total_step, local_train_loss.avg, train_loss.avg, current_lr))
                if local_train_loss.avg < best_train_loss:
                    best_train_loss = local_train_loss.avg
                    print('Best trian loss:',local_train_loss.avg)
                loss_list.append(local_train_loss.avg)
                local_train_loss.reset()
        # valid
        with torch.no_grad():
            model.eval()
            val_loss = AverageMeter()
            for _, valid_data in enumerate(valid_loader1):
                inputs = valid_data['image'].to(device) # inputs
                target = valid_data['target'].to(device) # targets
                embeddings = model(inputs)
                valid_loss= batch_hard_triplet_loss(target, embeddings, margin=margin)
                val_loss.update(valid_loss.item())
        dist1 = result(model,valid_loader1,device, loss_fn='triplet')
        dist2 = result(model,valid_loader2,device, loss_fn='triplet')
        try:
            same_hist = plt.hist(dist1, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='same')
            diff_hist = plt.hist(dist2, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='diff')
            plt.legend(loc='upper right')
            plt.savefig('result/distribution_epoch'+str(epoch+1)+'.png')
            difference = same_hist[0] - diff_hist[0]
            difference[:same_hist[0].argmax()] = np.Inf
            difference[diff_hist[0].argmax():] = np.Inf
            dist_threshold = (same_hist[1][np.where(difference <= 0)[0].min()] + same_hist[1][np.where(difference <= 0)[0].min() - 1])/2
            overlap = np.sum(dist1>=dist_threshold) + np.sum(dist2<=dist_threshold)
            IOU = overlap / (dist1.shape[0] * 2 - overlap)
        except:
            print("Model results in collapse") # if the collapse to 0 then, the result cannot be printed

        print('dist_threshold:',dist_threshold,'overlap:',overlap,'IOU:',IOU)
        plt.clf()
        IOU_list.append(IOU)
        if IOU < best_IOU:
            best_IOU = IOU
            save(name,model,optimizer,scheduler)

        print('Valid loss:',val_loss.avg)
        if val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            print(best_val_loss)
        val_loss.reset()
        # count the step of the scheduler in each epoch
        scheduler.step()
    # loss graph
    steps = range(len(loss_list))
    plt.plot(steps, loss_list)
    plt.title('Train loss')
    plt.ylabel('Loss')
    plt.xlabel('Steps')
    plt.savefig('train_loss.png')
    plt.clf()
    print('Finished Training')

# train function for arcface
def train2(model,train_loader,valid_loader1,valid_loader2,metric_crit,optimizer,scheduler,num_epochs,eval_every,num_class,device,name):
    IOU_list = []
    best_IOU = 1
    global_step = 0
    # loss from 1st epoch to nth epoch
    train_loss = AverageMeter()
    # loss to n step/eval_every
    local_train_loss = AverageMeter()
    # a list host the loss every eval_every
    loss_list = []
    best_train_loss = float("Inf")
    total_step = len(train_loader)*num_epochs
    print(f'total steps: {total_step}')
    for epoch in range(num_epochs):
        print(f'epoch {epoch+1}')
        for _, data in enumerate(tqdm(train_loader)):
            model.train()
            # original image
            inputs = data['image'].to(device)
            #targets = data['target'].to(device)
            outputs = model(inputs)
            loss = loss_fn(metric_crit, data, outputs, num_class, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.cpu().item(), inputs.size(0))
            local_train_loss.update(loss.cpu().item(), inputs.size(0))
            global_step += 1
            current_lr = optimizer.param_groups[0]['lr']
            if global_step % eval_every == 0:
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f} ({:.4f}), lr: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, total_step, local_train_loss.avg, train_loss.avg, current_lr))
                if local_train_loss.avg < best_train_loss:
                    best_train_loss = local_train_loss.avg
                    print('Best trian loss:',local_train_loss.avg)
                loss_list.append(local_train_loss.avg)
                local_train_loss.reset()
        # val
        dist1 = result(model,valid_loader1,device, loss_fn='arcface')
        dist2 = result(model,valid_loader2,device, loss_fn='arcface')
        try:
            same_hist = plt.hist(dist1, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='same')
            diff_hist = plt.hist(dist2, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='diff')
            plt.legend(loc='upper right')
            plt.savefig('result/distribution_epoch'+str(epoch+1)+'.png')
            difference = same_hist[0] - diff_hist[0]
            difference[:same_hist[0].argmax()] = np.Inf
            difference[diff_hist[0].argmax():] = np.Inf
            dist_threshold = (same_hist[1][np.where(difference <= 0)[0].min()] + same_hist[1][np.where(difference <= 0)[0].min() - 1])/2
            overlap = np.sum(dist1>=dist_threshold) + np.sum(dist2<=dist_threshold)
            IOU = overlap / (dist1.shape[0] * 2 - overlap)
        except:
            print("Model results in collapse") # if the collapse to 0 then, the result cannot be printed
        print('dist_threshold:',dist_threshold,'overlap:',overlap,'IOU:',IOU)
        plt.clf()
        IOU_list.append(IOU)
        if IOU < best_IOU:
            best_IOU = IOU
            save(name,model,optimizer,scheduler)
        scheduler.step()
    # loss graph
    steps = range(len(loss_list))
    plt.plot(steps, loss_list)
    plt.title('Train loss')
    plt.ylabel('Loss')
    plt.xlabel('Steps')
    plt.savefig('train_loss.png')
    plt.clf()
    print('Finished Training')