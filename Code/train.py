import torch
from metrics import AverageMeter
from loss import batch_all_triplet_loss, batch_hard_triplet_loss
from tqdm import tqdm
from test import result
import matplotlib.pyplot as plt
import numpy as np

# def save(save_path, model, optimiz er):
#     if save_path==None:
#         return
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict()
#     }, '../Model/' + save_path)
#     print('Model saved to ==> {../Model/save_path}')

def save(save_path, model, optimizer=None, scheduler=None):
    if save_path==None:
        return
    checkpoint = {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
    }
    save_path = '../Model/' + save_path
    torch.save(checkpoint, save_path)
    print(f'Model saved to ==> {save_path}')

def load(name, model, optimizer):
    checkpoint = torch.load('../Model/' + name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

def train(model,train_loader,valid_loader,valid_loader1,valid_loader2,optimizer,scheduler,num_epochs,eval_every,margin,device,name):
    epoch_loss_list = {'train':[], 'valid':[]}
    IOU_list = []
    global_step = 0
    train_loss = AverageMeter()
    valid_loss = AverageMeter()
    best_IOU = 1
    #train_average_margin = AverageMeter()
    total_step = len(train_loader)*num_epochs
    count = 0
    print(f'total steps: {total_step}')
    for epoch in range(num_epochs):
        print(f'epoch {epoch+1}')
        #losses = []
        for _, data in enumerate(tqdm(train_loader)):
            if count > 0:
                break
            count += 1
            model.train()
            inputs = data['image'].to(device) # inputs
            target = data['target'].to(device) # targets
            embeddings = model(inputs)
            loss, _ = batch_all_triplet_loss(target, embeddings, margin=margin, epoch=epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())
            global_step += 1
            current_lr = optimizer.param_groups[0]['lr']

            ### print
            if global_step % eval_every == 0:
                model.eval()
                for _, data in enumerate(tqdm(valid_loader)):
                    inputs = data['image'].to(device) # inputs
                    target = data['target'].to(device) # targets
                    embeddings = model(inputs)
                    loss, _ = batch_all_triplet_loss(target, embeddings, margin=margin, epoch=epoch)
                    valid_loss.update(loss.item())

                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, lr: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, total_step, train_loss.avg, valid_loss.avg ,current_lr))

        # valid
        dist1 = result(model,valid_loader1,device)
        dist2 = result(model,valid_loader2,device)
        same_hist = plt.hist(dist1, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='same')
        diff_hist = plt.hist(dist2, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='diff')
        plt.legend(loc='upper right')
        plt.savefig('../Result/distribution_epoch'+str(epoch+1)+'.png')
        difference = same_hist[0] - diff_hist[0]
        difference[:same_hist[0].argmax()] = np.Inf
        difference[diff_hist[0].argmax():] = np.Inf
        dist_threshold = (same_hist[1][np.where(difference <= 0)[0].min()] + same_hist[1][np.where(difference <= 0)[0].min() - 1])/2
        overlap = np.sum(dist1>=dist_threshold) + np.sum(dist2<=dist_threshold)
        IOU = overlap / (dist1.shape[0] * 2 - overlap)
        print('dist_threshold:',dist_threshold,'overlap:',overlap,'IOU:',IOU)
        plt.clf()

        epoch_loss_list['train'].append(train_loss.avg)
        epoch_loss_list['valid'].append(valid_loss.avg)
        IOU_list.append(IOU)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,9))
        ax1.plot(range(len(epoch_loss_list['train'])), epoch_loss_list['train'], label=('train_loss'))
        ax1.plot(range(len(epoch_loss_list['valid'])), epoch_loss_list['valid'], label=('valid_loss'))
        ax2.plot(range(len(IOU_list)), IOU_list, label=('IOU'))
        ax1.legend(prop={'size': 15})
        ax2.legend(prop={'size': 15})
        plt.savefig('../Result/loss.png')
        plt.clf()

        if IOU < best_IOU:
            best_IOU = IOU
            save(name, model, optimizer, scheduler)

        train_loss.reset()
        valid_loss.reset()

        # count the step of the scheduler in each epoch
        scheduler.step()
    print('Finished Training')
