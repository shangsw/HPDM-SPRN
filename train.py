# -*- coding: utf-8 -*-
import torch
import os
import numpy as np
from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window


def train(net, optimizer, criterion, data_loader, val_loader, epoch, saving_path, scheduler=None, device=torch.device('cpu')):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        data_loader: a PyTorch dataset loader
        val_loader: validation dataset
        epoch: int specifying the number of training epochs
        saving_path: model saved path
        scheduler (optional): PyTorch scheduler  
    """

    best_acc = 0.0
    losses = []

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        # Run the training loop for one epoch
        for batch_idx, (data, target) in enumerate(data_loader):
            # Load the data 
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
        # plot progress
        losses.append(loss.item())
        if e % 10 == 0:
            mean_losses = np.mean(losses)
            string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            string = string.format(
                e, epoch, batch_idx *
                len(data), len(data) * len(data_loader),
                100. * batch_idx / len(data_loader), mean_losses)
            tqdm.write(string)
            losses = []
             
        del(data, target, loss)
        #Run the validating and update the scheduler
        val_acc = val(net, val_loader, device=device)
        scheduler.step()
        # Save the best weights
        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint(net, is_best, saving_path, epoch=e, acc=best_acc)


def save_checkpoint(net, is_best, saving_path, **kwargs):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    if is_best:
        tqdm.write("Epoch{epoch}: Best validation accuracy:{acc:.3f}".format(**kwargs))
        torch.save(net.state_dict(), os.path.join(saving_path, 'model_best.pth'))
    else:
        if kwargs['epoch'] % 20 == 0:
            torch.save(net.state_dict(), os.path.join(saving_path,'model.pth'))


def test(net, model_dir, img, patch_size, n_classes, device=torch.device('cpu')):
    """
    Test a model on a specific image
    """
    net.load_state_dict(torch.load(model_dir + '/model_best.pth'))
    net.eval()
    
    patch_size = patch_size
    batch_size = 64
    window_size = (patch_size, patch_size)

    img_w, img_h  = img.shape[:2]
    pad_size = patch_size//2
    img = np.pad(img, ((pad_size,pad_size),(pad_size,pad_size),(0,0)), mode='reflect')

    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, window_size=window_size) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(img, window_size=window_size)),
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu').numpy()
            
            for (x, y, w, h), out in zip(indices, output):
                probs[x + w // 2, y + h // 2] += out
    return probs[pad_size:img_w+pad_size,pad_size:img_h+pad_size,:]


def val(net, data_loader, device=torch.device('cpu')):
    net.eval()
    accuracy, total = 0., 0.
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output = net(data)
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                accuracy += out.item() == pred.item()
                total += 1
    return accuracy / total
