import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from parameters import *
from function import *

        
def snn_train_spike(device, train_dataloader, test_dataloader, model):
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    model.to(device)
    print('device:',device)
    opt = optim.Adam(model.parameters().astorch(), lr=0.000572)
    # scheduler = lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    losslist = []
    accuracy = []
    f1s = []
    precision = []
    recall = []
    cmlist= []
    for epoch in range(epochs):
        train_preds = []
        train_targets = []
        sum_loss = 0.0
        for batch, target in tqdm(train_dataloader):
            batch = batch.to(torch.float32).to(device)
            batch = batch.reshape(batch_size, time_step, num_channel)
            target_loss = target.to(device)
            model.reset_state()
            opt.zero_grad()
            out_model, _, rec = model(batch, record=True)
            out = torch.sum(out_model,dim=1)
            loss = criterion(out, target_loss)
            loss.backward()
            print_colorful_text(f"Loss:{loss}", 'yellow')
            opt.step()

            with torch.no_grad():
                pred = out.argmax(1).detach().to(device)
                train_preds += pred.detach().cpu().numpy().tolist()
                train_targets += target.detach().cpu().numpy().tolist()
                sum_loss += loss.item() / len(train_dataloader)

        sum_f1 = f1_score(train_targets, train_preds, average="macro")
        _, train_precision, train_recall, _ = precision_recall_fscore_support(
            train_targets, train_preds, labels=np.arange(2)
        )
        train_accuracy = accuracy_score(train_targets, train_preds)

        print(f"Train Epoch = {epoch+1}, Loss = {sum_loss}, F1 Score = {sum_f1}")
        print(f"Train Precision = {train_precision}, Recall = {train_recall}")
        print(f"Train Accuracy = {train_accuracy}")

        test_preds = []
        test_targets = []
        test_loss = 0.0
        for batch, target in tqdm(test_dataloader):
            with torch.no_grad():
                batch = batch.to(torch.float32).to(device)
                model.reset_state()
                batch = batch.reshape(len(test_dataloader.dataset), time_step, num_channel)
                out_model, _, rec = model(batch, record=True)
                out = torch.sum(out_model,dim=1)
                pred = out.argmax(1).detach().to(device)
                test_preds += pred.detach().cpu().numpy().tolist()
                test_targets += target.detach().cpu().numpy().tolist()
 
        f1 = f1_score(test_targets, test_preds, average="macro")
        _, test_precision, test_recall, _ = precision_recall_fscore_support(
            test_targets, test_preds, labels=np.arange(2)
        )
        test_accuracy = accuracy_score(test_targets, test_preds)
        cm = confusion_matrix(test_targets, test_preds)
        # losslist.append(test_loss)
        f1s.append(f1)
        precision.append(test_precision)
        recall.append(test_recall)
        accuracy.append(test_accuracy)
        cmlist.append(cm)
        print(f"F1 Score = {f1}")
        print(f"Val Precision = {test_precision}, Recall = {test_recall}")
        print_colorful_text(f"Accuracy:{test_accuracy}", 'yellow')
        print("Confusion Matrix:")
        print(cm)
        if test_accuracy > 0.927:
            model.save(f'models/modelmix_spike_{epoch}.pth')
            print('模型已保存')
            # np.save('train_data_record/loss.npy', losslist)
            # np.save('train_data_record/f1s.npy', f1s)
            # np.save('train_data_record/precision.npy', precision)
            # np.save('train_data_record/recall.npy', recall)
            # np.save('train_data_record/accuracy.npy', accuracy)
            # np.save('train_data_record/cm.npy', cmlist)
            print('训练已完成，训练参数已保存') 
            break
        
