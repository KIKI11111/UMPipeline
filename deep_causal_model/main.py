from src.models import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
# from causalml.metrics import auuc_score
from sklearn.metrics import roc_auc_score


def calc_cfr_loss(x_emb, t):
    t[t > 0] = 1
    treated = t.nonzero(as_tuple=True)[0]
    control = t.nonzero(as_tuple=True)[1]
    x_treated, x_control = x_emb[treated], x_emb[control]
    loss = torch.norm(x_treated.mean(axis=0) - x_control.mean(axis=0))
    return loss

def evaluate(model, test_dataloader, model_name):
    model.eval()  # 设置模型为评估模式
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    labels, scores = [], []

    with torch.no_grad():
        for x, t, y in test_dataloader:
            y_outputs, x_emb, t_outputs = model(x, t)
            y_loss = criterion(y_outputs.float(), y.float())
            t_loss = ((torch.sigmoid(t_outputs) - t.unsqueeze(1))**2).sum()
            if model_name == 'tarnet':
                loss = y_loss
            if model_name == 'dargonnet':
                loss = y_loss + t_loss

            total_loss += loss.item()
            # 收集标签和预测值用于AUC计算
            labels.extend(y.numpy()) # 真实标签
            scores.extend(torch.sigmoid(y_outputs).numpy()) # 预估概率


        avg_loss = total_loss / len(test_dataloader) # 损失值
        labels = np.array(labels)
        scores = np.array(scores)
        auc_score = roc_auc_score(labels, scores)  # auc
        predict = (scores > 0.5)*1
        correct = (predict == labels).sum()
        accuracy = correct/len(labels)

        return avg_loss, accuracy, auc_score

if __name__ == '__main__':

    num_samples = 1000
    num_features = 20
    num_treatments = 2
    treatments = [0, 1]
    x = torch.randn(num_samples, num_features)
    t = torch.randint(0, 2, size=(num_samples,1))
    y = torch.randint(0, 2, size=(num_samples,1))
    dataset = TensorDataset(x, t, y)
    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    model = DeepCausalModel(n_treatments=num_treatments, treatments=treatments, x_dims=[20, 80, 160, 20], y_dims=[20, 40, 20]
                 ,x_emb_size=20, y_emb_size=20, output_dim=1)
    print(f'model para:{sum(p.numel() for p in model.parameters())}')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1
    batch_size = 200
    model_name = 'tarnet'

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for x, t, y in train_dataloader:
            optimizer.zero_grad()
            y_outputs, x_emb, t_outputs = model(x, t)

            # print('x shape', x.shape)
            # print('t shape', t.shape)
            # print('y shape', y.shape)
            # print('y_outputs size:', y_outputs.shape)
            # print('x_emb:', x_emb)
            # print('t:', t)
            # print('t_outputs shape', t_outputs.shape)
            # print('y_outputs', y_outputs)
            # print('torch.sigmoid(y_outputs)',torch.sigmoid(y_outputs))
            # print('y', y)

            y_loss = criterion(y_outputs.float(), y.float())
            t_loss = ((torch.sigmoid(t_outputs) - t.unsqueeze(1))**2).sum()
            cfr_loss = calc_cfr_loss(x_emb, t)
            if model_name == 'tarnet':
                loss = y_loss
            if model_name == 'dargonnet':
                loss = y_loss + t_loss
            if model_name == 'cfrnet':
                loss = y_loss + cfr_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        print(f'epoch {epoch+1} / {num_epochs}, loss : {epoch_loss}')


        train_loss, train_accuracy, train_auc_score = evaluate(model, train_dataloader, model_name)
        test_loss, test_accuracy, test_auc_score = evaluate(model, test_dataloader, model_name)
        print(f'Epoch {epoch + 1}/{num_epochs}------- ')
        print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Auc score: {train_auc_score: .4f}')
        print(f'Validation Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Auc score: {test_auc_score: .4f}')


    #
    # # AUUC
    # x = torch.randn(100, num_features)
    # t = torch.randint(0, 2, size=(100, 1))
    # y = torch.randint(0, 2, size=(100, 1))
    #
    # model.eval()
    # outputs = model(x)
    # outputs = outputs.squeeze(-1) # (sample_size, num_treatments)
    # scores = torch.sigmoid(outputs) # (sample_size, num_treatments)
    # base_rate = scores[:, 0:1]
    # treat_rate = scores[:, 1:]
    # result = treat_rate - base_rate  # (sample_size, num_treatments - 1)
    #
    # auuc_res = {}
    # for i, treatment in enumerate([x for x in treatments if x not in [0]]):
    #     mask = (t == 0) | (t == treatment)
    #     index = torch.nonzero(mask)[:, 0] # (sample_size, )
    #     predict_score = result[index, i-1].unsqueeze(1) # (sample_size, )
    #     t_result = t[index]
    #     t_result[t_result > 0] = 1  # (sample_size, )
    #     y_result = y[index] # (sample_size, )
    #     treatment_result = torch.cat([t_result, y_result, predict_score], dim=1) # (sample_size, 3)
    #     treatment_result = pd.DataFrame(treatment_result.detach().numpy(), columns=['treatment', 'label', 'predict'])
    #     # auuc = auuc_score(treatment_result, 'label', 'treatment', normalize=True)
    #     # auuc_lift = auuc.values[0] - auuc.values[1]
    #     # auuc_res[treatment] = round(auuc_lift, 3)
    #     print(treatment_result.shape)
    # # print(auuc_res)