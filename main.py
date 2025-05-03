
from src.models import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datasets import Dataset
# from causalml.metrics import auuc_score
from src.loss import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def get_batches(dataset, batch_size):
    """
    按批次取数据
    """
    num_rows = len(dataset)
    indices = np.arange(num_rows)
    np.random.shuffle(indices)  # Optional: Shuffle data

    for start_idx in range(0, num_rows, batch_size):
        end_idx = min(start_idx + batch_size, num_rows)
        batch_indices = indices[start_idx:end_idx]
        yield dataset.select(batch_indices)


def evaluate_model(model, dataset, loss_function, num_treatments, model_name, x_col, t_col, y_col):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_dataset in get_batches(dataset, batch_size):
            x_tensor = torch.tensor(batch_dataset[x_col], dtype=torch.float32)
            t_tensor = torch.tensor(batch_dataset[t_col], dtype=torch.int64)
            y_tensor = torch.tensor(batch_dataset[y_col], dtype=torch.int64)  # (batch_size, )

            y_outputs, x_emb, t_outputs = model(x_tensor, t_tensor)
            y_loss = loss_function(y_outputs.float(), y_tensor.unsqueeze(1).float())

            if num_treatments == 2:
                t_loss = ((torch.sigmoid(t_outputs) - t_tensor.unsqueeze(1)) ** 2).sum()

            if model_name == 'TarNet':
                loss = y_loss
            if model_name == 'CFRNet':
                loss = y_loss + ipm_loss
            if model_name == 'DargonNet':
                loss = y_loss + t_loss

            total_loss += loss.item()
            # 计算准确率
            predicted = torch.sigmoid(y_outputs).round()
            correct += (predicted.squeeze() == y_tensor).sum().item()
            total += y_tensor.size(0)
            # 收集标签和预测值用于AUC计算
            all_labels.extend(y_tensor.numpy())
            all_probs.extend(torch.sigmoid(y_outputs).numpy())

        # 损失值/准确率
        avg_loss = total_loss / len(dataset)
        accuracy = correct / total
        # AUC
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        auc_score = roc_auc_score(all_labels, all_probs)

        return avg_loss, accuracy, auc_score


if __name__ == '__main__':

    # data = pd.read_csv('./data/data.csv')
    data = pd.read_pickle('./data/data_demo.pkl')
    data['t'] = data['treatment'].apply(lambda x : 1 if x > 0 else 0)
    data = pd.concat([data.query('t == 0'), data.query('t == 1').sample(100000)])
    print(data.groupby(['t']).label.mean())
    print(data.groupby(['t']).userid.count())
    data.fillna(0, inplace=True)
    X_cols = [x for x in data.columns if x not in ['userid', 'treatment', 't', 'label']]

    data['X'] = data[X_cols].apply(lambda row: row.tolist(), axis=1)
    data.drop(columns=X_cols, inplace=True)
    t_mean = data['t'].mean()
    data['w'] = data['t'].apply(lambda x : 0.5*x/t_mean if x == 1 else 0.5*(1-x)/(1-t_mean) )


    train_data, test_data = train_test_split(data, test_size=0.2)
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(test_data)

    x_col = 'X'
    t_col = 't'
    y_col = 'label'

    num_features = len(X_cols)
    num_treatments = 2
    treatments = [0, 1]
    # treatments = [0] + list(range(3, num_treatments+2))

    model = DeepCausalModel(n_treatments=num_treatments, treatments=treatments, output_dim=1, units=num_features, phi_layers=2, y_layers=3, activation="relu")

    print(f'model para:{sum(p.numel() for p in model.parameters())}')

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1
    batch_size = 200
    model_name = 'TarNet'

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_dataset in get_batches(train_dataset, 2000):
            x_tensor = torch.tensor(batch_dataset[x_col], dtype=torch.float32)
            t_tensor = torch.tensor(batch_dataset[t_col], dtype=torch.int64)
            y_tensor = torch.tensor(batch_dataset[y_col], dtype=torch.int64) # (batch_size, )
            w_tensor = torch.tensor(batch_dataset['w'], dtype=torch.float32)

            optimizer.zero_grad()
            y_outputs, x_emb, t_outputs = model(x_tensor, t_tensor)

            y_loss = loss_function(y_outputs.float(), y_tensor.unsqueeze(1).float())
            if num_treatments == 2:
                t_loss = ((torch.sigmoid(t_outputs) - t_tensor.unsqueeze(1)) ** 2).sum()
                ipm_loss = compute_ipm_loss(x_emb, t_tensor, ipm_func='mmd')  # wasserstein, mmd

            if model_name == 'TarNet':
                loss = y_loss
            if model_name == 'CFRNet':
                loss = y_loss + ipm_loss
            if model_name == 'DargonNet':
                loss = y_loss + t_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # print('y_outputs :', y_outputs)
        # print(torch.sigmoid(y_outputs).round())

        epoch_loss /= len(train_dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        train_loss, train_accuracy, train_auc_score = evaluate_model(model, train_dataset, loss_function, num_treatments, model_name, x_col, t_col, y_col)
        val_loss, val_accuracy, val_auc_score = evaluate_model(model, val_dataset, loss_function, num_treatments, model_name, x_col, t_col, y_col)

        print(f'Epoch {epoch + 1}/{num_epochs}------- ')
        print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Auc score: {train_auc_score: .4f}')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Auc score: {val_auc_score: .4f}')


    # auuc
    # X_tensor = torch.tensor(test_dataset[x_col], dtype=torch.float32)
    # T_tensor = torch.tensor(test_dataset[t_col], dtype=torch.int64)
    # Y_tensor = torch.tensor(test_dataset[y_col], dtype=torch.int64)
    #
    # model.eval()
    # predictions = model(X_tensor)
    # predictions = predictions.squeeze(-1) # (sample_size, num_treatments)
    # predicted_rate = torch.sigmoid(predictions) # (sample_size, num_treatments)
    # base_rate = predicted_rate[:, 0:1]
    # treat_rate = predicted_rate[:, 1:]
    # result = treat_rate - base_rate  # (sample_size, num_treatments - 1)
    # print(result)
    #
    # auuc_res = {}
    # for i, treatment in enumerate([x for x in treatments if x not in [0]]):
    #     mask = (T_tensor == 0) | (T_tensor == treatment)
    #     indices = torch.nonzero(mask).squeeze() # (sample_size, )
    #     predict_result = result[indices, i-1] # (sample_size, )
    #     t_result = T_tensor[indices]
    #     t_result[t_result > 0] = 1  # (sample_size, )
    #     y_result = Y_tensor[indices] # (sample_size, )
    #     treatment_result = torch.stack([t_result, y_result, predict_result], dim=1) # (sample_size, 3)
    #     treatment_result = pd.DataFrame(treatment_result.detach().numpy(), columns=['treatment', 'label', 'predict'])
    #     auuc = auuc_score(treatment_result, 'label', 'treatment', normalize=True)
    #     auuc_lift = auuc.values[0] - auuc.values[1]
    #     auuc_res[treatment] = round(auuc_lift, 3)
    # print(auuc_res)
    #