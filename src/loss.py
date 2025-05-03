
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def data_split_with_treatment_type(data, treatment):

    treatment[treatment > 0] = 1
    control_indices = torch.nonzero(treatment == 0).squeeze()
    treated_indices = torch.nonzero(treatment == 1).squeeze()
    control_data = data[control_indices]
    treated_data = data[treated_indices]

    return treated_data, control_data



def compute_ipm_loss(x_emb, t, ipm_func):

    rep_t, rep_c = data_split_with_treatment_type(x_emb, t)

    # print('rep_t shape:', rep_t.shape)
    # print('rep_c shape:', rep_c.shape)

    if ipm_func == 'mmd':
        ipm_loss = 2 * torch.norm(rep_t.mean(axis=0) - rep_c.mean(axis=0))
    if ipm_func == 'wasserstein':
        n_t = rep_t.size(dim=0)
        n_c = rep_c.size(dim=0)

        print('n_t:', n_t)
        print('n_c:', n_c)

        if n_t == 0 or n_c == 0:
            ipm_loss = 0
        else:
            # compute distance matirx M
            M = torch.tensor(
                [[torch.linalg.vector_norm(rep_t[i] - rep_c[j]) ** 2 for j in range(n_c)] for i in range(n_t)])

            # compute transport matrix T
            print('M:', M)
            p = 0.5
            lam = 1
            iterations = 10
            a = p * torch.ones((n_t, 1)) / n_t
            b = (1 - p) * torch.ones((n_c, 1)) / n_c

            # K = torch.exp(-lam * M)
            log_K = -lam * M
            log_K_max = torch.max(log_K, dim=1, keepdim=True)[0]  # 取每一行的最大值
            log_K_stable = log_K - log_K_max  # 稳定化处理
            K = torch.exp(log_K_stable)

            K_tilde = K / a

            print('a:', a)
            print('b:', b)
            print('K:', K)
            print('K_tilde:', K_tilde)

            u = a
            for i in range(0, iterations):
                u = 1.0 / torch.matmul(K_tilde, b / torch.matmul(torch.transpose(K, 0, 1), u))

            v = b / torch.matmul(torch.transpose(K, 0, 1), u)

            T = u * (torch.transpose(v, 0, 1) * K)

            # compute wasserstein distance
            E = T * M
            ipm_loss = 2 * torch.sum(E)

    return ipm_loss



def compute_loss(x_emb, outputs, t,  y, use_ipm=False, ipm_func=None):

    loss_function = nn.BCEWithLogitsLoss()
    pred_loss = loss_function(outputs, y)

    if use_ipm:  # cfrnet
        if ipm_func == 'mmd':
            ipm_loss = compute_ipm_loss(x_emb, t, 'mmd')
        elif ipm_func == 'wasserstein':
            ipm_loss = compute_ipm_loss(x_emb, t, 'wasserstein')
        else:
            raise Exception(f"Unknown ipm function : {ipm_func}")
    else:  # tarnet
        ipm_loss = 0

    total_loss = pred_loss + ipm_loss
    return total_loss