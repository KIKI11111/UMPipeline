import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SplitStreams(nn.Module):
    def __init__(self):
        super(SplitStreams, self).__init__()

    def forward(self, x, t, t_cur):
        t = t.view(-1)
        index_position = (t == t_cur).nonzero().view(-1)
        return x[index_position], index_position


class GatherStreams(nn.Module):
    def __init__(self):
        super(GatherStreams, self).__init__()

    def forward(self, y_outputs, index_streams, n_treatments):
        # 将y_output按照index重新排列，使其和输入x的顺序保持一致
        combined = []
        for i in range(n_treatments):
            combined.extend([(y_outputs[i][j], index_streams[i][j]) for j in range(len(index_streams[i]))])
        combined = sorted(combined, key=lambda x: x[1])
        data = torch.stack([item[0] for item in combined])
        # print('data:', data)
        return data


class DeepCausalModel(nn.Module):
    def __init__(self,  n_treatments=2, treatments=[0, 1], x_dims=[20, 80, 160, 20], y_dims=[20, 40, 20]
                 ,x_emb_size=20, y_emb_size=20, output_dim=1):
        super(DeepCausalModel, self).__init__()

        # self.normalizer_layer = normalizer_layer
        self.n_treatments = n_treatments
        self.treatments = treatments
        self.x_layers = MLP(dims=x_dims)
        self.y_layers = nn.ModuleList([ MLP(dims=y_dims) for _ in range(n_treatments) ])
        self.y_outputs = nn.ModuleList([ nn.Linear(y_emb_size, output_dim) for _ in range(n_treatments) ])
        self.t_outputs = nn.Linear(x_emb_size, 1)
        self.splitter = SplitStreams()
        self.output = GatherStreams()

    def forward(self, x, treatment=None):

        # if self.normalizer_layer:
        #     x = self.normalizer_layer(x)
        x_emb = self.x_layers(x)

        if treatment is not None: # treatment有值，训练
            treatment = treatment.long()
            # shape [n_treatments, x_emb, index_treatment], 每个treatment下的x_emb，以及对应的index向量
            streams = [self.splitter(x_emb, treatment, torch.tensor(index_treatment, dtype=torch.int64)) for index_treatment in self.treatments ]
            x_streams, index_streams = zip(*streams)
            # 每个treatment下的x_emb经过y_layers得到的y_emb
            y_streams = [y_layer(x_stream) for y_layer, x_stream in zip(self.y_layers, x_streams)]
            # 每个treatment下的x_emb经过y_layers得到的y_emb, 经过输出头得到的输出
            y_outputs = [y_output(x_stream) for y_output, x_stream in zip(self.y_outputs, y_streams)]
            # 每个x_emb经过t分类头得到的倾向分数
            t_outputs = F.softmax(self.t_outputs(x_emb), dim=1)
            # print(y_outputs)
            # print(index_streams)
            # 按照index组装y的输出，使其与x的顺序一致
            return self.output(y_outputs, index_streams, self.n_treatments), x_emb, t_outputs

        else: # treatment无值，推理阶段，输出全部干预下的y预估值
            x_streams = [y_layer(x_emb) for y_layer in self.y_layers]
            x_streams = [y_output(x_stream) for y_output, x_stream in zip(self.y_outputs, x_streams)]
            return torch.stack(x_streams, dim=1)