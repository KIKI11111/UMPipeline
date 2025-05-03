import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


class MLP(nn.Module):
    def __init__(self, units, activation, num_layers):
        super(MLP, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(units, units))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError("Unsupported activation function")
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class SplitStreams(nn.Module):
    def __init__(self):
        super(SplitStreams, self).__init__()

    def forward(self, x, y, z):
        y = y.view(-1)
        indice_position = (y == z).nonzero(as_tuple=True)[0]
        return x[indice_position], indice_position


class GatherStreams(nn.Module):
    def __init__(self):
        super(GatherStreams, self).__init__()

    def forward(self, x_streams, indice_streams):
        # Concatenate the outputs from all treatments
        return torch.cat(x_streams, dim=0)


class DeepCausalModel(nn.Module):
    def __init__(self, normalizer_layer=None, n_treatments=2, treatments=[0, 1], output_dim=1, phi_layers=2, units=20, y_layers=3,
                 activation="relu", output_bias=None):
        super(DeepCausalModel, self).__init__()

        self.normalizer_layer = normalizer_layer
        self.phi = MLP(units=units, activation=activation, num_layers=phi_layers)

        self.splitter = SplitStreams()
        self.y_hiddens = nn.ModuleList([
            MLP(units=units, activation=activation, num_layers=y_layers)
            for _ in range(n_treatments)
        ])
        self.y_outputs = nn.ModuleList([
            nn.Linear(units, output_dim)
            for _ in range(n_treatments)
        ])
        self.t_outputs = nn.Linear(units, 1)
        if output_bias is not None:
            for layer in self.y_outputs:
                layer.bias.data.fill_(output_bias)

        self.n_treatments = n_treatments
        self.treatments = treatments
        self.output_ = GatherStreams()

    def forward(self, cofeatures_input, treatment_input=None):

        if self.normalizer_layer:
            cofeatures_input = self.normalizer_layer(cofeatures_input)

        x_flux = self.phi(cofeatures_input)

        if treatment_input is not None:
            treatment_cat = treatment_input.long()

            streams = [
                self.splitter(x_flux, treatment_cat, torch.tensor(indice_treatment, dtype=torch.int64))
                for indice_treatment in self.treatments
            ]

            x_streams, indice_streams = zip(*streams)

            x_streams = [
                y_hidden(x_stream) for y_hidden, x_stream in zip(self.y_hiddens, x_streams)
            ]

            x_streams = [
                y_output(x_stream) for y_output, x_stream in zip(self.y_outputs, x_streams)
            ]

            treatment_probs = F.softmax(self.t_outputs(x_flux), dim=1)

            return self.output_(x_streams, indice_streams), x_flux, treatment_probs

        else:
            x_streams = [
                y_hidden(x_flux) for y_hidden in self.y_hiddens
            ]
            x_streams = [
                y_output(x_stream) for y_output, x_stream in zip(self.y_outputs, x_streams)
            ]
            return torch.stack(x_streams, dim=1)