import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class SiameseAimCLR(nn.Module):
    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128,
                 momentum=0.999, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5, proj_depth=3,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        m: momentum of updating key encoder (default: 0.999)
        """
        super().__init__()
        encoder_type = base_encoder
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            if 'st_gcn' in encoder_type:
                self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                              hidden_dim=hidden_dim, num_class=num_class,
                                              dropout=dropout, graph_args=graph_args,
                                              edge_importance_weighting=edge_importance_weighting,
                                              **kwargs)
            elif 'agcn' in encoder_type:
                self.encoder_q = base_encoder(
                    in_channels=in_channels, num_class=num_class, graph_args=graph_args)

            elif 'stsgcn' in encoder_type:
                self.encoder_q = base_encoder(input_channels=in_channels, input_time_frame=50, st_gcnn_dropout=dropout,
                                              joints_to_consider=25, hidden_dim=hidden_dim, num_class=num_class)

            else:
                raise ValueError()

        else:
            self.m = momentum

            if 'st_gcn' in encoder_type:
                self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                              hidden_dim=hidden_dim, num_class=feature_dim,
                                              dropout=dropout, graph_args=graph_args,
                                              edge_importance_weighting=edge_importance_weighting,
                                              **kwargs)
                self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                              hidden_dim=hidden_dim, num_class=feature_dim,
                                              dropout=dropout, graph_args=graph_args,
                                              edge_importance_weighting=edge_importance_weighting,
                                              **kwargs)

            elif 'agcn' in encoder_type:
                self.encoder_q = base_encoder(
                    in_channels=in_channels, num_class=feature_dim, graph_args=graph_args)
                self.encoder_k = base_encoder(
                    in_channels=in_channels, num_class=feature_dim, graph_args=graph_args)

            elif 'stsgcn' in encoder_type:
                self.encoder_q = base_encoder(input_channels=in_channels, input_time_frame=50, st_gcnn_dropout=dropout,
                                              joints_to_consider=25, hidden_dim=hidden_dim, num_class=feature_dim)
                self.encoder_k = base_encoder(input_channels=in_channels, input_time_frame=50, st_gcnn_dropout=dropout,
                                              joints_to_consider=25, hidden_dim=hidden_dim, num_class=feature_dim)

            else:
                raise ValueError()

            if mlp:  # hack: brute-force replacement
                self.add_projector(self.encoder_q, proj_depth)
                self.add_projector(self.encoder_k, proj_depth)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            pred_hidden_dim = feature_dim//2
            self.predictor = nn.Sequential(nn.Linear(feature_dim, pred_hidden_dim, bias=False),
                                           nn.BatchNorm1d(pred_hidden_dim),
                                           nn.ReLU(inplace=True),  # hidden layer
                                           nn.Linear(pred_hidden_dim, feature_dim))  # output layer

    def add_projector(self, encoder=None, num_layers=3):
        feature_dim, dim_mlp = encoder.fc.weight.shape

        all_layers = []
        for _ in range(num_layers-1):
            all_layers += [nn.Linear(dim_mlp, dim_mlp, bias=False),
                           nn.BatchNorm1d(dim_mlp),
                           nn.ReLU(inplace=True)]

        # all_layers += [encoder.fc, nn.BatchNorm1d(feature_dim)]
        all_layers += [encoder.fc]
        encoder.fc = nn.Sequential(*all_layers)

    @ torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, im_q_extreme, im_q, im_k=None, topk=1):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        q = self.predictor(q)
        q_extreme = self.predictor(q_extreme)
        q_extreme_drop = self.predictor(q_extreme_drop)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        return q, q_extreme, q_extreme_drop, k
