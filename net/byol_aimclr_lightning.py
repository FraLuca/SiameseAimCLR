import torch
import torch.nn as nn
import torch.nn.functional as F


import copy
import random
from functools import wraps


# helper functions

def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn


def loss_fn(x, y, loss_name='cosine_sim'):

    if loss_name == 'cosine_sim':
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    elif loss_name == 'mse':
        loss = nn.MSELoss()
        return loss(x, y).mean(dim=-1)


# augmentation utils


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor


class MLP(nn.Module):
    def __init__(self, dim, projection_size):
        super().__init__()

        n_layers = len(projection_size)
        layer_list = []

        for idx, next_dim in enumerate(projection_size):

            if idx == n_layers-1:
                layer_list.append(nn.Linear(dim, next_dim))
            else:
                layer_list.append(nn.Linear(dim, next_dim))
                layer_list.append(nn.BatchNorm1d(next_dim))
                layer_list.append(nn.ReLU(inplace=True))
                dim = next_dim

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_hidden_size, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        # self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x, drop=False):
        if self.layer == -1:
            return self.net(x, drop)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x, drop)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection=True, drop=False):
        if drop:
            representation, representation_drop = self.get_representation(x, drop)
        else:
            representation = self.get_representation(x, drop)

        if not return_projection:
            return representation

        # just for the instanciation of the singleton
        projector = self._get_projector(representation)

        if drop:
            projection_ext = projector(representation)
            projection_drop = projector(representation_drop)
            return projection_ext, projection_drop
        else:
            projection = projector(representation)
            return projection


# main class
class BYOLAimCLR(nn.Module):
    def __init__(
        self, base_encoder=None, pretrain=True, use_nnm=False, queue_size=32768,
        time_dim=50, spat_dim=25, in_channels=3, hidden_channels=64, out_channels=1024,  # encoder parameters
        # projector and predictor parameters
        hidden_layer=-1, projection_hidden_size=[4096, 1024], predictor_hidden_size=[1024, 1024],
        moving_average_decay=0.999, use_momentum=True,  # momentum update parameters
        # other encoder parameters
        dropout=0.5, graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'}, edge_importance_weighting=True,
        loss_name='cosine_sim', **kwargs
    ):
        super().__init__()

        self.loss_name = loss_name
        encoder_type = base_encoder

        self.pretrain = pretrain

        if not self.pretrain:
            net = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                               dropout=dropout, graph_args=graph_args,
                               edge_importance_weighting=edge_importance_weighting, **kwargs)
            self.online_encoder = NetWrapper(
                net, projection_hidden_size, layer=hidden_layer)
        else:
            net = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                               dropout=dropout, graph_args=graph_args,
                               edge_importance_weighting=edge_importance_weighting, **kwargs)
            # self.net = net
            self.online_encoder = NetWrapper(
                net, projection_hidden_size, layer=hidden_layer)

            self.online_predictor = MLP(projection_hidden_size[-1], predictor_hidden_size)

            self.use_momentum = use_momentum
            self.target_encoder = None
            self.target_ema_updater = EMA(moving_average_decay)

            self.use_nnm = use_nnm
            if use_nnm:
                # create the queue
                self.K = queue_size
                self.register_buffer("queue", torch.randn(projection_hidden_size[-1], queue_size))
                self.queue = F.normalize(self.queue, dim=0)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)
        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, in_channels, time_dim, spat_dim, 2, device=device), torch.randn(
            2, in_channels, time_dim, spat_dim, 2, device=device), torch.randn(2, in_channels, time_dim, spat_dim, 2, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        self.queue = torch.cat((self.queue[:, batch_size:], keys.T), dim=1)

    def forward(
        self,
        image_one_extreme=None, image_one=None, image_two=None,
        return_projection=True,
        nnm=False, topk=1
    ):
        assert not (
            self.training and image_one.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if nnm:
            return self.nearest_neighbors_mining(image_one, image_two, image_one_extreme, topk)

        if not self.pretrain:
            return self.online_encoder(image_one, return_projection=return_projection)

        online_proj_one = self.online_encoder(image_one)
        online_proj_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        if image_one_extreme != None:
            online_proj_one_ext, online_proj_one_ext_drop = self.online_encoder(
                image_one_extreme, drop=True)
            online_pred_one_ext = self.online_predictor(online_proj_one_ext)
            online_pred_one_ext_drop = self.online_predictor(online_proj_one_ext_drop)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one = target_encoder(image_one)
            target_proj_two = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

            if image_one_extreme != None:
                target_proj_one_ext, target_proj_one_ext_drop = target_encoder(
                    image_one_extreme, drop=True)
                target_proj_one_ext.detach_()
                target_proj_one_ext_drop.detach_()

        # Online VS Target
        loss_one = loss_fn(online_pred_one, target_proj_two.detach(), loss_name=self.loss_name)
        loss_two = loss_fn(online_pred_two, target_proj_one.detach(), loss_name=self.loss_name)
        loss = loss_one + loss_two
        loss = loss.mean()

        loss_ext = None
        loss_ext_drop = None
        if image_one_extreme != None:
            # Online_Extreme VS Target
            loss_one_ext = loss_fn(online_pred_one_ext,
                                   target_proj_two.detach(), loss_name=self.loss_name)
            loss_two_ext = loss_fn(
                online_pred_two, target_proj_one_ext.detach(), loss_name=self.loss_name)
            loss_ext = loss_one_ext + loss_two_ext
            loss_ext = loss_ext.mean()

            # Online_Extreme_Drop VS Target
            loss_one_ext_drop = loss_fn(online_pred_one_ext_drop,
                                        target_proj_two.detach(), loss_name=self.loss_name)
            loss_two_ext_drop = loss_fn(
                online_pred_two, target_proj_one_ext_drop.detach(), loss_name=self.loss_name)
            loss_ext_drop = loss_one_ext_drop + loss_two_ext_drop
            loss_ext_drop = loss_ext_drop.mean()

        if self.use_nnm and not target_proj_one.shape[0] == 2:
            self._dequeue_and_enqueue(target_proj_one)

        return loss, loss_ext, loss_ext_drop

    def nearest_neighbors_mining(self, image_one, image_two, image_one_extreme, topk):

        online_proj_one = self.online_encoder(image_one)
        online_proj_two = self.online_encoder(image_two)

        l_neg = torch.einsum('nc,ck->nk', [online_proj_one, self.queue.clone().detach()])

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        if image_one_extreme != None:
            online_proj_one_ext, online_proj_one_ext_drop = self.online_encoder(
                image_one_extreme, drop=True)
            online_pred_one_ext = self.online_predictor(online_proj_one_ext)
            online_pred_one_ext_drop = self.online_predictor(online_proj_one_ext_drop)

            l_neg_e = torch.einsum('nc,ck->nk', [online_proj_one_ext, self.queue.clone().detach()])
            l_neg_ed = torch.einsum(
                'nc,ck->nk', [online_proj_one_ext_drop, self.queue.clone().detach()])

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one = target_encoder(image_one)
            target_proj_two = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

            if image_one_extreme != None:
                target_proj_one_ext, target_proj_one_ext_drop = target_encoder(
                    image_one_extreme, drop=True)
                target_proj_one_ext.detach_()
                target_proj_one_ext_drop.detach_()

        # Online VS Target
        loss_one = loss_fn(online_pred_one, target_proj_two.detach(), loss_name=self.loss_name)
        loss_two = loss_fn(online_pred_two, target_proj_one.detach(), loss_name=self.loss_name)
        loss = loss_one + loss_two
        loss = loss.mean()

        loss_ext = None
        loss_ext_drop = None
        if image_one_extreme != None:

            # Online_Extreme VS Target
            loss_one_ext = loss_fn(online_pred_one_ext,
                                   target_proj_two.detach(), loss_name=self.loss_name)
            loss_two_ext = loss_fn(
                online_pred_two, target_proj_one_ext.detach(), loss_name=self.loss_name)
            loss_ext = loss_one_ext + loss_two_ext
            loss_ext = loss_ext.mean()

            # Online_Extreme_Drop VS Target
            loss_one_ext_drop = loss_fn(online_pred_one_ext_drop,
                                        target_proj_two.detach(), loss_name=self.loss_name)
            loss_two_ext_drop = loss_fn(
                online_pred_two, target_proj_one_ext_drop.detach(), loss_name=self.loss_name)
            loss_ext_drop = loss_one_ext_drop + loss_two_ext_drop
            loss_ext_drop = loss_ext_drop.mean()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        new_positive = self.queue.T[topkdix.reshape(-1)]
        new_positive_ext = self.queue.T[topkdix_e.reshape(-1)]
        new_positive_ext_drop = self.queue.T[topkdix_ed.reshape(-1)]

        online_pred_new = self.online_predictor(new_positive)
        online_pred_new_ext = self.online_predictor(new_positive_ext)
        online_pred_new_ext_drop = self.online_predictor(new_positive_ext_drop)

        # Online VS Target
        loss_one_new = loss_fn(online_pred_new, target_proj_two.detach(), loss_name=self.loss_name)
        loss_two_new = loss_fn(online_pred_new, target_proj_one.detach(), loss_name=self.loss_name)
        loss_new = loss_one_new + loss_two_new
        loss_new = loss_new.mean()

        loss_ext_new = None
        loss_ext_drop_new = None
        if image_one_extreme != None:
            # Online_Extreme VS Target
            loss_one_ext_new = loss_fn(
                online_pred_new_ext, target_proj_two.detach(), loss_name=self.loss_name)
            loss_two_ext_new = loss_fn(
                online_pred_new_ext, target_proj_one.detach(), loss_name=self.loss_name)
            loss_ext_new = loss_one_ext_new + loss_two_ext_new
            loss_ext_new = loss_ext_new.mean()

            # Online_Extreme_Drop VS Target
            loss_one_ext_drop_new = loss_fn(
                online_pred_new_ext_drop, target_proj_two.detach(), loss_name=self.loss_name)
            loss_two_ext_drop_new = loss_fn(
                online_pred_new_ext_drop, target_proj_one.detach(), loss_name=self.loss_name)
            loss_ext_drop_new = loss_one_ext_drop_new + loss_two_ext_drop_new
            loss_ext_drop_new = loss_ext_drop_new.mean()

        self._dequeue_and_enqueue(target_proj_one)

        return loss, loss_ext, loss_ext_drop, loss_new, loss_ext_new, loss_ext_drop_new
