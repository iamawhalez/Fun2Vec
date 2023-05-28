import torch.nn as nn
import torchvision.models as models
import torch
# import torch.nn as nn
import torch.nn.functional as F
from exceptions.exceptions import InvalidBackboneError


class mytrexforward(nn.Module):

    def __init__(self, input_dim, out_dim):
        super(mytrexforward, self).__init__()
        # self.mytrexforward = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
        #                     "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        # self.backbone = self._get_basemodel(base_model)
        # dim_mlp = self.backbone.fc.out_dim

        # add mlp projection self.backbone.fc head Linear(in_features=512, out_features=128, bias=True)
        # self.dense = nn.Linear(input_dim, input_dim)
        # self.activation_fn = utils.get_activation_fn(activation_fn)
        # self.activation_fn = nn.GELU()
        # self.dropout = nn.Dropout(p=0.0)
        # self.out_proj = nn.Linear(input_dim, input_dim)
        self.backbone = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, out_dim))

    # def _get_basemodel(self, model_name):
    #     try:
    #         model = self.resnet_dict[model_name]
    #     except KeyError:
    #         raise InvalidBackboneError(
    #             "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
    #     else:
    #         return model

    def forward(self, features):
        # x = torch.mean(features, dim=1)
        # x = self.dropout(features)
        # x = self.dropout(features)
        # x = self.dense(features)
        # x = self.dense(x)

        # x = self.activation_fn(x)
        # x = self.dropout(x)
        # x = self.out_proj(x)
        # x = F.normalize(x)
        x=self.backbone(features)
        return x