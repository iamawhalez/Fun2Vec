import torch.nn as nn
import torchvision.models as models
import torch
# import torch.nn as nn
import torch.nn.functional as F
from exceptions.exceptions import InvalidBackboneError
from fairseq.models.trex import TrexModel
# from command import configs


class func2vecforward(nn.Module):

    # def __init__(self, dirname,filename,input_dim, out_dim):
    def __init__(self, TrexModel,input_dim, out_dim):

        super(func2vecforward, self).__init__()
        self.trex = TrexModel.cuda()
        self.trex.model.register_similarity_head('similarity')
        self.dense = nn.Linear(input_dim, input_dim)
        self.activation_fn =nn.GELU()
        self.dropout = nn.Dropout(p=0.0)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, features):
        emb0_rep = self.trex.model(features, features_only=True, classification_head_name='similarity')[0]['features']
        emb0_rep=torch.mean(emb0_rep, dim=1)

        x = self.dense(emb0_rep)
        x = self.activation_fn(x)
        x = self.out_proj(x)
        x = F.normalize(x)

        return x
        # return emb0_rep