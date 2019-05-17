import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_spagat import SpGraphAttentionLayer


class SpaGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpaGAT, self).__init__()
        self.dropout = dropout
        
        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True, layerN='1_'+str(i)) for i in range(nheads[0])]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        self.out_atts = [SpGraphAttentionLayer(nhid * nheads[0], 
                                             nclass,
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False, layerN='2_'+str(i))  for i in range(nheads[1])]

        for i, out_att in enumerate(self.out_atts):
            self.add_module('out_att_{}'.format(i), out_att)


    def forward(self, x, adj, pathM=None, pathlens=[2,3], genPath=False, mode='GAT'):
        # mode can be GAT, PathAT, Combine
        # pathM: layer, head, pathLen
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, (None if pathM is None else pathM[0][0]), pathlens, genPath, mode) \
                    for head, att in enumerate(self.attentions)], dim=1)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.stack([out_att(x, adj, (None if pathM is None else pathM[0][0]), pathlens, genPath, mode) \
                    for head, out_att in enumerate(self.out_atts)], dim=2)
        
        x = torch.mean(x, dim=2)
        x = F.elu(x)
        return F.log_softmax(x, dim=1)