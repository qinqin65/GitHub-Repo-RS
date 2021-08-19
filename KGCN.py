import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class KGCN(nn.Module):
    def __init__(self, in_feat, out_feat, etype):
        super(KGCN, self).__init__()
        self.aggregate = concact
        self.linear = nn.Linear(in_feat * 2, out_feat)
        self.etype = etype

    def forward(self, g):
        with g.local_scope():
            g.apply_edges(fn.u_dot_e('h_%s' % self.etype, 'e', 'pi_r_u'), etype=self.etype)
            g.edges[self.etype].data['pi_r_u_exp'] = torch.exp(g.edges[self.etype].data['pi_r_u'])
            g.update_all(fn.copy_e('pi_r_u_exp', 'm'), fn.sum('m', 'pi_r_u_exp_sum'), etype=self.etype)
            g.apply_edges(fn.e_div_v('pi_r_u_exp', 'pi_r_u_exp_sum', 'pi_normalized'), etype=self.etype)
            g.update_all(fn.u_mul_e('h_%s' % self.etype, 'pi_normalized', 'm'), fn.sum('m', 'h_sum'), etype=self.etype)
            h_sum = g.nodes['repo'].data['h_sum']
            h_total = self.aggregate(g.nodes['repo'].data['h_%s' % self.etype], h_sum)
            return self.linear(h_total)

class Model(nn.Module):
    def __init__(self, in_feat, h_feats, etype):
        super(Model, self).__init__()
        self.conv1 = KGCN(in_feat, h_feats, etype)

    def forward(self, g):
        h = self.conv1(g)
        h = F.relu(h)
        y = torch.matmul(g.nodes['user'].data['h'], h.T)
        return torch.flatten(y)

def concact(self_vectors, neighbour_vectors, dim=1):
    return torch.cat((self_vectors, neighbour_vectors), dim)