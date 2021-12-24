import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn.pytorch as dglnn

class Embedding(nn.Module):
    def __init__(self, in_embedding, out_embedding):
        super(Embedding, self).__init__()
        self.linear = nn.Linear(in_embedding, out_embedding)
    
    def forward(self, node_features):
        return self.linear(node_features)

class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g, h):
        with g.local_scope():
            for etype in g.canonical_etypes:
                try:
                    g.nodes[etype[0]].data['norm_h'] = F.normalize(h[etype[0]], p=2, dim=-1)
                    g.nodes[etype[2]].data['norm_h'] = F.normalize(h[etype[2]], p=2, dim=-1)
                    g.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'), etype=etype)
                except KeyError:
                    pass
            ratings = g.edata['cos']
        return ratings

class Model(nn.Module):
    def __init__(self, g, in_feat_user, in_feat_repo, out_user_repo, out_hidden, out_feats, drop_out_rate=0.4):
        super(Model, self).__init__()
        self.aggregate = aggregate_sum
        self.user_embedding = Embedding(in_feat_user, out_user_repo)
        self.repo_embedding = Embedding(in_feat_repo, out_user_repo)
        self.hidden = dglnn.HeteroGraphConv(
            { etype[1]: dglnn.GraphConv(out_user_repo, out_hidden) for etype in g.canonical_etypes },
            aggregate='sum')
        self.out = dglnn.HeteroGraphConv(
            { etype[1]: dglnn.GraphConv(out_hidden, out_feats) for etype in g.canonical_etypes },
            aggregate='sum')
        self.predict = CosineSimilarity()
        self.dropout = nn.Dropout(p=drop_out_rate)

    def forward(self, g, pos_g, neg_g, user_feat, repo_feat):
        h_user = self.user_embedding(user_feat)
        h_repo = self.repo_embedding(repo_feat)
        
        h_dict = {
            'user': self.dropout(h_user),
            'repo': self.dropout(h_repo)
        }

        h = self.hidden(g, h_dict)

        h_dict = {
            'user': self.dropout(h['user']),
            'repo': self.dropout(h['repo'])
        }
        
        out = self.out(g, h_dict)
        
        pos_score = self.predict(pos_g, out)
        neg_score = self.predict(neg_g, out)
        
        return pos_score, neg_score

def concact(self_vectors, neighbour_vectors, dim=1):
    return torch.cat((self_vectors, neighbour_vectors), dim)

def aggregate_sum(self_vectors, neighbour_vectors, dim=0):
    return torch.sum(torch.stack((self_vectors, neighbour_vectors)), dim)

def loss_fn(pos_score, neg_score):
    delta = 0.25
    all_scores = torch.empty(0)

    for etype in pos_score.keys():
        neg_score_tensor = neg_score[etype]
        pos_score_tensor = pos_score[etype]
        negative_mask_tensor = torch.zeros(size=neg_score_tensor.shape)
        scores = neg_score_tensor + delta - pos_score_tensor - negative_mask_tensor
        relu = nn.ReLU()
        scores = relu(scores)
        all_scores = torch.cat((all_scores, scores), 0)
    return torch.mean(all_scores)