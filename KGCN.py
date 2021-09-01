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

class Model(nn.Module):
    def __init__(self, g, in_feat_user, in_feat_repo, out_feats):
        super(Model, self).__init__()
        self.aggregate = aggregate_sum
        self.user_embedding = Embedding(in_feat_user, 125)
        self.repo_embedding = Embedding(in_feat_repo, 125)
        self.hidden = dglnn.HeteroGraphConv(
            { etype[1]: dglnn.GraphConv(125, 96) for etype in g.canonical_etypes },
            aggregate='sum')
        self.out = dglnn.HeteroGraphConv(
            { etype[1]: dglnn.GraphConv(96, out_feats) for etype in g.canonical_etypes },
            aggregate='sum')
        self.hidden_current_user = nn.Linear(125, 96)
        self.hidden_current_repo = nn.Linear(125, 96)
        self.out_current_user = nn.Linear(96, out_feats)
        self.out_current_repo = nn.Linear(96, out_feats)
        self.predict = nn.CosineSimilarity(dim=0)

    def forward(self, g, user_feat, repo_feat):
        h_user = self.user_embedding(user_feat)
        h_repo = self.repo_embedding(repo_feat)
        
        h_dict = {
            'user': h_user,
            'repo': h_repo
        }

        h = self.hidden(g, h_dict)
        
        h_current_user = self.hidden_current_user(h_user)
        h_current_repo = self.hidden_current_repo(h_repo)
        out_dict = {
            'user': F.relu(self.aggregate(h_current_user, h['user']) if 'user' in h else h_current_user),
            'repo': F.relu(self.aggregate(h_current_repo, h['repo']))
        }
        
        out = self.out(g, out_dict)
        
        out_current_user = self.out_current_user(out_dict['user'])
        out_current_repo = self.out_current_repo(out_dict['repo'])
        h_user_new = F.relu(self.aggregate(out_current_user, out['user']) if 'user' in out else out_current_user)
        h_repo_new = F.relu(self.aggregate(out_current_repo, out['repo']))
        
        # calculate the cosine similarity
        h_user_new_normalized = F.normalize(h_user_new, p=2, dim=1)
        h_repo_new_normalized = F.normalize(h_repo_new, p=2, dim=1)
        ratings = h_user_new_normalized @ h_repo_new_normalized.T
        
        return torch.flatten(ratings)

def concact(self_vectors, neighbour_vectors, dim=1):
    return torch.cat((self_vectors, neighbour_vectors), dim)

def aggregate_sum(self_vectors, neighbour_vectors, dim=0):
    return torch.sum(torch.stack((self_vectors, neighbour_vectors)), dim)