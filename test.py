import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn.pytorch as dglnn

class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, g, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv(
            { etype[1]: dglnn.GraphConv(in_features, hidden_features) for etype in g.canonical_etypes },
            aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv(
            { etype[1]: dglnn.GraphConv(hidden_features, out_features) for etype in g.canonical_etypes },
            aggregate='sum')

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x['game'] = F.relu(x['game'])
        x['user'] = F.relu(x['user'])
        x = self.conv2(blocks[1], x)
        x['game'] = F.relu(x['game'])
        x['user'] = F.relu(x['user'])
        return x

class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            for etype in g.canonical_etypes:
                try:
                    edge_subgraph.nodes[etype[0]].data['x'] = x[etype[0]]
                    edge_subgraph.nodes[etype[2]].data['x'] = x[etype[2]]
                    edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
                except KeyError as e:
                    print(e)
                    pass

            return edge_subgraph.edata['score']

class Model(nn.Module):
    def __init__(self, g, in_features, hidden_features, out_features):
        super().__init__()
        self.gcn = StochasticTwoLayerGCN(g, 
            in_features, hidden_features, out_features)
        self.predictor = ScorePredictor()

    def forward(self, positive_graph, negative_graph, blocks, x):
        x = self.gcn(blocks, x)
        pos_score = self.predictor(positive_graph, x)
        neg_score = self.predictor(negative_graph, x)
        return pos_score, neg_score

def compute_loss(pos_score, neg_score):
    # an example hinge loss
    score = 0
    i = 0
    for etype in pos_score.keys():
        neg_score_tensor = neg_score[etype]
        pos_score_tensor = pos_score[etype]
        n = pos_score_tensor.shape[0]
        score += (neg_score_tensor.view(n, -1) - pos_score_tensor.view(n, -1) + 1).clamp(min=0).mean()
        i += 1
    return score / i

if __name__ == "__main__":
    data_dict = {
        ('user', 'buys', 'game'): (torch.tensor([1, 1]), torch.tensor([1, 2])),
        ('user', 'plays', 'game'): (torch.tensor([0, 3]), torch.tensor([3, 4])),
        ('game', 'bought-by', 'user'): (torch.tensor([1, 2]), torch.tensor([1, 1])),
        ('game', 'played-by', 'user'): (torch.tensor([3, 4]), torch.tensor([0, 3]))
    }
    g = dgl.heterograph(data_dict)
    g.ndata['features'] = {
        'user': torch.randn((g.num_nodes('user'), 10)),
        'game': torch.randn((g.num_nodes('game'), 10))
    }
    edges_buys = g.edges(etype=('user', 'buys', 'game'))
    edges_plays = g.edges(etype=('user', 'plays', 'game'))
    train_seeds = {
        ('user', 'buys', 'game'): g.edge_ids(edges_buys[0], edges_buys[1], etype=('user', 'buys', 'game')),
        ('user', 'plays', 'game'): g.edge_ids(edges_plays[0], edges_plays[1], etype=('user', 'plays', 'game'))
    }

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    dataloader = dgl.dataloading.EdgeDataLoader(
    g, train_seeds, sampler,
    negative_sampler=dgl.dataloading.negative_sampler.Uniform(2),
    exclude='reverse_types',
    reverse_etypes={'buys': 'bought-by', 'bought-by': 'buys',
                    'plays': 'played-by', 'played-by': 'plays'},
    batch_size=2,
    shuffle=True,
    drop_last=False,
    pin_memory=True)

    model = Model(g, 10, 5, 2)
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(10):
        for input_nodes, positive_graph, negative_graph, blocks in dataloader:
            input_features = blocks[0].srcdata['features']
            pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
            loss = compute_loss(pos_score, neg_score)
            print(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()