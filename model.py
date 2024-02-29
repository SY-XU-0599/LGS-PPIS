from torch_geometric.nn import MessagePassing,GCNConv
import torch.nn as nn
import torch,math
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class EA_GCN(MessagePassing):
    def __init__(self,hidden_dim):
        super(EA_GCN, self).__init__(aggr='add')
        self.lin = nn.Linear(16,hidden_dim)
        self.W_EV = nn.Sequential(nn.Linear(hidden_dim*3, hidden_dim*2, bias=True),  # + num_in
                                  nn.GELU(),
                                  nn.Linear(hidden_dim*2, hidden_dim, bias=True),
                                  nn.GELU(),
                                  nn.Linear(hidden_dim, 16, bias=True)
                                  )
    def forward(self,x,edge_index,norm,distance_map):
        edge_attr = self.lin(distance_map.type(torch.float32))
        message = self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr)
        x_out = message + x
        distance_map_out = self.W_EV(torch.cat([x_out[edge_index[0]],edge_attr,x_out[edge_index[1]]],dim=-1))
        return x_out,distance_map_out
    def message(self, x_j,norm,edge_attr):
        return norm.view(-1, 1) * (x_j+edge_attr)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False):
        super(GraphConvolution, self).__init__()

        self.in_features = 2*in_features

        self.out_features = out_features
        self.residual = residual
        self.weight_local = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.weight_global = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.lightgcn = EA_GCN(out_features)

        # self.GCN = GCNConv(256,256)

        self.self_attn = torch.nn.MultiheadAttention(
            256, num_heads=2, dropout=0.1, batch_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight_local.data.uniform_(-stdv, stdv)
        self.weight_global.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l,distance_map,norm):
        theta = min(1, math.log(lamda/l+1))
        # hi = torch.spmm(adj, input)
        h_local,distance_map = self.lightgcn(input,adj,norm,distance_map)

        # h_local = self.GCN(input,adj)

        h_global = self.self_attn(input, input, input,
                                attn_mask=None,
                                key_padding_mask=None,
                                need_weights=False)[0]

        support_local = torch.cat([h_local,h0],1)
        r_local = (1-alpha)*h_local+alpha*h0

        support_global = torch.cat([h_global, h0], 1)
        r_global = (1 - alpha) * h_global + alpha * h0

        output_local = theta*torch.mm(support_local, self.weight_local)+(1-theta)*r_local

        output_global = theta * torch.mm(support_global, self.weight_global) + (1 - theta) * r_global

        if self.residual: # speed up convergence of the training process
            output = output_local+input+output_global
        else:
            output = output_local+output_global

        return output,distance_map


class deepGCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha):
        super(deepGCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):#8
            self.convs.append(GraphConvolution(nhidden, nhidden,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat+64, nhidden))#nhidden = 256
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        # self.self_attn = torch.nn.MultiheadAttention(
        #     nhidden, num_heads=1, dropout=dropout, batch_first=True)


    def forward(self, x, adj,distance_map,norm):


        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner,distance_map = con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1,distance_map,norm)
            layer_inner = self.act_fn(layer_inner)

            # layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            # layer_inner = self.act_fn(self.self_attn(layer_inner, layer_inner, layer_inner,
            #                         attn_mask=None,
            #                         key_padding_mask=None,
            #                         need_weights=False)[0])
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner


class LGS_PPIS(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha):
        super(LGS_PPIS, self).__init__()

        self.deep_gcn = deepGCN(nlayers = nlayers, nfeat = nfeat, nhidden = nhidden, nclass = nclass,
                                dropout = dropout, lamda = lamda, alpha = alpha)
        self.criterion = nn.CrossEntropyLoss() # automatically do softmax to the predicted value and one-hot to the label
        self.optimizer = torch.optim.Adam(self.parameters(), lr = (1E-3)/2, weight_decay = 0)
        self.FC_layer = nn.Sequential(nn.Linear(1280, 512),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(512, 128),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(128, 64)
                                 )
    def forward(self, x, adj,distance_map,norm):          # x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)

        x = x.float()
        x_lm = self.FC_layer(x[:,-1280:])
        x = torch.cat([x[:,:54],x_lm],dim=-1)#N,118
        output = self.deep_gcn(x,adj,distance_map,norm)  # output.shape = (seq_len, NUM_CLASSES)
        return output