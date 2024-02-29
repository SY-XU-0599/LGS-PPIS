import numpy as np
from torch.utils.data import Dataset
import torch,pickle
from torch_geometric.utils import degree

def load_graph(sequence_name,args):
    dismap = np.load("./Feature/distance_map/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= args.MAP_CUTOFF))
    distance_filter = (mask+0.0)*dismap
    # if MAP_TYPE == "d":
    adjacency_matrix = mask.astype(np.int)
    # elif MAP_TYPE == "c":
    #     adjacency_matrix = norm_dis(dismap)
    #     adjacency_matrix = mask * adjacency_matrix
    # norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return adjacency_matrix,distance_filter


def embedding(sequence_name, seq, embedding_type):
    if embedding_type == "b":
        seq_embedding = []
        Max_blosum = np.array([4, 5, 6, 6, 9, 5, 5, 6, 8, 4, 4, 5, 5, 6, 7, 4, 5, 11, 7, 4])
        Min_blosum = np.array([-3, -3, -4, -4, -4, -3, -4, -4, -3, -4, -4, -3, -3, -4, -4, -3, -2, -4, -3, -3])
        with open("./Feature/blosum/blosum_dict.pkl", "rb") as f:
            blosum_dict = pickle.load(f)
        for aa in seq:
            seq_embedding.append(blosum_dict[aa])
        seq_embedding = (np.array(seq_embedding) - Min_blosum) / (Max_blosum - Min_blosum)
    elif embedding_type == "e":
        pssm_feature = np.load("./Feature/pssm/" + sequence_name + '.npy')
        hmm_feature = np.load("./Feature/hmm/" + sequence_name + '.npy')
        seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis = 1)
    return seq_embedding.astype(np.float32)


def get_dssp_features(sequence_name):
    dssp_feature = np.load("./Feature/dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)


class ProDataset(Dataset):
    def __init__(self, dataframe,args):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.args = args
        # self.lm_emb_dict = {}
        # self.structural_features_dict = {}
        # self.sequence_embedding_dict = {}
        # self.graph_dict = {}
        # self.distance_map_dict = {}
        # for idx,sequence_name in enumerate(tqdm(self.names)):
        #     lm_emb = np.load('/media/ST-18T/Ma/GraphPPIS-master/ESM2_emb/' + sequence_name + '.npy')
        #     self.lm_emb_dict[sequence_name] = lm_emb
        #
        #     self.structural_features_dict[sequence_name] = get_dssp_features(sequence_name)  #
        #
        #     sequence_embedding = embedding(sequence_name, self.sequences[idx], EMBEDDING)
        #     self.sequence_embedding_dict[sequence_name] = sequence_embedding
        #
        #     graph, distance_map = load_graph(sequence_name)
        #
        #     self.graph_dict[sequence_name] = graph
        #     self.distance_map_dict[sequence_name] = distance_map



    def _rbf(self,D, num_rbf=16):
        D_min, D_max, D_count = 0., 20., num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count)#.to(D.device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = np.expand_dims(D, (0,-2,-1))
        RBF = np.exp(-((D_expand - D_mu.numpy()) / D_sigma) ** 2)
        return RBF.squeeze()
    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])

        lm_emb = np.load('./ESM2_emb/'+sequence_name+'.npy')#
        # lm_emb = self.lm_emb_dict[sequence_name]

        sequence_embedding = embedding(sequence_name, sequence, 'e')#
        # sequence_embedding = self.sequence_embedding_dict[sequence_name]

        structural_features = get_dssp_features(sequence_name)#
        # structural_features = self.structural_features_dict[sequence_name]

        node_features = np.concatenate([sequence_embedding, structural_features,lm_emb], axis = 1)

        graph,distance_map = load_graph(sequence_name,self.args)#
        # graph = self.graph_dict[sequence_name]
        # distance_map = self.distance_map_dict[sequence_name]

        adj = np.argwhere(graph==1).transpose(1,0)
        dist_rbf = self._rbf(distance_map[adj[0],adj[1]])

        #edge_index = torch.stack(torch.where(graph == 1), dim=0)
        row, col = adj
        deg = degree(torch.LongTensor(col), node_features.shape[0])
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return sequence_name, sequence, label, node_features, adj,dist_rbf,norm

    def __len__(self):
        return len(self.labels)