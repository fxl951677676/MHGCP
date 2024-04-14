import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import math

class MODEL(nn.Module):
    def __init__(self, args, userNum, itemNum, userMat, itemMat, uiMat, hide_dim, Layers):
        super(MODEL, self).__init__()
        self.args = args
        self.userNum = userNum
        self.itemNum = itemNum
        self.uuMat = userMat
        self.iiMat = itemMat
        self.uiMat = uiMat

        
        self.hide_dim = hide_dim
        self.LayerNums = Layers
        hete_hide_dim = args.hete_hide_dim
        self.FLayers = args.FLayers

        self.initializer = nn.init.xavier_uniform_
        uimat   = self.uiMat[: self.userNum,  self.userNum:]
        values  = torch.FloatTensor(uimat.tocoo().data)
        indices = np.vstack(( uimat.tocoo().row,  uimat.tocoo().col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape =  uimat.tocoo().shape
        uimat1=torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.uiadj = uimat1
        self.iuadj = uimat1.transpose(0,1)
        self.act = self.args.activation
        # self.init_all_weight(hide_dim, hete_hide_dim)
        self.proj_w1_uu=nn.Parameter(torch.FloatTensor(hide_dim,hete_hide_dim))
        self.initializer(self.proj_w1_uu.data)

        self.proj_w2_uu=nn.Parameter(torch.FloatTensor(hide_dim,hete_hide_dim))
        self.initializer(self.proj_w2_uu.data)

        self.proj_w1_iti=nn.Parameter(torch.FloatTensor(hide_dim,hete_hide_dim))
        self.initializer(self.proj_w1_iti.data)

        self.proj_w2_iti=nn.Parameter(torch.FloatTensor(hide_dim,hete_hide_dim))
        self.initializer(self.proj_w2_iti.data)

        self.proj_b_uu=nn.Parameter(torch.FloatTensor(1,hete_hide_dim))
        self.initializer(self.proj_b_uu.data)

        self.proj_b_iti=nn.Parameter(torch.FloatTensor(1,hete_hide_dim))
        self.initializer(self.proj_b_iti.data)

        self.inv_proj_w1_uu=nn.Parameter(torch.FloatTensor(hete_hide_dim,hide_dim))
        self.initializer(self.inv_proj_w1_uu.data)

        self.inv_proj_w2_uu=nn.Parameter(torch.FloatTensor(hete_hide_dim,hide_dim))
        self.initializer(self.inv_proj_w2_uu.data)

        self.inv_proj_w1_iti=nn.Parameter(torch.FloatTensor(hete_hide_dim,hide_dim))
        self.initializer(self.inv_proj_w1_iti.data)

        self.inv_proj_w2_iti=nn.Parameter(torch.FloatTensor(hete_hide_dim,hide_dim))
        self.initializer(self.inv_proj_w2_iti.data)

        self.inv_proj_b_uu=nn.Parameter(torch.FloatTensor(1,hide_dim))
        self.initializer(self.inv_proj_b_uu.data)

        self.inv_proj_b_iti=nn.Parameter(torch.FloatTensor(1,hide_dim))
        self.initializer(self.inv_proj_b_iti.data)
        self.a_u=nn.Parameter(torch.FloatTensor([0.5]))
        self.a_i=nn.Parameter(torch.FloatTensor([0.5]))

        self.encoder = nn.ModuleList()
        self.heteencoder = nn.ModuleList()
        for i in range(0, self.LayerNums):
            self.encoder.append(GCN_layer())
            self.heteencoder.append(Hete_GCN_layers(self.FLayers))
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(self.initializer(t.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(self.initializer(t.empty(itemNum, hide_dim)))
        })
        self.auu=nn.Parameter(torch.FloatTensor([1 , 3 , 3]))
        self.aii=nn.Parameter(torch.FloatTensor([1 , 3 , 3]))



    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def proj(self, emb, weight1, weight2, b):
        if self.act == 'Tanh':
            m = nn.Tanh()
        if self.act == 'ELU':
            m = nn.ELU()
        if self.act == 'LeakyReLU':
            m = nn.LeakyReLU()
        return torch.mul((torch.matmul(emb, weight2)),(m(torch.matmul(emb, weight1) + b)))  
    
    def inv_proj(self, emb, weight1, weight2, b):
        if self.act == 'Tanh':
            m = nn.Tanh()
        if self.act == 'ELU':
            m = nn.ELU()
        if self.act == 'LeakyReLU':
            m = nn.LeakyReLU()
        return torch.mul((torch.matmul(emb, weight2)),m(torch.matmul(emb, weight1) + b))

    

    
    def l2_penalty(self, w):
        return torch.sum(w.pow(2)) / 2

    def forward(self, iftraining, uid, iid, warm_up_flag = 0, norm = 1 ):
        if warm_up_flag == 1:
            self.auu.requires_grad = False
            self.aii.requires_grad = False
            self.a_u.requires_grad = False
            self.a_i.requires_grad = False
        if warm_up_flag == 0:
            self.auu.requires_grad = True
            self.aii.requires_grad = True
            self.a_u.requires_grad = True
            self.a_i.requires_grad = True
        item_index=np.arange(0,self.itemNum)
        user_index=np.arange(0,self.userNum)
        ui_index = np.array(user_index.tolist() + [ i + self.userNum for i in item_index])
        
        # Initialize Embeddings
        userembed0 = self.embedding_dict['user_emb']
        itemembed0 = self.embedding_dict['item_emb']
        self.ui_embeddings = t.cat([ userembed0, itemembed0], 0)




        # Inv-Proj layer
        uu_embed0 = self.proj(userembed0, self.proj_w1_uu, self.proj_w2_uu, self.proj_b_uu)
        iti_embed0 = self.proj(itemembed0, self.proj_w1_iti, self.proj_w2_iti, self.proj_b_iti)



        self.all_user_embeddings = [userembed0]
        self.all_item_embeddings = [itemembed0]
        self.all_ui_embeddings   = [self.ui_embeddings]
        
        # Agg
        for i in range(len(self.encoder)):
            layer = self.encoder[i]
            hete_layers = self.heteencoder[i]
            if i == 0:  
                uuEmbeddings = hete_layers(uu_embed0, self.uuMat, user_index, self.auu)
                itiEmbeddings = hete_layers(iti_embed0, self.iiMat, item_index, self.aii)
                uiEmbeddings0   = layer(self.ui_embeddings, self.uiMat, ui_index)


                
            else:
                uuEmbeddings = hete_layers(uuEmbeddings, self.uuMat, user_index, self.auu)
                itiEmbeddings = hete_layers(itiEmbeddings, self.iiMat, item_index, self.aii)
                uiEmbeddings0   = layer(uiEmbeddings,   self.uiMat, ui_index)



            #Proj layer 
            user_hete_Embeddings_projed = self.inv_proj(uuEmbeddings, self.inv_proj_w1_uu, self.inv_proj_w2_uu ,self.inv_proj_b_uu)
            item_hete_Embeddings_projed = self.inv_proj(itiEmbeddings, self.inv_proj_w1_iti, self.inv_proj_w2_iti, self.inv_proj_b_iti)

            
            # Aggregation of message features across the two related views in the middle layer then fed into the next layer
            self.ui_userEmbedding0, self.ui_itemEmbedding0 = t.split(uiEmbeddings0, [self.userNum, self.itemNum])

            userEd=self.a_u*self.ui_userEmbedding0 + (1-self.a_u)*user_hete_Embeddings_projed
            itemEd=self.a_i*self.ui_itemEmbedding0 + (1-self.a_i)*item_hete_Embeddings_projed

            uuEmbeddings = self.proj(userEd, self.proj_w1_uu, self.proj_w2_uu,self.proj_b_uu)
            itiEmbeddings = self.proj(itemEd, self.proj_w1_iti, self.proj_w2_iti, self.proj_b_iti)

            userEmbeddings=userEd
            itemEmbeddings=itemEd
            uiEmbeddings=torch.cat([userEd,itemEd], 0)

            userEmbeddings = F.normalize(userEmbeddings, p=2, dim=1)
            self.all_user_embeddings += [userEmbeddings]
            itemEmbeddings = F.normalize(itemEmbeddings, p=2, dim=1)
            self.all_item_embeddings += [itemEmbeddings]

        self.userEmbedding = t.stack(self.all_user_embeddings, dim=1)
        self.userEmbedding = t.mean(self.userEmbedding, dim = 1)
        self.itemEmbedding = t.stack(self.all_item_embeddings, dim=1)  
        self.itemEmbedding = t.mean(self.itemEmbedding, dim = 1)

        projregjLoss = self.l2_penalty(self.proj_w1_uu.data) + self.l2_penalty(self.proj_w1_iti.data) + self.l2_penalty(self.inv_proj_w1_uu.data) +self.l2_penalty(self.inv_proj_w1_iti.data)+self.l2_penalty(self.proj_w2_uu.data) + self.l2_penalty(self.proj_w2_iti.data) + self.l2_penalty(self.inv_proj_w2_uu.data) + self.l2_penalty(self.inv_proj_w2_iti.data)
        return self.userEmbedding, self.itemEmbedding, projregjLoss


class GCN_layer(nn.Module):
    def __init__(self):
        super(GCN_layer, self).__init__()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()    

    def forward(self, features, Mat, index):
        subset_Mat = Mat
        subset_features = features
        subset_Mat = self.normalize_adj(subset_Mat)
        subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat).cuda()
        out_features = torch.spmm(subset_sparse_tensor, subset_features)
        new_features = torch.empty(features.shape).cuda()
        new_features[index] = out_features
        dif_index = np.setdiff1d(torch.arange(features.shape[0]), index)
        new_features[dif_index] = features[dif_index]
        return new_features


class Hete_GCN_layers(nn.Module):
    def __init__(self, FLayers = 2):
        super(Hete_GCN_layers, self).__init__()
        self.softmax = nn.Softmax(dim = 0)
        self.FeatureGCN = nn.ModuleList()
        self.Layers = FLayers
        for i in range(0,FLayers):
            self.FeatureGCN.append(GCN_layer())

    def forward(self, features, Mat, index, a_in):
        num = self.Layers + 1
        a_layers = a_in[:num]
        a = self.softmax(a_layers)
        result = a[0]*features
        hide_features = features
        for j in range(len(self.FeatureGCN)):
            GCN = self.FeatureGCN[j]
            hide_features = GCN(hide_features, Mat, index)
            result = result + a[j+1]*hide_features
        return result