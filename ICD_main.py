# -*- coding: UTF-8 -*-
from manifolds.poincare import PoincareManifold
import torch
import random
torch.manual_seed(1)
random.seed(1)
from torch import nn
import treelib
import pickle
from tqdm import tqdm
from rsgd import RiemannianSGD
import matplotlib.pyplot as plt
from treelib import Tree
import json


class PoincareEmbed(nn.Module):
    def __init__(self, num_nodes, reset_root=False, hidden_dim=2, epsilon=1e-10):
        super(PoincareEmbed, self).__init__()
        # resetting root node's embedding to all zero,
        # ensuring root node's embedding in the center of poincaré ball
        self.reset_root = reset_root
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.epsilon = epsilon
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        self.embedding.weight.data.uniform_(-1e-3, 1e-3)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.poincare = PoincareManifold()
    def proj(self):
        norm = self.embedding.weight.data.norm(p=2, dim=1).unsqueeze(1)
        norm[norm < 1] = 1
        self.embedding.weight.data /= norm
        self.embedding.weight.data -= self.epsilon
        if self.reset_root:
            self.embedding.weight.data[0] *= 0
    def distance(self, u, v):
        return self.poincare.distance(u, v)
    def forward(self, data):
        # data [batch_size, num_positive_sample(default 1) + num_negative_sample]
        data = self.embedding(data)
        v = data[:, 1:, :]
        u = data[:, 0, :].unsqueeze(1).expand_as(v)
        dist = - self.distance(u, v)
        # dist [batch_size, ]
        # first index represent positive sample
        labels = torch.zeros(u.shape[0]).long()
        loss = self.loss(dist, labels)
        return loss

if __name__ == '__main__':
    def build_tree(dic, tree):
        for curr_node, children_nodes in dic.items():
            curr_id = curr_node.split("$")[0]
            if tree.get_node(curr_id) is None:
                tree.create_node(curr_node, curr_node.split("$")[0])
            for child_node in children_nodes['children']:
                if isinstance(child_node, dict):
                    c_tag = list(child_node.keys())[0]
                else:
                    c_tag = child_node
                tree.create_node(c_tag, c_tag.split("$")[0], parent=curr_node.split("$")[0])
                if isinstance(child_node, dict):
                    build_tree(child_node, tree)
    tree = Tree()

    with open('tree.json', 'r', encoding='utf-8') as f:
        tree_json = eval(json.load(f))
    build_tree(tree_json, tree)
    root_node = tree.root
    print(tree.show())

    lr = 1
    epochs = 1000
    # burn-in step
    burn_in = 100
    # lambda for L2 regularization
    l2_lambda = 0.01
    # num of negative samples, num of positive sample is 1
    num_negative = 10
    output_image_filename = 'poincare.png'
    # resetting root node's embedding to all zero,
    # ensuring root node's embedding in the center of poincaré ball
    reset_root = False
    all_nodes = tree.all_nodes()
    # dict map indices to ICD code
    idx2icd = {i: node.identifier for i, node in enumerate(all_nodes)}
    # dict map ICD code to idx
    icd2idx = {v: k for k, v in idx2icd.items()}
    num_nodes = len(idx2icd)
    model = PoincareEmbed(num_nodes=num_nodes, reset_root=reset_root)
    optimizer = RiemannianSGD(params=model.parameters(),
                              lr=lr,
                              rgrad=model.poincare.rgrad,
                              expm=model.poincare.expm)

    def get_pos_and_neg_nodes(code, tree, num_negs=5):
        # positive codes is the direct children nodes of current node
        positive_icds = [i.identifier for i in tree.children(code)]
        # exclude nodes connected to current node from negative samples
        negative_icds = set(icd2idx.keys()) - set(positive_icds) - set(code)
        if code != root_node:
            negative_icds -= set(tree.parent(code).identifier)
        positive_icds = sorted(positive_icds)
        negative_icds = sorted(negative_icds)
        random.shuffle(positive_icds)
        random.shuffle(negative_icds)
        return [positive_icds[0], ], negative_icds[:num_negs]

    def get_data(train_nodes, num_negative):
        train_data = []
        # random.shuffle(train_nodes)
        for node_ in train_nodes:
            data_ = [icd2idx[node_]]
            pos_, neg_ = get_pos_and_neg_nodes(node_, tree, num_negs=num_negative)
            data_ += [icd2idx[i] for i in pos_]
            data_ += [icd2idx[i] for i in neg_]
            train_data.append(data_)
        train_data = torch.tensor(train_data)
        return train_data


    # exclude leaf nodes from training data
    train_nodes = sorted(set(icd2idx) - set([i.identifier for i in tree.leaves()]))

    for epoch in tqdm(range(epochs)):
        if epoch < burn_in:
            for g in optimizer.param_groups:
                g['lr'] = lr / 10
        else:
            for g in optimizer.param_groups:
                g['lr'] = lr
        train_data = get_data(train_nodes, num_negative)
        loss = model(train_data)
        # L2 regularization
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
        if epoch % 50 == 0:
            print('loss:', loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.proj()

    # get 2d coordinates
    coordinates = model.embedding.weight.data.numpy()
    fig = plt.figure(figsize=(20, 20))
    ax = plt.gca()
    ax.cla()  # clear things for fresh plot
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    # draw circle
    circle = plt.Circle((0, 0), 1., color='black', fill=False)
    ax.add_artist(circle)
    # draw nodes
    for i in range(coordinates.shape[0]):
        plt.annotate(idx2icd[i], (coordinates[i, 0], coordinates[i, 1]),
                     bbox={"fc": "white", "alpha": 0.5})
    # draw lines
    for node in tree.all_nodes():
        curr_id = node.identifier
        curr_id_coord = coordinates[icd2idx[curr_id]]
        for child in tree.children(curr_id):
            child_coord = coordinates[icd2idx[child.identifier]]
            ax.plot([curr_id_coord[0], child_coord[0]], [curr_id_coord[1], child_coord[1]], c='k', alpha=.3)

    plt.savefig(output_image_filename, bbox_inches='tight')
    print('saved to {}'.format(output_image_filename))















