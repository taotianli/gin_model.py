import torch
import dgl
import numpy as np
from loading_data import loading_feature
from dgl.nn.pytorch import pairwise_squared_distance
from typing import List, Tuple
import math


"""
输入：左/右脑特征矩阵
输出：每个脑区一个链接矩阵
特征按照脑区分类、计算相似性、kNN建图
采集每个vertex所属脑区，构建相关性矩阵
有几个脑区没有，有些脑区数据点很多，考虑随机采样，有些脑区很少，考虑不要了？
DGL有计算相关性的公式：无
"""


def split_rand(labels, split_ratio=0.7, seed=0, shuffle=True):
    '''

    Args:
        labels:
        split_ratio:
        seed:
        shuffle:

    Returns:

    '''
    num_entries = labels
    valid_mask = torch.from_numpy(np.array([0 for _ in range(labels)]).astype(bool))
    train_mask = torch.from_numpy(np.array([0 for _ in range(labels)]).astype(bool))
    test_mask = torch.from_numpy(np.array([0 for _ in range(labels)]).astype(bool))
    indices = list(range(num_entries))
    np.random.seed(seed)
    np.random.shuffle(indices)
    split = int(math.floor(split_ratio * num_entries))
    test_split = int(math.floor(0.9 * num_entries))
    train_idx, test_idx, valid_idx = indices[:split], indices[split:test_split], indices[test_split:]
    print("node_num = %d. train_set : test_set: valid_set = %d : %d : %d" % (labels, len(train_idx), len(test_idx), len(valid_idx)))
    for tid in train_idx:
        train_mask[tid] = torch.tensor(True)
    for teid in test_idx:
        test_mask[teid] = torch.tensor(True)
    for vid in valid_idx:
        valid_mask[vid] = torch.tensor(True)
    return train_mask, test_mask, valid_mask


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Atom3D (https://github.com/drorlab/atom3d/blob/master/benchmarking/pytorch_geometric/ppi_dataloader.py):
# -------------------------------------------------------------------------------------------------------------------------------------
def reading_brain_region(node_feats, knn: int):
    '''

    Args:
        node_feats: 节点特征
        knn: kNN中k的数值

    Returns:
        feature_matrix_dict: 一个被试所有脑区的knn graph
        knn_g: 建好的knn graph

    '''
    feature = node_feats[np.argsort(node_feats[:, 3])]
    np.delete(feature, 3, axis=1)
    counter = 0
    feature_matrix_dict = dict()
    node_coor_dict = dict()
    for i in range(5):#annot 标签从-1到35
        # feature shape (# of vertex number, # of feature)
        brain_region_data = feature[counter:counter+np.sum(feature == i - 1),:]
        avg_brain_region_data = np.average(brain_region_data, axis=0)
        print(avg_brain_region_data.shape)
        #使用correlation 建图
        # br_df = pd.DataFrame(brain_region_data)
        # counter += np.sum(feature == i - 1) + 1
        # feature_matrix_dict[i] = br_df.T.corr()

        #使用dgl.knn建图，明确特征计算方法
        brain_data_torch = torch.from_numpy(brain_region_data)
        # ndata = feature
        if brain_region_data.shape[0] !=0:
            knn_g = dgl.knn_graph(brain_data_torch, knn) #节点很多，应当适当增加节点数量
            # knn_g.edata['w'] = torch.tensor(edata).float()
            # print(graph.edata['w'].size())
            # knn_g.ndata['w'] = torch.tensor(ndata[0:knn_g.num_nodes(), :]).float()
            # knn_g.edata = {}
            knn_g.ndata['feat'] = torch.topk(pairwise_squared_distance(brain_data_torch), knn, 1, largest=False).values.to(torch.float32)
            # knn_g.ndata['label'] = torch.from_numpy(np.array([1 for _ in range(knn_g.num_nodes())]))

            # split_rand(labels=knn_g.num_nodes())
            # knn_g.ndata['train_mask'], knn_g.ndata['test_mask'], knn_g.ndata['val_mask'] = split_rand(labels=knn_g.num_nodes())
            pairwise_dists = torch.topk(pairwise_squared_distance(brain_data_torch).to(torch.float32), knn, 1, largest=False).values
            feature_matrix_dict[i] = knn_g
            node_coor_dict[i] = pairwise_dists
        else:
            feature_matrix_dict[i] = None
        counter += np.sum(feature == i - 1) + 1
    # return feature_matrix_dict, node_coor_dict
    return feature_matrix_dict, knn_g


# f_path = 'D:/Down/Output/subjects/sub-01' + '/ses-M00/t1/freesurfer_cross_sectional/sub-01_ses-M00'  #file root path
# lh_feature, rh_feature = loading_feature(f_path)
# knn_g = reading_brain_region(lh_feature, knn=5)
# print(knn_g)
# lh_feature_dict, lh_node_coor_dict = reading_brain_region(lh_feature, knn=5)
# rh_feature_dict, rh_node_coor_dict = reading_brain_region(rh_feature, knn=5)
# print(lh_feature_dict[1].edges(), rh_feature_dict[1].edata)

root_path = 'D:/Down/Output/subjects/sub-02'
lh_feature, rh_feature = loading_feature(root_path)
lh_feature_dict, kg = reading_brain_region(lh_feature, knn=5)