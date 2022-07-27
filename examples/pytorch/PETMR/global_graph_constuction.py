from sklearn import preprocessing
import pandas as pd
import glob
import numpy as np
import torch
from dgl.nn.pytorch import pairwise_squared_distance

def loading_global_graph(root_path):
    area_file_path = glob.glob(root_path + '/ses-M00/t1/freesurfer_cross_sectional/regional_measures/*desikan_area*')[0]
    area_data = preprocessing.scale(pd.read_csv(area_file_path, sep='\t')['label_value'][0:68])
    meancurv_file_path = glob.glob(root_path + '/ses-M00/t1/freesurfer_cross_sectional/regional_measures/*desikan_meancurv*')[0]
    meancurv_data = pd.read_csv(meancurv_file_path, sep='\t')['label_value'][0:68]
    thickness_file_path = glob.glob(root_path + '/ses-M00/t1/freesurfer_cross_sectional/regional_measures/*desikan_thickness*')[0]
    thickness_data = pd.read_csv(thickness_file_path, sep='\t')['label_value'][0:68]
    volume_file_path = glob.glob(root_path + '/ses-M00/t1/freesurfer_cross_sectional/regional_measures/*desikan_volume*')[0]
    volume_data = pd.read_csv(volume_file_path, sep='\t')['label_value'][0:68]
    pet_file_path = glob.glob(root_path + '/ses-M00/pet/surface/atlas_statistics/*desi*')[0]
    pet_data = pd.read_csv(pet_file_path, sep='\t')['mean_scalar']
    pet_data.drop([0, 1, 8, 9],inplace = True)
    pet_data = pet_data[0:68]
    global_feature = np.c_[area_data, meancurv_data, thickness_data, volume_data, pet_data]
    return global_feature


def concat_global_local_graph():
    pass

root_path = 'D:/Down/Output/subjects/sub-02'
global_feature = loading_global_graph(root_path)
global_matrix = torch.topk(pairwise_squared_distance(global_feature).to(torch.float32), 5, 1, largest=False).values