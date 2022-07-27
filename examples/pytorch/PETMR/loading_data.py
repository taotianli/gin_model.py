import nibabel as nib
import glob
import pandas as pd
import numpy as np


def reading_mgh(root_path):
    '''

    Args:
        root_path:

    Returns:

    '''
    lhm_path = glob.glob(root_path + '/ses-M00/pet/surface/*hemi-lh_projection.mgh')[0]
    # print(lhm_path)
    lhm_data = nib.load(lhm_path)
    rhm_path = glob.glob(root_path + '/ses-M00/pet/surface/*hemi-rh_projection.mgh')[0]
    rhm_data = nib.load(rhm_path)
    # print(rhm_path)
    raw_lh_data = lhm_data.get_fdata(dtype="float32")
    raw_rh_data = rhm_data.get_fdata(dtype="float32")
    return raw_lh_data[:, 0, 0], raw_rh_data[:, 0, 0]


def loading_feature(root_path):

    """
    left surface feature:
        lh_white = nib.freesurfer.io.read_morph_data(f_path + '/surf/lh.white')
        lh_sphere = nib.freesurfer.io.read_morph_data(f_path + '/surf/lh.sphere')
        lh_smoothwm = nib.freesurfer.io.read_morph_data(f_path + '/surf/lh.smoothwm')
        lh_inflated = nib.freesurfer.io.read_morph_data(f_path + '/surf/lh.inflated')
    """

    f_path = glob.glob(root_path + '/ses-M00/t1/freesurfer_cross_sectional/sub*')[0]
    print('feature path = ', f_path)
    lh_pet_data, rh_pet_data = reading_mgh(root_path)
    lh_geo = nib.freesurfer.io.read_geometry(f_path + '/surf/lh.pial', read_metadata=True, read_stamp=True)
    lh_annot = nib.freesurfer.io.read_annot(f_path + '/label/lh.aparc.DKTatlas.annot',orig_ids=False)
    lh_volume = nib.freesurfer.io.read_morph_data(f_path + '/surf/lh.volume')
    lh_thickness = nib.freesurfer.io.read_morph_data(f_path + '/surf/lh.thickness')
    lh_sulc = nib.freesurfer.io.read_morph_data(f_path + '/surf/lh.sulc')
    lh_curv = nib.freesurfer.io.read_morph_data(f_path + '/surf/lh.curv')
    lh_jacobian_white = nib.freesurfer.io.read_morph_data(f_path + '/surf/lh.jacobian_white')
    lh_area = nib.freesurfer.io.read_morph_data(f_path + '/surf/lh.area')
    lh_avg_curv = nib.freesurfer.io.read_morph_data(f_path + '/surf/lh.avg_curv')
    lh_total_feature = np.c_[lh_geo[0], lh_annot[0], lh_volume.T, lh_thickness.T, lh_sulc.T, lh_curv.T, lh_jacobian_white.T, lh_area.T, lh_avg_curv.T, lh_pet_data]
    # print(lh_volume.T.shape, lh_pet_data.shape)
    """
    right surface feature
        rh_white = nib.freesurfer.io.read_morph_data(f_path + '/surf/rh.white')
        rh_sphere = nib.freesurfer.io.read_morph_data(f_path + '/surf/rh.sphere')
        rh_smoothwm = nib.freesurfer.io.read_morph_data(f_path + '/surf/rh.smoothwm')
        rh_inflated = nib.freesurfer.io.read_morph_data(f_path + '/surf/rh.inflated')
    """
    rh_geo = nib.freesurfer.io.read_geometry(f_path + '/surf/rh.pial', read_metadata=True, read_stamp=True)
    rh_annot = nib.freesurfer.io.read_annot(f_path + '/label/rh.aparc.DKTatlas.annot',orig_ids=False)
    rh_volume = nib.freesurfer.io.read_morph_data(f_path + '/surf/rh.volume')
    rh_thickness = nib.freesurfer.io.read_morph_data(f_path + '/surf/rh.thickness')
    rh_sulc = nib.freesurfer.io.read_morph_data(f_path + '/surf/rh.sulc')
    rh_curv = nib.freesurfer.io.read_morph_data(f_path + '/surf/rh.curv')
    rh_jacobian_white = nib.freesurfer.io.read_morph_data(f_path + '/surf/rh.jacobian_white')
    rh_area = nib.freesurfer.io.read_morph_data(f_path + '/surf/rh.area')
    rh_avg_curv = nib.freesurfer.io.read_morph_data(f_path + '/surf/rh.avg_curv')
    rh_total_feature = np.c_[rh_geo[0], rh_annot[0], rh_volume.T, rh_thickness.T, rh_sulc.T, rh_curv.T, rh_jacobian_white.T, rh_area.T, rh_avg_curv.T, rh_pet_data]

    return lh_total_feature, rh_total_feature


def loading_label(xlsx_path):
    sub_df = pd.read_excel(xlsx_path, engine='openpyxl')  #读取病历信息
    sub_list = pd.read_excel(xlsx_path, engine='openpyxl')
    sub_info = dict(zip(sub_df['ID'], sub_df['Label']))
    return sub_info


# for sub_path in glob.glob('D:/Down/Output/subjects/*'):
#     # print(sub_path)
#     lh_feature, rh_feature = loading_feature(sub_path)
#     # sub = loading_label(xlsx_path)
#     lh_data, rh_data = reading_mgh(sub_path)

# root_path = 'D:/Down/Output/subjects/sub-02'
# # f_path = root_path + '/ses-M00/t1/freesurfer_cross_sectional/sub-01_ses-M00'  #file root path
# # xlsx_path = 'D:/Down/PET_data/Diagnosis_Information.xlsx' #AD & MCI label file path
# lh_feature, rh_feature = loading_feature(root_path)
# # sub = loading_label(xlsx_path)
# lh_data, rh_data = reading_mgh(root_path)

root_path = 'D:/Down/Output/subjects/sub-02'
lh_feature, rh_feature = loading_feature(root_path)