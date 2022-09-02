from __future__ import print_function
import nibabel as nib
import numpy as np
import cv2
from radiomics import featureextractor
import radiomics
import logging
radiomics.logger.setLevel(logging.ERROR)
import SimpleITK as sitk
from radiomics import firstorder, glcm, shape, glrlm, glszm, ngtdm, gldm
import numpy as np
import os
import pandas as pd
from glob import glob
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# def ReadDicomSeries(dicom_dir):
#     '''
#     dicom_dir : dicom directory path
#     '''
    
#     reader = sitk.ImageSeriesReader()

#     # Set Reader's File Names
#     dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
#     reader.SetFileNames(dicom_names)

#     # Get Image Object from Reader
#     sitk_file = reader.Execute()
    
#     return sitk_file
# ####dicom series -> nii.gz 변환####
# img_path =sorted(glob('/home/sungjoo/Radiomics_data/noncontrast2018_issuefile/*'))
# for CT_dir in img_path:
#     print(CT_dir)
#     for CT_dirs in os.listdir(CT_dir):
        
#         if not(CT_dirs.endswith('nii.gz')):
            
#             for p in os.listdir(os.path.join(CT_dir,CT_dirs)):
#                 CT_img = ReadDicomSeries(os.path.join(CT_dir,CT_dirs,p))
#                 print(CT_img)
# #                 print(os.path.join(CT_dir, 'CT_img.nii.gz'))
#                 sitk.WriteImage(CT_img, os.path.join(CT_dir, 'CT_img.nii.gz'))

def feature_extract(image, mask, config, features = ['firstorder', 'glcm', 'glszm', 'glrlm', 'ngtdm', 'shape']):
    '''
    :param image_origin: image_array (numpy array)
    :param image_mask: mask_array (numpy array)
    :return: whole features, featureVector
    '''
    settings = {}
    settings['binCount'] = 32
    settings['resampledPixelSpacing'] = (2, 2, 2)
    settings['verbose'] = True
    settings['label'] = 255
    settings['resegmentMode'] = 'sigma'
    
    if config == 'E':
        clamp = sitk.ClampImageFilter()
        clamp.SetLowerBound(-1000)
        clamp.SetUpperBound(400)
        image = clamp.Execute(image)
        settings['interpolator'] = 'sitkBSpline'
    else:
        clamp = sitk.ClampImageFilter()
        clamp.SetLowerBound(-1000)
        clamp.SetUpperBound(400)
        image = clamp.Execute(image)
        settings['interpolator'] = 'sitkLinear'
        
    
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.settings['enableCExtensions'] = True
    
    for feature in features:
        extractor.enableFeatureClassByName(feature.lower())
        
    featureVector = extractor.execute(image, mask)
    
    cols = []; feats = []
    for feature in features:
        for featureName in sorted(featureVector.keys()):
            if feature in featureName:
                cols.append(featureName)
                feats.append(featureVector[featureName])
    return feats, cols

def ReadDicomSeries(dicom_dir):
    '''
    dicom_dir : dicom directory path
    '''
    
    reader = sitk.ImageSeriesReader()

    # Set Reader's File Names
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)

    # Get Image Object from Reader
    sitk_file = reader.Execute()
    
    return sitk_file
def subdir_finder(path):
    if 'contrast' in path:
        return [path]
    else:
        return glob(path+'/*')

orglist =sorted(glob('/data1/workspace/sungjoo/data/lungcancer_KUMC_issuefile/*'))
cfg_D_df, cfg_E_df = pd.DataFrame(), pd.DataFrame()


error = []

for CT_dirs in tqdm(orglist):
    patID = os.path.basename(CT_dirs)
    print(patID)
    # contrast = 'Noncontrast' if 'noncontrast'in CT_dir else 'Contrast'
    contrast = '-'

    # Mask finding
    mask_dir = []
    for (root, dirs, files) in os.walk(CT_dirs):
        new_files =  [file for file in files if 'CT_img' not in file]
        for file in new_files:
            if file.endswith('nii.gz'):
                mask_dir.append(os.path.join(root, file))
    print(mask_dir)


    # Image finding
    for files in os.listdir(CT_dirs):
        if files == 'CT_img.nii.gz':
            CT_dcm_dir = os.path.join(CT_dirs,files)
    print(CT_dcm_dir)
    # for (root, _, files) in os.walk(CT_dirs):
    #     if len(files) > 20:
    #         CT_dcm_dir = root
    # for (root, dirs, files) in os.walk(CT_dirs):
    #     if file.endswith('CT_img.nii.gz'):
    #         CT_dcm_dir =os.path.join(root, file)
     
    #         break
    # print(CT_dcm_dir)
#     path= CT_dcm_dir+'/*'
#     img_list=[]
#     for n,dcm in enumerate(sorted(glob(path))):
# #         print(dcm)
#         img = sitk.ReadImage(dcm)
        
# #         print(img.GetSize())
#         img_arr = sitk.GetArrayFromImage(img)[0]
# #         print(img_arr.shape)
#         img_list.append(img_arr)
#     CT_img= np.stack(img_list) 
#     CT_img = sitk.GetImageFromArray(CT_img)
    
    
    
#     print(CT_img)
    
        
    # CT_img = ReadDicomSeries(CT_dcm_dir)
    

#     # Container
    finished_mask_type = []
    cfg_D_feats_total, cfg_D_cols_total = [patID, contrast], ['ID', 'Contrast']
    try:
        for mask_path in mask_dir:
            
            mask_type = mask_path.split('/')[-1].split('.')[-3]
            print(mask_type)
            if mask_type not in finished_mask_type:
                mask = sitk.ReadImage(mask_path)
                CT_img = sitk.ReadImage(CT_dcm_dir)
        #         print(mask.GetOrigin())
        #         print(mask.GetDirection())
        #         print(mask.GetSpacing())

        #         CT_img.SetOrigin(mask.GetOrigin())
        #         CT_img.SetDirection(mask.GetDirection())
        #         CT_img.SetSpacing(mask.GetSpacing())
                
        #         print(CT_img.GetOrigin())
        #         print(CT_img.GetDirection())
        #         print(CT_img.GetSpacing())
                    
                print('CT', CT_img.GetSize(), CT_img.GetSpacing())
                print('Mask', mask.GetSize(), mask.GetSpacing())
                if CT_img.GetSpacing()!=mask.GetSpacing():
                    print(mask.GetOrigin())
                    print(mask.GetDirection())
                    print(mask.GetSpacing())
                    CT_img.SetOrigin(mask.GetOrigin())
                    CT_img.SetDirection(mask.GetDirection())
                    CT_img.SetSpacing(mask.GetSpacing())

                    print(CT_img.GetOrigin())
                    print(CT_img.GetDirection())
                    print(CT_img.GetSpacing())

                if CT_img.GetSize()[2] !=mask.GetSize()[2]:
                    print(CT_img.GetSize()[2])
                    error.append(mask_path)
                    pass
                cfg_D_feats, cfg_D_cols = feature_extract(CT_img, mask, 'D')
                # cfg_E_feats, cfg_E_cols = feature_extract(CT_img, mask, 'E')
                print(cfg_D_feats)

                cfg_D_cols = [mask_type+'_mask_'+col.lstrip('original_') for col in cfg_D_cols]
                # cfg_E_cols = [mask_type+'_mask_'+col.lstrip('original_') for col in cfg_E_cols]

                cfg_D_feats_total += cfg_D_feats
                # cfg_E_feats_total += cfg_E_feats
                cfg_D_cols_total  += cfg_D_cols
                # cfg_E_cols_total  += cfg_E_cols

                finished_mask_type.append(mask_type)

    except Exception as e:
        print("Error for %s\n\n" % CT_dirs, e)

    cfg_D_row = pd.DataFrame([cfg_D_feats_total], columns=cfg_D_cols_total)
    # cfg_E_row = pd.DataFrame([cfg_E_feats_total], columns=cfg_E_cols_total)

    cfg_D_df = pd.concat([cfg_D_df, cfg_D_row])
cfg_D_df.to_csv('lung cancer_Lung Cancer_KUMC_Dicom+Mask_issue.csv', index=False)

#     # sitk.WriteImage(CT_img, os.path.join(CT_dirs, 'CT_img.nii.gz'))