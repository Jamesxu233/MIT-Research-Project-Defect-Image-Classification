# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import sys
import glob

import cv2
# from tqdm import tqdm
import sod_metrics as M
sys.path.append('../')
# print(sys.path)

testnumber=80


######DUTLF
FM = M.Fmeasure()
WFM = M.WeightedFmeasure()
SM = M.Smeasure()
EM = M.Emeasure()
MAE = M.MAE()
        # data_root = './test_data'
        # mask_root = os.path.join(data_root, 'masks')
        # pred_root = os.path.join(data_root, 'preds')
        # mask_root = 'D:\lvchengtao\dataset\saliency detection\Light-Field_dataset\\test_data\\test_masks'
        # pred_root =  'D:\lvchengtao\dataset\saliency detection\data\Light-Field_dataset\er\sal_Teacher\DUT'
mask_name_list = glob.glob('E:/data/transistor/ground_truth/bent_lead/*.png')
pred_name_list= glob.glob('E:/data/result/*.png')
        # for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
for i in range(len(mask_name_list)):
    mask_path = mask_name_list[i]
    pred_path = pred_name_list[i]
            #print(pred_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    FM.step(pred=pred, gt=mask)
    WFM.step(pred=pred, gt=mask)
    SM.step(pred=pred, gt=mask)
    EM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)

    fm = FM.get_results()['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']

        #print('pre path is', pred_root, '     ', i, 'epoch')
    print(
        'MAE:', mae.round(4), '; ',
        'adpFm:', fm['adp'].round(4), '; ',
        'Smeasure:', sm.round(4), '; ',
        'wFmeasure:', wfm.round(4), '; ',
        'adpEm:', em['adp'].round(4), '; ',
            # 'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(4), '; ',
            # 'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(4), '; ',
        'meanFm:', fm['curve'].mean().round(4), '; ',
            # 'maxFm:', fm['curve'].max().round(4),
        sep=''
    )

