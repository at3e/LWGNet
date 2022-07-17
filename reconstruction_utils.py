#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:17:40 2020

@author: at3ee
"""
import os
import datetime
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
from utils import alpha_blending

today = datetime.datetime.now()

class Base_configuration:
    
    def __init__(self, device, root_dir, data_config, model_config, model_params, model):
        
        self.device = device
        self.model_config = model_config
        self.model_params = model_params
        self.model = model.to(self.device)
        self.data_config = data_config
        self.root_dir = root_dir
        self.checkpoint_file = self.model_config['checkpoint_filename']
       
        self._resume_checkpoint()
        
    def _resume_checkpoint(self):
        model_path = os.path.join(self.root_dir, self.checkpoint_file)
        print(model_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict=False)

        print("Model checkpoint loaded.")

    
    def _set_model_to_eval_mode(self):
         self.model.eval()
         
    def _reconstruct(self):        
        self._set_model_to_eval_mode()
        self._test_epoch()

class test_reconstruction(Base_configuration):
    
    def __init__(
            self,
            device,
            root_dir,
            data_config, 
            model_config,
            model_params,
            model
    ):
        
        super(test_reconstruction, self).__init__(device, root_dir, data_config,
                                                  model_config, model_params, model)

    @torch.no_grad()
    def _test_epoch(self): 
        test_data_dir = self.data_config["test_dir"]
        test_filenames = [line.rstrip('\n') for line in open(
            os.path.join(self.root_dir, self.data_config["test_filenames"]), "r")]
        idx = sio.loadmat(os.path.join(self.root_dir, self.model_params["led_idx"]))["idx_led"] - 1
        Ns = np.squeeze(sio.loadmat(os.path.join(self.root_dir, self.model_params["K_uv"]))["Ns"])

        Y = []
        for i in range(len(test_filenames)):
            file_path = os.path.join(self.root_dir, test_data_dir, test_filenames[i])
            x = np.array(sio.loadmat(file_path)["X"])
            x = x[idx,...].squeeze()
            x = (x-np.min(x))/(np.max(x)-np.min(x))
            x_ = torch.tensor(x, dtype=torch.float64, device=self.device).unsqueeze(0)
            y = self.model(x_, torch.tensor(np.expand_dims(Ns, axis=0)))
            Y.append(y.cpu().numpy())
        Y = np.array(Y)
        sio.savemat(self.data_config["test_reconstruction_path"], {"O":Y})
        
        print("Done!")

 
def reconstructFoV(O, n_p=64, n_width=256, n_height=256, n_ovlap=96, n_row=8, n_col=8):
    '''

    Parameters
    ----------
    O : numpy.complex128
        stack of reconstructed object-field.
    n_p : numpy.int, optional
        #input patches. The default is 64.
    n_width : numpy.int , optional
        width of input patch. The default is 256.
    n_height : numpy.int, optional
        height of input patch. The default is 256.
    n_ovlap : numpy.int, optional
        #pixels overlap between patches. The default is 96.
    n_row :  numpy.int, optional
        #rows in FoV. The default is 8.
    n_col :  numpy.int, optional
        #columns in FoV. The default is 8.

    Returns
    -------
    Recinstructed FoV phase.

    '''
    X = np.angle(O)
    H = n_row*n_height-(n_row-1)*n_ovlap
    W = n_col*n_width-(n_col-1)*n_ovlap
    

    # Concatenate row-wise
    
    H0 = X[0:n_p:n_col, :, :]
    C0 = np.zeros((n_col, n_width, W))
    for k in range(n_row):
        I01 = H0[k, :, :]
        for l in range(1, n_col):
            I01 = alpha_blending(I01, X[k*n_col+l, :, :], n_ovlap, 'horizontal')
        C0[k, ...] = I01
        plt.imshow(I01)

    # Concatenate col-wise
    
    I02 = C0[0, ...]
    for k in range(1, n_row):
        plt.imshow(C0[k, :, :])
        I02 = alpha_blending(I02, C0[k, :, :], n_ovlap, 'vertical')

    P = I02
    P = (P-np.min(P))/(np.max(P)-np.min(P))

    return P
