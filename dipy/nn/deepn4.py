#!/usr/bin/python
"""
Class and helper functions for fitting the DeepN4 model.
"""
import logging
import numpy as np

import torch
import nibabel as nib
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter
from nilearn.image import resample_img

from dipy.data import get_fnames
from dipy.testing.decorators import doctest_skip_parser
from dipy.utils.optpkg import optional_package
from dipy.nn.utils import set_logger_level

tf, have_tf, _ = optional_package('tensorflow', min_version='2.0.0')
if have_tf:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Add
else:
    logging.warning('This model requires Tensorflow.\
                    Please install these packages using \
                    pip. If using mac, please refer to this \
                    link for installation. \
                    https://github.com/apple/tensorflow_macos')


logging.basicConfig()
logger = logging.getLogger('histo_resdnn')

class DeepN4:
    """
    This class is intended for the DeepN4 model.
    """

    @doctest_skip_parser
    def __init__(self, verbose=False):
        r"""

        To obtain the pre-trained model, use::
        >>> deepn4_model = DeepN4() # skip if not have_tf
        >>> fetch_model_weights_path = get_fnames('deepn4_weights') # skip if not have_tf
        >>> deepn4_model.load_model_weights(fetch_model_weights_path) # skip if not have_tf

        This model is designed to take as input file T1 signal and predict
        bias field. Effectively, this model is mimicking bias correction.

        Parameters
        ----------
        verbose : bool (optional)
            Whether to show information about the processing.
            Default: False

        References
        ----------
        Kanakaraj, P., Yao, T., Cai, L. Y., Lee, H. H., Newlin, N. R., 
        Kim, M. E., & Moyer, D. (2023). DeepN4: Learning N4ITK Bias Field 
        Correction for T1-weighted Images.
        """

        if not have_tf:
            raise tf()

    def fetch_default_weights(self):
        r"""
        Load the model pre-training weights to use for the fitting.
        Will not work if the declared SH_ORDER does not match the weights
        expected input.
        """
        fetch_model_weights_path = get_fnames('deepn4_weights')
        self.load_model_weights(fetch_model_weights_path)

    def load_model_weights(self, weights_path):
        r"""
        Load the custom pre-training weights to use for the fitting.
        Will not work if the declared SH_ORDER does not match the weights
        expected input.

        The weights for a sh_order of 8 can be obtained via the function:
            get_fnames('histo_resdnn_weights').

        Parameters
        ----------
        weights_path : str
            Path to the file containing the weights (hdf5, saved by tensorflow)
        """
        try:
            # self.model = tf.saved_model.load(checkpoint_file)
            # checkpoint_file = https://drive.google.com/drive/folders/1mdBsV0kHRRV_Alu1QJrTT7N0GGNJDuiu 
            # folder checkpoint_epoch_264.pd
            self.model.load_weights(weights_path)
        except ValueError:
            raise ValueError('Expected input for the provided model weights '
                             'do not match the declared model ({})'
                             .format(self.sh_size))

    def __predict(self, x_test):
        r"""
        Internal prediction function
        Predict bias field from input T1 signal 

        Parameters
        ----------
        x_test : np.ndarray (1, 1, 128, 128, 128)
            Image should match the required shape of the model.

        Returns
        -------
        np.ndarray (1, 1, 128, 128, 128)
            Predicted bias field
        """

        return self.model.predict(input_1=x_test)
    
    def pad(self, img, sz):

        tmp = np.zeros((sz, sz, sz))

        diff = int((sz-img.shape[0])/2)
        lx = max(diff,0)
        lX = min(img.shape[0]+diff,sz)

        diff = (img.shape[0]-sz) / 2
        rx = max(int(np.floor(diff)),0)
        rX = min(img.shape[0]-int(np.ceil(diff)),img.shape[0])

        diff = int((sz - img.shape[1]) / 2)
        ly = max(diff, 0)
        lY = min(img.shape[1] + diff, sz)

        diff = (img.shape[1] - sz) / 2
        ry = max(int(np.floor(diff)), 0)
        rY = min(img.shape[1] - int(np.ceil(diff)), img.shape[1])

        diff = int((sz - img.shape[2]) / 2)
        lz = max(diff, 0)
        lZ = min(img.shape[2] + diff, sz)

        diff = (img.shape[2] - sz) / 2
        rz = max(int(np.floor(diff)), 0)
        rZ = min(img.shape[2] - int(np.ceil(diff)), img.shape[2])

        tmp[lx:lX,ly:lY,lz:lZ] = img[rx:rX,ry:rY,rz:rZ]
        
        return tmp, [lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ]

    def normalize_img(self, img, max_img, min_img, a_max, a_min):

        img = (img - min_img)/(max_img - min_img)
        img = np.clip(img, a_max=a_max, a_min=a_min)

        return img
    
    def load_resample(self, subj ):

        input_data, [lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ] = self.pad(subj.get_fdata(), 128)
        in_max = np.percentile(input_data[np.nonzero(input_data)], 99.99)
        input_data = self.normalize_img(input_data, in_max, 0, 1, 0)
        input_data = np.squeeze(input_data)
        input_vols = np.zeros((1,1, 128, 128, 128))
        input_vols[0,0,:,:,:] = input_data

        return torch.from_numpy(input_vols).float(), lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ, in_max

    def predict(self, input_file):
        """ Wrapper function to facilitate prediction of larger dataset.
        The function will mask, normalize, split, predict and 're-assemble'
        the data as a volume.

        Parameters
        ----------
        input_file : string
            Path to the T1 scan

        Returns
        -------
        final_corrected : np.ndarray (x, y, z)
            Predicted bias corrected image. The volume has matching shape to the input
            data

        """
        # Preprocess input data (resample, normalize, and pad)
        new_voxel_size = [2, 2, 2]  
        resampled_T1 = resample_img(input_file, target_affine=np.diag(new_voxel_size))
        in_features, lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ, in_max = self.load_resample(resampled_T1)

        # Load the model 
        # model = tf.saved_model.load(checkpoint_file)

        # Run the model to get the bias field
        # logfield = model(input_1=Variable(in_features))
        logfield = self.__predict(Variable(in_features))
        field = np.exp(logfield['120'])
        field = field.squeeze()

        # Postprocess predicted field (reshape - unpad, smooth the field, upsample)
        org_data = resampled_T1.get_fdata()
        final_field = np.zeros([org_data.shape[0], org_data.shape[1], org_data.shape[2]])
        final_field[rx:rX,ry:rY,rz:rZ] = field[lx:lX,ly:lY,lz:lZ]
        final_fields = gaussian_filter(final_field, sigma=3)
        ref = nib.load(input_file)
        upsample_final_field = resample_img(nib.Nifti1Image(final_fields,resampled_T1.affine), target_affine=ref.affine, target_shape=ref.shape)

        # Correct the image
        upsample_data = upsample_final_field.get_fdata()
        ref_data = ref.get_fdata()
        with np.errstate(divide='ignore', invalid='ignore'):
            final_corrected = np.where(upsample_data != 0, ref_data / upsample_data, 0)

        return final_corrected

