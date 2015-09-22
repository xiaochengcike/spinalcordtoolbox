#!/usr/bin/env python
########################################################################################################################
#
# Asman et al. groupwise multi-atlas segmentation method implementation, with a lot of changes
#
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux, Sara Dupont
# Modified: 2015-05-19
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

# TODO change 'target' by 'input'
# TODO : make it faster

# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt

from msct_pca import PCA
# from msct_image import Image
# from msct_parser import *
from msct_gmseg_utils import *
import sct_utils as sct
import pickle, gzip
import commands
from math import exp


class SegmentationParam:
    def __init__(self):
        status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

        self.debug = 0
        self.path_model = path_sct+'/data/gm_model' # None  # '/Volumes/folder_shared/greymattersegmentation/data_asman/dictionary'
        self.todo_model = 'load'  # 'compute'
        self.new_model_dir = './gm_model'
        self.output_name = ''
        self.reg = ['Affine']  # default is Affine  TODO : REMOVE THAT PARAM WHEN REGISTRATION IS OPTIMIZED
        self.reg_metric = 'MI'
        self.target_denoising = True
        self.target_normalization = True
        self.target_means = None
        self.first_reg = False
        self.use_levels = True
        self.weight_gamma = 2.5
        self.equation_id = 1
        self.weight_label_fusion = False
        self.mode_weight_similarity = False
        self.z_regularisation = False
        self.res_type = 'prob'
        self.dev = False
        self.verbose = 1

    def __repr__(self):
        s = ''
        s += 'path_model: ' + str(self.path_model) + '\n'
        s += 'todo_model: ' + str(self.todo_model) + '\n'
        s += 'new_model_dir: ' + str(self.new_model_dir) + '  *** only used if todo_model=compute ***\n'
        s += 'output_name: ' + str(self.output_name) + '\n'
        s += 'reg: ' + str(self.reg) + '\n'
        s += 'reg_metric: ' + str(self.reg_metric) + '\n'
        s += 'target_denoising: ' + str(self.target_denoising) + ' ***WARNING: used in sct_segment_gray_matter not in msct_multiatlas_seg***\n'
        s += 'target_normalization: ' + str(self.target_normalization) + '\n'
        s += 'target_means: ' + str(self.target_means) + '\n'
        s += 'first_reg: ' + str(self.first_reg) + '\n'
        s += 'use_levels: ' + str(self.use_levels) + '\n'
        s += 'weight_gamma: ' + str(self.weight_gamma) + '\n'
        s += 'equation_id: ' + str(self.equation_id) + '\n'
        s += 'weight_label_fusion: ' + str(self.weight_label_fusion) + '\n'
        s += 'mode_weight_similarity: ' + str(self.mode_weight_similarity) + '\n'
        s += 'z_regularisation: ' + str(self.z_regularisation) + '\n'
        s += 'res_type: ' + str(self.res_type) + '\n'
        s += 'verbose: ' + str(self.verbose) + '\n'

        return s


########################################################################################################################
# ----------------------------------------------------- Classes ------------------------------------------------------ #
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# MODEL DICTIONARY SLICE BY SLICE---------------------------------------------------------------------------------------
class ModelDictionary:
    """
    Dictionary used by the supervised gray matter segmentation method
    """
    def __init__(self, dic_param=None):
        """
        model dictionary constructor

        :param dic_param: dictionary parameters, type: Param
        """
        if dic_param is None:
            self.param = SegmentationParam()
        else:
            self.param = dic_param

        self.level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}

        # Initialisation of the parameters
        self.coregistration_transfos = None
        self.slices = None
        self.J = None
        self.N = None
        self.mean_seg = None
        self.mean_image = None

        # list of transformation to apply to each slice to co-register the data into the common groupwise space
        self.coregistration_transfos = self.param.reg

        if self.param.todo_model == 'compute':
            self.compute_model()
        elif self.param.todo_model == 'load':
            self.load_model()
        # self.extract_metric_from_dic(gm_percentile=0, wm_percentile=0, save=True)
        # self.mean_seg_by_level(save=True)

    # ------------------------------------------------------------------------------------------------------------------
    # FUNCTIONS USED TO COMPUTE THE MODEL
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def compute_model(self):

        sct.printv('\nComputing the model dictionary ...', self.param.verbose, 'normal')
        sct.run('mkdir ' + self.param.new_model_dir)
        param_fic = open(self.param.new_model_dir + '/info.txt', 'w')
        param_fic.write(str(self.param))
        param_fic.close()

        sct.printv('\nLoading data dictionary ...', self.param.verbose, 'normal')
        # List of T2star images (im) and their label decision (gmseg) (=segmentation of the gray matter), slice by slice
        self.slices = self.load_data_dictionary()  # type: list of slices
        self.mean_image = np.mean([dic_slice.im for dic_slice in self.slices], axis=0)
        # number of slices in the data set
        self.J = len([dic_slice.im for dic_slice in self.slices])  # type: int
        # dimension of the slices (flattened)
        self.N = len(self.slices[0].im.flatten())  # type: int

        # inverts the segmentation slices : the model uses segmentation of the WM instead of segmentation of the GM
        self.invert_seg()

        sct.printv('\nComputing the transformation to co-register all the data into a common groupwise space (using the white matter segmentations) ...', self.param.verbose, 'normal')
        # mean segmentation image of the dictionary, type: numpy array
        self.mean_seg = self.seg_coregistration(transfo_to_apply=self.coregistration_transfos)

        sct.printv('\nCo-registering all the data into the common groupwise space ...', self.param.verbose, 'normal')
        self.coregister_data(transfo_to_apply=self.coregistration_transfos)

        # update the mean image
        self.mean_image = np.mean([dic_slice.im_M for dic_slice in self.slices], axis=0) # type: numpy array

        self.save_model()

    # ------------------------------------------------------------------------------------------------------------------
    def load_data_dictionary(self):
        """
        each slice of each subject will be loaded separately in a Slice object containing :

        - a slice id

        - the original T2star image crop around the spinal cord: im

        - a manual segmentation of the gray matter: seg

        :return slices: numpy array of all the slices of the data dictionary
        """
        # initialization
        slices = []
        j = 0
        for subject_dir in os.listdir(self.param.path_model):
            subject_path = self.param.path_model + '/' + subject_dir
            if os.path.isdir(subject_path):
                for file_name in os.listdir(subject_path):
                    if 'im' in file_name:  # or 'seg_in' in file_name:
                        slice_level = 0
                        name_list = file_name.split('_')
                        for word in name_list:
                            if word.upper() in self.level_label.values():
                                slice_level = get_key_from_val(self.level_label, word.upper())

                        slices.append(Slice(slice_id=j, im=Image(subject_path + '/' + file_name).data, level=slice_level, reg_to_m=[]))

                        seg_file = sct.extract_fname(file_name)[1][:-3] + '_seg.nii.gz'
                        slices[j].set(gm_seg=Image(subject_path + '/' + seg_file).data)
                        j += 1

        return np.asarray(slices)

    # ------------------------------------------------------------------------------------------------------------------
    def invert_seg(self):
        """
        Invert the gray matter segmentation to get segmentation of the white matter instead
        keeps more information, better results
        """
        for dic_slice in self.slices:
            inverted_slice_decision = inverse_gmseg_to_wmseg(dic_slice.gm_seg, dic_slice.im, save=False)
            dic_slice.set(wm_seg=inverted_slice_decision)

    # ------------------------------------------------------------------------------------------------------------------
    def seg_coregistration(self, transfo_to_apply=None):
        """
        For all the segmentation slices, do a registration of the segmentation slice to the mean segmentation
         applying all the transformations in transfo_to_apply

        Compute, apply and save each transformation warping field for all the segmentation slices

        Compute the new mean segmentation at each step and update self.mean_seg

        :param transfo_to_apply: list of string
        :return resulting_mean_seg:
        """

        current_mean_seg = compute_majority_vote_mean_seg(np.asarray([dic_slice.wm_seg for dic_slice in self.slices]))
        first = True
        for transfo in transfo_to_apply:
            sct.printv('Doing a ' + transfo + ' registration of each segmentation slice to the mean segmentation ...', self.param.verbose, 'normal')
            current_mean_seg = self.find_coregistration(mean_seg=current_mean_seg, transfo_type=transfo, first=first)
            first = False

        resulting_mean_seg = current_mean_seg

        return resulting_mean_seg

    # ------------------------------------------------------------------------------------------------------------------
    def find_coregistration(self, mean_seg=None, transfo_type='Affine', first=True):
        """
        For each segmentation slice, apply and save a registration of the specified type of transformation
        the name of the registration file (usually a matlab matrix) is saved in self.RtoM

        :param mean_seg: current mean segmentation

        :param transfo_type: type of transformation for the registration

        :return mean seg: updated mean segmentation
        """

        # Coregistration of the white matter segmentations
        for dic_slice in self.slices:
            name_j_transform = 'transform_slice_' + str(dic_slice.id) + find_ants_transfo_name(transfo_type)[0]
            new_reg_list = dic_slice.reg_to_M.append(name_j_transform)
            dic_slice.set(reg_to_m=new_reg_list)

            if first:
                seg_m = apply_ants_transfo(mean_seg, dic_slice.wm_seg,  transfo_name=name_j_transform, path=self.param.new_model_dir + '/', transfo_type=transfo_type, metric=self.param.reg_metric)
            else:
                seg_m = apply_ants_transfo(mean_seg, dic_slice.wm_seg_M,  transfo_name=name_j_transform, path=self.param.new_model_dir + '/', transfo_type=transfo_type, metric=self.param.reg_metric)
            dic_slice.set(wm_seg_m=seg_m.astype(int))
            dic_slice.set(wm_seg_m_flat=seg_m.flatten().astype(int))

        mean_seg = compute_majority_vote_mean_seg([dic_slice.wm_seg_M for dic_slice in self.slices])

        return mean_seg

    # ------------------------------------------------------------------------------------------------------------------
    def coregister_data(self,  transfo_to_apply=None):
        """
        Apply to each image slice of the dictionary the transformations found registering the segmentation slices.
        The co_registered images are saved for each slice as im_M

        Delete the directories containing the transformation matrix : not needed after the coregistration of the data.

        :param transfo_to_apply: list of string
        :return:
        """
        list_gm_seg = [dic_slice.gm_seg for dic_slice in self.slices]
        mean_gm_seg = compute_majority_vote_mean_seg(list_gm_seg)

        for dic_slice in self.slices:
            for n_transfo, transfo in enumerate(transfo_to_apply):
                im_m = apply_ants_transfo(self.mean_image, dic_slice.im, search_reg=False, transfo_name=dic_slice.reg_to_M[n_transfo], binary=False, path=self.param.new_model_dir+'/', transfo_type=transfo, metric=self.param.reg_metric)
                gm_seg_m = apply_ants_transfo(mean_gm_seg, dic_slice.gm_seg, search_reg=False, transfo_name=dic_slice.reg_to_M[n_transfo], binary=True, path=self.param.new_model_dir+'/', transfo_type=transfo, metric=self.param.reg_metric)
                # apply_2D_rigid_transformation(self.im[j], self.RM[j]['tx'], self.RM[j]['ty'], self.RM[j]['theta'])

            dic_slice.set(im_m=im_m)
            dic_slice.set(gm_seg_m=gm_seg_m)
            dic_slice.set(im_m_flat=im_m.flatten())

        # Delete the directory containing the transformations : They are not needed anymore
        for transfo_type in transfo_to_apply:
            transfo_dir = transfo_type.lower() + '_transformations'
            if transfo_dir in os.listdir(self.param.new_model_dir + '/'):
                sct.run('rm -rf ' + self.param.new_model_dir + '/' + transfo_dir + '/')

    # ------------------------------------------------------------------------------------------------------------------
    def save_model(self):
        model_slices = np.asarray([(dic_slice.im_M, dic_slice.wm_seg_M, dic_slice.gm_seg_M, dic_slice.level) for dic_slice in self.slices])
        pickle.dump(model_slices, gzip.open(self.param.new_model_dir + '/dictionary_slices.pklz', 'wb'), protocol=2)

    # ------------------------------------------------------------------------------------------------------------------
    # END OF FUNCTIONS USED TO COMPUTE THE MODEL
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def load_model(self):

        model_slices = pickle.load(gzip.open(self.param.path_model + '/dictionary_slices.pklz', 'rb'))

        self.slices = [Slice(slice_id=i_slice, level=dic_slice[3], im_m=dic_slice[0], wm_seg_m=dic_slice[1], gm_seg_m=dic_slice[2], im_m_flat=dic_slice[0].flatten(),  wm_seg_m_flat=dic_slice[1].flatten()) for i_slice, dic_slice in enumerate(model_slices)]  # type: list of slices

        # number of slices in the data set
        self.J = len([dic_slice.im_M for dic_slice in self.slices])  # type: int
        # dimension of the slices (flattened)
        self.N = len(self.slices[0].im_M_flat)  # type: int

        self.mean_seg = compute_majority_vote_mean_seg([dic_slice.wm_seg_M for dic_slice in self.slices])

    # ------------------------------------------------------------------------------------------------------------------
    def mean_seg_by_level(self, type='binary', save=False):
        gm_seg_by_level = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': [], '': []}
        im_by_level = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': [], '': []}
        for dic_slice in self.slices:
            gm_seg_by_level[self.level_label[dic_slice.level]].append(dic_slice.gm_seg_M)
            im_by_level[self.level_label[dic_slice.level]].append(dic_slice.im_M)
        seg_averages = {}
        im_averages = {}
        for level, seg_data_set in gm_seg_by_level.items():
            seg_averages[level] = compute_majority_vote_mean_seg(seg_data_set=seg_data_set, type=type)
        seg_averages[''] = compute_majority_vote_mean_seg(seg_data_set=[dic_slice.gm_seg_M for dic_slice in self.slices], type=type)
        for level, im_data_set in im_by_level.items():
            im_averages[level] = np.mean(im_data_set, axis=0)
        im_averages[''] = np.mean([dic_slice.im_M for dic_slice in self.slices], axis=0)
        if save:
            for level, mean_gm_seg in seg_averages.items():
                Image(param=mean_gm_seg, absolutepath='./mean_seg_' + level + '.nii.gz').save()
            for level, mean_im in im_averages.items():
                Image(param=mean_im, absolutepath='./mean_im_' + level + '.nii.gz').save()
        return seg_averages, im_averages

    # ------------------------------------------------------------------------------------------------------------------
    def show_dictionary_data(self):
        """
        show the 10 first slices of the model dictionary
        """
        for dic_slice in self.slices[:10]:
            fig = plt.figure()

            if dic_slice.wm_seg is not None:
                seg_subplot = fig.add_subplot(2, 3, 1)
                seg_subplot.set_title('Original space - seg')
                im_seg = seg_subplot.imshow(dic_slice.wm_seg)
                im_seg.set_interpolation('nearest')
                im_seg.set_cmap('gray')

            seg_m_subplot = fig.add_subplot(2, 3, 2)
            seg_m_subplot.set_title('Common groupwise space - seg')
            im_seg_m = seg_m_subplot.imshow(dic_slice.wm_seg_M)
            im_seg_m.set_interpolation('nearest')
            im_seg_m.set_cmap('gray')

            if self.mean_seg is not None:
                mean_seg_subplot = fig.add_subplot(2, 3, 3)
                mean_seg_subplot.set_title('Mean seg')
                im_mean_seg = mean_seg_subplot.imshow(np.asarray(self.mean_seg))
                im_mean_seg.set_interpolation('nearest')
                im_mean_seg.set_cmap('gray')

            if dic_slice.im is not None:
                slice_im_subplot = fig.add_subplot(2, 3, 4)
                slice_im_subplot.set_title('Original space - data ')
                im_slice_im = slice_im_subplot.imshow(dic_slice.im)
                im_slice_im.set_interpolation('nearest')
                im_slice_im.set_cmap('gray')

            slice_im_m_subplot = fig.add_subplot(2, 3, 5)
            slice_im_m_subplot.set_title('Common groupwise space - data ')
            im_slice_im_m = slice_im_m_subplot.imshow(dic_slice.im_M)
            im_slice_im_m.set_interpolation('nearest')
            im_slice_im_m.set_cmap('gray')

            plt.suptitle('Slice ' + str(dic_slice.id))
            plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# MODEL ---------------------------------------------------------------------------------------------------------------
class Model:
    """
    Model used by the supervised gray matter segmentation method

    """
    def __init__(self, model_param=None, k=0.8):
        """
        Model constructor

        :param model_param: model parameters, type: Param

        :param k: Amount of variability to keep in the PCA reduced space, type: float
        """
        if model_param is None:
            self.param = SegmentationParam()
        else:
            self.param = model_param

        # Model dictionary
        self.dictionary = ModelDictionary(dic_param=self.param)

        sct.printv("The shape of the dictionary used for the PCA is (" + str(self.dictionary.N) + "," + str(self.dictionary.J) + ")", verbose=self.param.verbose)
        # Instantiate a PCA object given the dictionary just build
        if self.param.todo_model == 'compute':
            sct.printv('\nCreating a reduced common space (using a PCA) ...', self.param.verbose, 'normal')
            self.pca = PCA(np.asarray(self.dictionary.slices), k=k)
            self.pca.save_data(self.param.new_model_dir)
        elif self.param.todo_model == 'load':
            sct.printv('\nLoading a reduced common space (using a PCA) ...', self.param.verbose, 'normal')
            pca_data = pickle.load(gzip.open(self.param.path_model + '/pca_data.pklz', 'rb'))
            self.pca = PCA(np.asarray(self.dictionary.slices), mean_vect=pca_data[0], eig_pairs=pca_data[1], k=k)

        # updating the dictionary mean_image
        self.dictionary.mean_image = self.pca.mean_image

        # Other model parameters
        self.epsilon = round(1.0/self.dictionary.J, 4)/2

        if self.param.todo_model == 'compute':
            self.tau = self.compute_tau()
            pickle.dump(self.tau, open(self.param.new_model_dir + '/tau_levels_'+str(self.param.use_levels)+'.txt', 'w'), protocol=0)  # or protocol=2 and 'wb'
        elif self.param.todo_model == 'load':
            self.tau = pickle.load(open(self.param.path_model + '/tau_levels_'+str(self.param.use_levels)+'.txt', 'r'))  # if protocol was 2 : 'rb'

        if self.param.verbose == 2:
            self.pca.plot_projected_dic()

    # ------------------------------------------------------------------------------------------------------------------
    def compute_beta(self, coord_target, target_levels=None, dataset_coord=None, dataset_levels=None, tau=0.006):
        """
        Compute the model similarity (beta) between each model slice and each target image slice

        beta_j = (1/Z)exp(-tau*square_norm(target_coordinate - slice_j_coordinate))

        Z is the partition function that enforces the constraint that sum(beta)=1

        :param coord_target: coordinates of the target image in the reduced model space

        :param tau: weighting parameter indicating the decay constant associated with a geodesic distance
        between a given dictionary slice and a projected target image slice

        :return:
        """
        if dataset_coord is None:
            # in the dataset_coord matrix, each column correspond to the projection of one of the original data image,
            # the transpose operator .T enable the loop to iterate over all the images coord
            dataset_coord = self.pca.dataset_coord.T
            dataset_levels = [dic_slice.level for dic_slice in self.dictionary.slices]

        beta = []
        if self.param.mode_weight_similarity:
            mode_weight = [val/sum(self.pca.kept_eigenval) for val in self.pca.kept_eigenval]
            # TODO: WARNING: see if the weights shouldnt be inversed: a bigger weight for the first modes will make the distances along those modes bigger: maybe we want to do the opposite
        else:
            mode_weight = None

        # 3D TARGET
        if isinstance(coord_target[0], (list, np.ndarray)):
            for i_target, coord_projected_slice in enumerate(coord_target):
                beta_slice = []
                for j_slice, coord_slice_j in enumerate(dataset_coord):
                    if mode_weight is None:
                        square_norm = np.linalg.norm((coord_projected_slice - coord_slice_j), 2)
                    else:
                        from scipy.spatial.distance import wminkowski
                        square_norm = wminkowski(coord_projected_slice, coord_slice_j, 2, mode_weight)

                    if target_levels is not None and target_levels is not [None] and self.param.use_levels:
                        if self.param.equation_id == 1:
                            # EQUATION #1 (better results ==> kept)
                            beta_slice.append(exp(-self.param.weight_gamma*abs(target_levels[i_target] - dataset_levels[j_slice]))*exp(-tau*square_norm))  # TODO: before = no absolute
                        elif self.param.equation_id == 2:
                            # EQUATION #2
                            if target_levels[i_target] == dataset_levels[j_slice]:
                                beta_slice.append(exp(tau*square_norm))
                            else:
                                beta_slice.append(exp(-tau*square_norm)/self.param.weight_gamma*abs(target_levels[i_target] - dataset_levels[j_slice])) #TODO: before = no absolute

                    else:
                        beta_slice.append(exp(-tau*square_norm))

                try:
                    beta_slice /= np.sum(beta_slice)
                except ZeroDivisionError:
                    sct.printv('WARNING : similarities are null', self.param.verbose, 'warning')
                    print beta_slice

                beta.append(beta_slice)

        # 2D TARGET
        else:
            for j_slice, coord_slice_j in enumerate(dataset_coord):
                if mode_weight is None:
                    square_norm = np.linalg.norm((coord_target - coord_slice_j), 2)
                else:
                    from scipy.spatial.distance import wminkowski
                    square_norm = wminkowski(coord_target, coord_slice_j, 2, mode_weight)
                if target_levels is not None and self.param.use_levels:
                    if self.param.equation_id == 1:
                        # EQUATION #1 (better results ==> kept)
                        beta.append(exp(-self.param.weight_gamma*abs(target_levels - dataset_levels[j_slice]))*exp(-tau*square_norm) )#TODO: before = no absolute
                    elif self.param.equation_id == 2:
                        # EQUATION #2
                        if target_levels == dataset_levels[j_slice]:
                            beta.append(exp(tau*square_norm))
                        else:
                            beta.append(exp(-tau*square_norm)/self.param.weight_gamma*abs(target_levels - dataset_levels[j_slice]))
                else:
                    beta.append(exp(-tau*square_norm))

            try:
                beta /= np.sum(beta)
            except ZeroDivisionError:
                sct.printv('WARNING : similarities are null', self.param.verbose, 'warning')
                print beta

        return np.asarray(beta)

    # ------------------------------------------------------------------------------------------------------------------
    def compute_tau(self):
        """
        Compute the weighting parameter indicating the decay constant associated with a geodesic distance
        between a given dictionary slice and a projected target image slice
        :return:
        """
        sct.printv('\nComputing Tau ... \n'
                   '(Tau is a weighting parameter indicating the decay constant associated with a geodesic distance between a given atlas and a projected target image, see Asman paper, eq (16))', 1, 'normal')
        from scipy.optimize import minimize

        def to_minimize(tau):
            """
            Compute the sum of the L0 norm between a slice segmentation and the resulting segmentation that would be
            found if the slice was a target image for a given tau

            For a given model, Tau is the parameter that would minimize this function

            :param tau:

            :return sum_norm:

            """
            sum_norm = 0
            for dic_slice in self.dictionary.slices:
                projected_dic_slice_coord = self.pca.project_array(dic_slice.im_M_flat)
                coord_dic_slice_dataset = np.delete(self.pca.dataset_coord.T, dic_slice.id, 0)
                if self.param.use_levels:
                    dic_slice_dataset_levels = np.delete(np.asarray(dic_levels), dic_slice.id, 0)
                    beta_dic_slice = self.compute_beta(projected_dic_slice_coord, target_levels=dic_slice.level, dataset_coord=coord_dic_slice_dataset, dataset_levels=dic_slice_dataset_levels, tau=tau)
                else:
                    beta_dic_slice = self.compute_beta(projected_dic_slice_coord, target_levels=None, dataset_coord=coord_dic_slice_dataset, dataset_levels=None, tau=tau)
                kj = self.select_k_slices(beta_dic_slice)
                if self.param.weight_label_fusion:
                    est_segm_j = self.label_fusion(dic_slice, kj, beta=beta_dic_slice)[0]
                else:
                    # default case
                    est_segm_j = self.label_fusion(dic_slice, kj)[0]

                sum_norm += l0_norm(dic_slice.wm_seg_M, est_segm_j.data)

            return sum_norm

        dic_levels = [dic_slice.level for dic_slice in self.dictionary.slices]

        est_tau = minimize(to_minimize, 0.001, method='Nelder-Mead', options={'xtol': 0.0005})
        sct.printv('Estimated tau : ' + str(est_tau.x[0]))

        return float(est_tau.x[0])

    # ------------------------------------------------------------------------------------------------------------------
    def select_k_slices(self, beta):
        """
        Select the K dictionary slices most similar to the target slice

        :param beta: Dictionary similarities

        :return selected: numpy array of segmentation of the selected dictionary slices
        """
        kept_slice_index = []

        if isinstance(beta[0], (list, np.ndarray)):
            for beta_slice in beta:
                selected_index = beta_slice > self.epsilon
                kept_slice_index.append(selected_index)

        else:
            kept_slice_index = beta > self.epsilon

        return np.asarray(kept_slice_index)

    # ------------------------------------------------------------------------------------------------------------------
    def label_fusion(self, target, selected_index, beta=None, type='binary'):

        """
        Compute the resulting segmentation by label fusion of the segmentation of the selected dictionary slices

        :param selected_index: array of indexes (as a boolean array) of the selected dictionary slices

        :return res_seg_model_space: Image of the resulting segmentation for the target image (in the model space)
        """
        wm_segmentation_slices = np.asarray([dic_slice.wm_seg_M for dic_slice in self.dictionary.slices])
        gm_segmentation_slices = np.asarray([dic_slice.gm_seg_M for dic_slice in self.dictionary.slices])

        res_wm_seg_model_space = []
        res_gm_seg_model_space = []

        if isinstance(selected_index[0], (list, np.ndarray)):
            # 3D image
            for i, selected_ind_by_slice in enumerate(selected_index):  # selected_slices:
                if beta is None:
                    n_selected_dic_slices = wm_segmentation_slices[selected_ind_by_slice].shape[0]
                    if n_selected_dic_slices > 0:
                        weights = [1.0/n_selected_dic_slices] * n_selected_dic_slices
                    else:
                        weights = None
                else:
                    weights = beta[i][selected_ind_by_slice]
                    weights = [w/sum(weights) for w in weights]
                wm_slice_seg = compute_majority_vote_mean_seg(wm_segmentation_slices[selected_ind_by_slice], weights=weights, type=type, threshold=0.50001)
                res_wm_seg_model_space.append(wm_slice_seg)
                target[i].set(wm_seg_m=wm_slice_seg)

                gm_slice_seg = compute_majority_vote_mean_seg(gm_segmentation_slices[selected_ind_by_slice], weights=weights, type=type)
                res_gm_seg_model_space.append(gm_slice_seg)
                target[i].set(gm_seg_m=gm_slice_seg)

        else:
            # 2D image
            if beta is None:
                n_selected_dic_slices = wm_segmentation_slices[selected_index].shape[0]
                weights = [1.0/n_selected_dic_slices] * n_selected_dic_slices
            else:
                weights = beta[selected_index]
                weights = [w/sum(weights) for w in weights]
            res_wm_seg_model_space = compute_majority_vote_mean_seg(wm_segmentation_slices[selected_index], weights=weights, type=type, threshold=0.50001)
            res_gm_seg_model_space = compute_majority_vote_mean_seg(gm_segmentation_slices[selected_index], weights=weights, type=type)

        res_wm_seg_model_space = np.asarray(res_wm_seg_model_space)
        res_gm_seg_model_space = np.asarray(res_gm_seg_model_space)

        return Image(param=res_wm_seg_model_space), Image(param=res_gm_seg_model_space)


# ----------------------------------------------------------------------------------------------------------------------
# TARGET SEGMENTATION PAIRWISE -----------------------------------------------------------------------------------------
class TargetSegmentationPairwise:
    """
    Contains all the function to segment the gray matter an a target image given a model

        - registration of the target to the model space

        - projection of the target slices on the reduced model space

        - selection of the model slices most similar to the target slices

        - computation of the resulting target segmentation by label fusion of their segmentation
    """
    def __init__(self, model, target_image=None, levels_image=None, epsilon=None):
        """
        Target gray matter segmentation constructor

        :param model: Model used to compute the segmentation, type: Model

        :param target_image: Target image to segment gray matter on, type: Image

        """
        self.model = model

        # Initialization of the target image
        if len(target_image.data.shape) == 3:
            self.target = [Slice(slice_id=i_slice, im=target_slice, reg_to_m=[]) for i_slice, target_slice in enumerate(target_image.data)]
            self.target_dim = 3
        elif len(target_image.data.shape) == 2:
            self.target = [Slice(slice_id=0, im=target_image.data, reg_to_m=[])]
            self.target_dim = 2

        if levels_image is not None and self.model.param.use_levels:
            self.load_level(levels_image)

        if self.model.param.first_reg:
            self.first_reg()

        self.target_pairwise_registration()

        if self.model.param.target_normalization:
            self.target_normalization(method='median')  # 'mean')

        # TODO: remove after testing:
        Image(param=np.asarray([target_slice.im_M for target_slice in self.target]), absolutepath='target_moved_after_normalization.nii.gz').save()

        sct.printv('\nProjecting the target image in the reduced common space ...', model.param.verbose, 'normal')
        # coord_projected_target is a list of all the coord of the target's projected slices
        self.coord_projected_target = model.pca.project([target_slice.im_M for target_slice in self.target])

        sct.printv('\nComputing the similarities between the target and the model slices ...', model.param.verbose, 'normal')
        if levels_image is not None and self.model.param.use_levels:
            self.beta = self.model.compute_beta(self.coord_projected_target, target_levels=np.asarray([target_slice.level for target_slice in self.target]), tau=self.model.tau)
        else:
            self.beta = self.model.compute_beta(self.coord_projected_target, tau=self.model.tau)

        sct.printv('\nSelecting the dictionary slices most similar to the target ...', model.param.verbose, 'normal')
        self.selected_k_slices = self.model.select_k_slices(self.beta)
        self.save_selected_slices(target_image.file_name[:-3])

        if self.model.param.verbose == 2:
            self.plot_projected_dic(nb_modes=3, to_highlight=None)  # , to_highlight='all')  # , to_highlight=(6, self.selected_k_slices[6]))

        sct.printv('\nComputing the result gray matter segmentation ...', model.param.verbose, 'normal')
        if self.model.param.weight_label_fusion:
            use_beta = self.beta
        else:
            use_beta = None
        self.model.label_fusion(self.target, self.selected_k_slices, beta=use_beta, type=self.model.param.res_type)

        if self.model.param.z_regularisation:
            sct.printv('\nRegularisation of the segmentation along the Z axis ...', model.param.verbose, 'normal')
            self.z_regularisation_2d_iteration()

        sct.printv('\nRegistering the result gray matter segmentation back into the target original space...',
                   model.param.verbose, 'normal')
        self.target_pairwise_registration(inverse=True)

    # ------------------------------------------------------------------------------------------------------------------
    def load_level(self, level_image):
        """
        Find the vertebral level of the target image slice(s) for a level image (or a string if the target is 2D)
        :param level_image: image (or a string if the target is 2D) containing level information
        :return None: the target level is set in the function
        """
        if isinstance(level_image, Image):
            '''
            nz_coord = level_image.getNonZeroCoordinates()
            for i_level_slice, level_slice in enumerate(level_image.data):
                nz_val = []
                for coord in nz_coord:
                    if coord.x == i_level_slice:
                        nz_val.append(level_slice[coord.y, coord.z])
                try:
                    self.target[i_level_slice].set(level=int(round(sum(nz_val)/len(nz_val))))
                except ZeroDivisionError:
                            sct.printv('WARNING: No level label for slice ' + str(i_level_slice) + ' of target', self.model.param.verbose, 'warning')
                            self.target[i_level_slice].set(level=0)
            '''
            for i_level_slice, level_slice in enumerate(level_image.data):
                try:
                    l = int(round(np.mean(level_slice[level_slice > 0])))
                    self.target[i_level_slice].set(level=l)
                except Exception, e:
                    sct.printv('WARNING: ' + str(e) + '\nNo level label for slice ' + str(i_level_slice) + ' of target', self.model.param.verbose, 'warning')
                    self.target[i_level_slice].set(level=0)
        elif isinstance(level_image, str):
            self.target[0].set(level=get_key_from_val(self.model.dictionary.level_label, level_image.upper()))

    # ------------------------------------------------------------------------------------------------------------------
    def first_reg(self):
        """
        Do a registration of rhe target image on the mean spinal cord segmentation to hhelp the target registration
        WARNING: DOESN'T IMPROVE THE GM SEGMENTATION RESULT

        :return None: the target moved image is set in the function
        """
        mean_sc_seg = (np.asarray(self.model.pca.mean_image) > 0).astype(int)
        Image(param=self.model.pca.mean_image, absolutepath='mean_image.nii.gz').save(type='minimize')
        # save_image(self.model.pca.mean_image, 'mean_image')
        for i, target_slice in enumerate(self.target):
            moving_target_seg = (np.asarray(target_slice.im) > 0).astype(int)
            transfo = 'BSplineSyN'
            transfo_name = transfo + '_first_reg_slice_' + str(i) + find_ants_transfo_name(transfo)[0]

            apply_ants_transfo(mean_sc_seg, moving_target_seg, binary=True, apply_transfo=False, transfo_type=transfo, transfo_name=transfo_name, metric=self.model.param.reg_metric)
            moved_target_slice = apply_ants_transfo(mean_sc_seg, target_slice.im, binary=False, search_reg=False, transfo_type=transfo, transfo_name=transfo_name, metric=self.model.param.reg_metric)

            target_slice.set(im_m=moved_target_slice)
            target_slice.reg_to_M.append((transfo, transfo_name))

            Image(param=target_slice.im, absolutepath='slice' + str(target_slice.id) + '_original_im.nii.gz').save(type='minimize')
            Image(param=target_slice.im_M, absolutepath='slice' + str(target_slice.id) + '_moved_im.nii.gz').save(type='minimize')
            # save_image(target_slice.im, 'slice' + str(target_slice.id) + '_original_im')
            # save_image(target_slice.im_M, 'slice' + str(target_slice.id) + '_moved_im')

    # ------------------------------------------------------------------------------------------------------------------
    def target_normalization(self, method='median'):
        """
        Normalization of the target using the intensity values of the mean dictionary image
        :return None: the target image is modified
        """
        sct.printv('Linear target normalization using '+method+' ...', self.model.param.verbose, 'normal')

        if method == 'mean' or method == 'median':
            dic_metrics = extract_metric_from_dic(self.model.dictionary.slices, metric=method, save=True)
            wm_metrics = []
            gm_metrics = []
            for wm_m, gm_m, wm_s, gm_s in dic_metrics.values():
                wm_metrics.append(wm_m)
                gm_metrics.append(gm_m)
            dic_wm_mean = np.mean(wm_metrics)
            dic_gm_mean = np.mean(gm_metrics)

            # getting the mean values of WM and GM in the target
            if self.model.param.target_means is None:
                if self.model.param.use_levels:
                    seg_averages_by_level = self.model.dictionary.mean_seg_by_level(type='binary')[0]
                    mean_seg_by_level = [seg_averages_by_level[self.model.dictionary.level_label[target_slice.level]] for target_slice in self.target]

                    print mean_seg_by_level
                    print 'type : ', type(mean_seg_by_level)

                    Image(param=np.asarray(mean_seg_by_level), absolutepath='mean_seg_by_level.nii.gz').save()
                    Image(param=np.asarray([target_slice.im_M for target_slice in self.target]), absolutepath='target_moved.nii.gz').save()

                    target_metric = extract_metric_from_dic(self.target, seg_to_use=mean_seg_by_level, metric=method, save=True, output='metric_in_target.txt')

                    # metric averaged overall
                    '''
                    print 'USE AVERAGED TARGET METRIC'
                    target_metric = [np.mean(target_metric.values(), axis=0) for i in range(len(self.target))]
                    '''
                    # metric averaged by level
                    '''
                    target_metric_by_level = {}
                    for target_slice in self.target:
                        slice_metric = target_metric[target_slice.id]
                        if target_slice.level in target_metric_by_level.keys():
                            target_metric_by_level[target_slice.level].append(slice_metric)
                        else:
                            target_metric_by_level[target_slice.level] = [slice_metric]
                    for l, metric in target_metric_by_level.items():
                        target_metric_by_level[l] = np.mean(metric, axis=0)
                    '''

                else:
                    sct.printv('WARNING: No mean value of the white matter and gray matter intensity were provided, nor the target vertebral levels to estimate them\n'
                               'The target will not be normalized.', self.model.param.verbose, 'warning')
                    self.model.param.target_normalization = False
                    target_metric = None
            else:
                target_metric = [(self.model.param.target_means[0], self.model.param.target_means[1], 0, 0) for i in range(len(self.target))]

            if type(target_metric) == type({}): # if target_metric is a dictionary
                differences = [m[1]-m[0] for m in target_metric.values()]
                lim_diff = np.median(differences) - np.std(differences)
            else:
                differences = [0]
                lim_diff = 0

            # normalizing
            if target_metric is not None:
                i = 0
                for target_slice in self.target:
                    old_image = target_slice.im_M
                    '''
                    if self.model.param.target_means is None and self.model.param.use_levels:
                        wm_mean, gm_mean, wm_std, gm_std = target_metric_by_level[target_slice.level]
                        print 'USE MEAN BY LEVEL OF THE TARGET METRIC'
                    else:
                    '''
                    wm_metric, gm_metric, wm_std, gm_std = target_metric[target_slice.id]
                    if gm_metric-wm_metric < lim_diff:
                        print 'CORRECTING WM VALUE FOR SLICE ', i
                        wm_metric = gm_metric-np.median(differences)
                    new_image = (old_image - wm_metric)*(dic_gm_mean - dic_wm_mean)/(gm_metric - wm_metric) + dic_wm_mean
                    new_image[old_image < 1] = 0  # put a 0 the min background

                    target_slice.im_M = new_image
                    Image(param=new_image, absolutepath='target_slice'+str(i)+'_mean_normalized.ni.gz').save()
                    i += 1

        if method == 'min-max':
            min_sum = 0
            for model_slice in self.model.dictionary.slices:
                min_sum += model_slice.im_M[model_slice.im_M > 1].min()
            new_min = min_sum/self.model.dictionary.J
            # new_min = self.model.dictionary.mean_image[self.model.dictionary.mean_image > 300].min()
            new_max = self.model.dictionary.mean_image.max()

            i = 0
            for target_slice in self.target:
                # with mean image as reference
                # target_slice.im = target_slice.im/self.model.dictionary.mean_image*self.model.dictionary.mean_image.max()

                # linear with min=0
                # target_slice.im = target_slice.im*self.model.dictionary.mean_image.max()/(target_slice.im.max()-target_slice.im.min())

                # linear with actual min (WM min)
                old_image = target_slice.im_M
                old_min = target_slice.im_M[target_slice.im_M > 0].min()
                old_max = target_slice.im_M.max()
                new_image = (old_image - old_min)*(new_max - new_min)/(old_max - old_min) + new_min
                # new_image[new_image < new_min+1] = 0  # put a 0 the min background
                new_image[old_image < 1] = 0

                target_slice.im_M = new_image
                Image(param=new_image, absolutepath='target_slice'+str(i)+'_min_max_normalized.ni.gz').save()
                i += 1

        if method == 'mean-sep':
            # test normalization with separate means
            import copy
            dic_metrics = extract_metric_from_dic(self.model.dictionary.slices, save=True)
            wm_metrics = []
            gm_metrics = []
            wm_stds = []
            gm_stds = []
            for wm_m, gm_m, wm_s, gm_s in dic_metrics.values():
                wm_metrics.append(wm_m)
                gm_metrics.append(gm_m)
                wm_stds.append(wm_s)
                gm_stds.append(gm_s)
            dic_wm_mean = np.mean(wm_metrics)
            dic_gm_mean = np.mean(gm_metrics)
            dic_wm_std = np.std(wm_stds)
            dic_gm_std = np.std(gm_stds)

            print 'Dic wm: ', dic_wm_mean, ' +- ', dic_wm_std
            print 'Dic gm: ', dic_gm_mean, ' +- ', dic_gm_std

            seg_averages_by_level_bin = self.model.dictionary.mean_seg_by_level(type='binary')[0]
            mean_seg_by_level_bin = [seg_averages_by_level_bin[self.model.dictionary.level_label[target_slice.level]] for target_slice in self.target]
            seg_averages_by_level_prob = self.model.dictionary.mean_seg_by_level(type='prob')[0]
            mean_seg_by_level_prob = [seg_averages_by_level_prob[self.model.dictionary.level_label[target_slice.level]] for target_slice in self.target]

            target_metric = extract_metric_from_dic(self.target, seg_to_use=mean_seg_by_level_bin, save=True)

            for i, target_slice in enumerate(self.target):
                old_image = target_slice.im_M
                # new_image = copy.deepcopy(old_image)
                wm_metric, gm_metric, wm_std, gm_std = target_metric[target_slice.id]

                # GM:
                new_gm = ((old_image - gm_metric)*dic_gm_std/gm_std+dic_gm_mean)*mean_seg_by_level_prob[i]
                # WM:
                new_wm = ((old_image - gm_metric)*dic_gm_std/gm_std+dic_gm_mean)*(1-mean_seg_by_level_prob[i])
                # concatenation of GM and WM:
                new_image = new_wm + new_gm

                new_image[old_image < 1] = 0
                target_slice.im_M = new_image

                Image(param=new_image, absolutepath='target_slice'+str(i)+'_mean_sep_normalized.ni.gz').save()

    # ------------------------------------------------------------------------------------------------------------------
    def target_pairwise_registration(self, inverse=False):
        """
        Register the target image into the model space

        Affine (or rigid + affine) registration of the target on the mean model image --> pairwise

        :param inverse: if True, apply the inverse warping field of the registration target -> model space
        to the result gray matter segmentation of the target
        (put it back in it's original space)

        :return None: the target attributes are set in the function
        """
        if not inverse:
            # Registration target --> model space
            mean_dic_im = self.model.pca.mean_image
            for i, target_slice in enumerate(self.target):
                if not self.model.param.first_reg:
                    moving_target_slice = target_slice.im
                else:
                    moving_target_slice = target_slice.im_M
                for transfo in self.model.dictionary.coregistration_transfos:
                    transfo_name = transfo + '_transfo_target2model_space_slice_' + str(i) + find_ants_transfo_name(transfo)[0]
                    target_slice.reg_to_M.append((transfo, transfo_name))

                    moving_target_slice = apply_ants_transfo(mean_dic_im, moving_target_slice, binary=False, transfo_type=transfo, transfo_name=transfo_name, metric=self.model.param.reg_metric)
                self.target[i].set(im_m=moving_target_slice)

        else:
            # Inverse registration result in model space --> target original space
            for i, target_slice in enumerate(self.target):
                moving_wm_seg_slice = target_slice.wm_seg_M
                moving_gm_seg_slice = target_slice.gm_seg_M

                for transfo in target_slice.reg_to_M:
                    if self.model.param.res_type == 'binary':
                        bin = True
                    else:
                        bin = False
                    moving_wm_seg_slice = apply_ants_transfo(self.model.dictionary.mean_seg, moving_wm_seg_slice, search_reg=False, binary=bin, inverse=1, transfo_type=transfo[0], transfo_name=transfo[1], metric=self.model.param.reg_metric)
                    moving_gm_seg_slice = apply_ants_transfo(self.model.dictionary.mean_seg, moving_gm_seg_slice, search_reg=False, binary=bin, inverse=1, transfo_type=transfo[0], transfo_name=transfo[1], metric=self.model.param.reg_metric)

                target_slice.set(wm_seg=moving_wm_seg_slice)
                target_slice.set(gm_seg=moving_gm_seg_slice)

    # ------------------------------------------------------------------------------------------------------------------
    def z_regularisation_2d_iteration(self, coeff=0.4):
        """
        Z regularisation option WARNING: DOESN'T IMPROVE THE GM SEGMENTATION RESULT
        Use the result segmentation of the first iteration, the segmentation of slice i is the weighted average of the segmentations of slices i-1 and i+1 and the segmentation of slice i
        :param coeff: weight on each adjacent slice
        :return:
        """
        for i, target_slice in enumerate(self.target[1:-1]):
            adjacent_wm_seg = []  # coeff * self.target[i-1].wm_seg_M, (1-2*coeff) * target_slice.wm_seg_M, coeff * self.target[i+1].wm_seg_M
            adjacent_gm_seg = []  # coeff * self.target[i-1].gm_seg_M, (1-2*coeff) * target_slice.gm_seg_M, coeff * self.target[i+1].gm_seg_M

            precision = 100
            print int(precision*coeff)
            for k in range(int(precision*coeff)):
                adjacent_wm_seg.append(self.target[i-1].wm_seg_M)
                adjacent_wm_seg.append(self.target[i+1].wm_seg_M)
                adjacent_gm_seg.append(self.target[i-1].gm_seg_M)
                adjacent_gm_seg.append(self.target[i+1].gm_seg_M)

            for k in range(precision - 2*int(precision*coeff)):
                adjacent_wm_seg.append(target_slice.wm_seg_M)
                adjacent_gm_seg.append(target_slice.gm_seg_M)

            adjacent_wm_seg = np.asarray(adjacent_wm_seg)
            adjacent_gm_seg = np.asarray(adjacent_gm_seg)

            new_wm_seg = compute_majority_vote_mean_seg(adjacent_wm_seg, type=self.model.param.res_type, threshold=0.50001)
            new_gm_seg = compute_majority_vote_mean_seg(adjacent_gm_seg, type=self.model.param.res_type)  # , threshold=0.4999)

            target_slice.set(wm_seg_m=new_wm_seg)
            target_slice.set(gm_seg_m=new_gm_seg)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_projected_dic(self, nb_modes=3, to_highlight=1):
        """
        plot the pca first modes and the target projection if target is provided.

        on a second plot, highlight the selected dictionary slices for one target slice in particular

        :param nb_modes:
        :return:
        """
        self.model.pca.plot_projected_dic(nb_modes=nb_modes, target_coord=self.coord_projected_target, target_levels=[t_slice.level for t_slice in self.target]) if self.coord_projected_target is not None \
            else self.model.pca.plot_projected_dic(nb_modes=nb_modes)

        if to_highlight == 'all':
            for i in range(len(self.target)):
                self.model.pca.plot_projected_dic(nb_modes=nb_modes, target_coord=self.coord_projected_target, target_levels=[t_slice.level for t_slice in self.target], to_highlight=(i, self.selected_k_slices[i]))
        elif to_highlight is not None:
            self.model.pca.plot_projected_dic(nb_modes=nb_modes, target_coord=self.coord_projected_target, target_levels=[t_slice.level for t_slice in self.target], to_highlight=(to_highlight, self.selected_k_slices[to_highlight])) if self.coord_projected_target is not None \
            else self.model.pca.plot_projected_dic()

    # ------------------------------------------------------------------------------------------------------------------
    def save_selected_slices(self, target_name):
        slice_levels = np.asarray([(dic_slice.id, self.model.dictionary.level_label[dic_slice.level]) for dic_slice in self.model.dictionary.slices])
        fic_selected_slices = open(target_name + '_selected_slices.txt', 'w')
        if self.target_dim == 2:
            fic_selected_slices.write(str(slice_levels[self.selected_k_slices.reshape(self.model.dictionary.J,)]))
        elif self.target_dim == 3:
            for target_slice in self.target:
                fic_selected_slices.write('slice ' + str(target_slice.id) + ': ' + str(slice_levels[self.selected_k_slices[target_slice.id]]) + '\n')
        fic_selected_slices.close()


# ----------------------------------------------------------------------------------------------------------------------
# GRAY MATTER SEGMENTATION SUPERVISED METHOD ---------------------------------------------------------------------------
class GMsegSupervisedMethod():
    """
    Gray matter segmentation supervised method:

    Load a dictionary (training data set), compute or load a model from this dictionary
sct_Image
    Load a target image to segment and do the segmentation using the model
    """
    def __init__(self, target_fname, level_fname, model, gm_seg_param=None):
        # build the appearance model
        self.model = model

        sct.printv('\nConstructing target image ...', verbose=gm_seg_param.verbose, type='normal')
        # construct target image
        self.target_image = Image(target_fname)
        original_hdr = self.target_image.hdr
        # build a target segmentation
        level_im = None
        if level_fname is not None:
            if len(level_fname) < 3: #TODO: replace by a check if file
                # in this case the level is a string and not an image
                level_im = level_fname
            else:
                level_im = Image(level_fname)
        else:
            gm_seg_param.use_levels = False

        # TARGET PAIRWISE SEGMENTATION
        if level_im is not None:
            self.target_seg_methods = TargetSegmentationPairwise(self.model, target_image=self.target_image, levels_image=level_im)
        else:
            self.target_seg_methods = TargetSegmentationPairwise(self.model, target_image=self.target_image)

        # get & save the result gray matter segmentation
        if gm_seg_param.output_name == '':
            suffix = ''
            if self.model.param.dev:
                suffix += '_' + gm_seg_param.res_type
                for transfo in self.model.dictionary.coregistration_transfos:
                    suffix += '_' + transfo
                if self.model.param.use_levels:
                    suffix += '_with_levels_' + '_'.join(str(self.model.param.weight_gamma).split('.'))  # replace the '.' by a '_'
                else:
                    suffix += '_no_levels'
                if self.model.param.z_regularisation:
                    suffix += '_Zregularisation'
                if self.model.param.target_normalization:
                    suffix += '_normalized'

            name_res_wmseg = sct.extract_fname(target_fname)[1] + '_wmseg' + suffix  # TODO: remove suffix when parameters are all optimized
            name_res_gmseg = sct.extract_fname(target_fname)[1] + '_gmseg' + suffix  # TODO: remove suffix when parameters are all optimized
            ext = sct.extract_fname(target_fname)[2]
        else:
            name_res_wmseg = ''.join(sct.extract_fname(gm_seg_param.output_name)[:-1]) + '_wmseg'
            name_res_gmseg = ''.join(sct.extract_fname(gm_seg_param.output_name)[:-1]) + '_gmseg'
            ext = sct.extract_fname(gm_seg_param.output_name)[2]

        if len(self.target_seg_methods.target) == 1:
            self.res_wm_seg = Image(param=np.asarray(self.target_seg_methods.target[0].wm_seg), absolutepath=name_res_wmseg + ext)
            self.res_gm_seg = Image(param=np.asarray(self.target_seg_methods.target[0].gm_seg), absolutepath=name_res_gmseg + ext)
        else:
            self.res_wm_seg = Image(param=np.asarray([target_slice.wm_seg for target_slice in self.target_seg_methods.target]), absolutepath=name_res_wmseg + ext)
            self.res_gm_seg = Image(param=np.asarray([target_slice.gm_seg for target_slice in self.target_seg_methods.target]), absolutepath=name_res_gmseg + ext)

        self.res_wm_seg.hdr = original_hdr
        self.res_wm_seg.file_name = name_res_wmseg
        self.res_wm_seg.save(type='minimize')

        self.res_gm_seg.hdr = original_hdr
        self.res_gm_seg.file_name = name_res_gmseg
        self.res_gm_seg.save(type='minimize')

        self.corrected_wm_seg = correct_wmseg(self.res_gm_seg, self.target_image, name_res_wmseg, original_hdr)

    def show(self):

        sct.printv('\nShowing the pca modes ...')
        self.model.pca.show_all_modes()

        sct.printv('\nPloting the projected dictionary ...')
        self.target_seg_methods.plot_projected_dic(nb_modes=3)

        sct.printv('\nShowing PCA mode graphs ...')
        self.model.pca.show_mode_variation()




########################################################################################################################
# ------------------------------------------------------  MAIN ------------------------------------------------------- #
########################################################################################################################

if __name__ == "__main__":
    param = SegmentationParam()
    input_target_fname = None
    input_level_fname = None
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_input = param.path_model + "/errsm_34.nii.gz"
        fname_input = param.path_model + "/errsm_34_seg_in.nii.gz"
    else:
        param_default = SegmentationParam()

        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Project all the input image slices on a PCA generated from set of t2star images')
        parser.add_option(name="-i",
                          type_value="file",
                          description="target image to segment"
                                      "if -i isn't used, only the model is computed/loaded",
                          mandatory=False,
                          example='t2star.nii.gz')
        parser.add_option(name="-o",
                          type_value="str",
                          description="output name for the results",
                          mandatory=False,
                          example='t2star_res.nii.gz')
        parser.add_option(name="-model",
                          type_value="folder",
                          description="Path to the dictionary of images",
                          mandatory=False,
                          example='/home/jdoe/data/dictionary')
        parser.add_option(name="-todo-model",
                          type_value="multiple_choice",
                          description="Load or compute the model",
                          mandatory=False,
                          example=['load', 'compute'])
        parser.add_option(name="-l",
                          type_value="str",
                          description="Image containing level labels for the target or str indicating the level",
                          mandatory=False,
                          example='MNI-Poly-AMU_level_IRP.nii.gz')
        parser.add_option(name="-reg",
                          type_value=[[','], 'str'],
                          description="list of transformations to apply to co-register the dictionary data",
                          mandatory=False,
                          default_value=['Affine'],
                          example=['SyN'])
        parser.add_option(name="-weight",
                          type_value='float',
                          description="weight parameter on the level differences to compute the similarities (beta)",
                          mandatory=False,
                          default_value=2.5,
                          example=2.0)
        parser.add_option(name="-use-levels",
                          type_value='multiple_choice',
                          description="1: Use vertebral level information, 0: no ",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1'])
        parser.add_option(name="-denoising",
                          type_value='multiple_choice',
                          description="1: Adaptative denoising from F. Coupe algorithm, 0: no  WARNING: It affects the model you should use (if denoising is applied to the target, the model should have been coputed with denoising too)",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1'])
        parser.add_option(name="-normalize",
                          type_value='multiple_choice',
                          description="1: Normalization of the target image's intensity using mean intensity values of the WM and the GM",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1'])
        parser.add_option(name="-means",
                          type_value=[[','], 'float'],
                          description="Mean intensity values in the target white matter and gray matter (separated by a comma without white space)\n"
                                      "If not specified, the mean intensity values of the target WM and GM  are estimated automatically using the dictionary average segmentation by level.\n"
                                      "Only if the -normalize flag is used",
                          mandatory=False,
                          default_value=None,
                          example=["450,540"])
        parser.add_option(name="-res-type",
                          type_value='multiple_choice',
                          description="Type of result segmentation : binary or probabilistic",
                          mandatory=False,
                          default_value='prob',
                          example=['binary', 'prob'])
        parser.add_option(name="-v",
                          type_value='multiple_choice',
                          description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                          mandatory=False,
                          default_value=0,
                          example=['0', '1', '2'])
        '''
        parser.add_option(name="-first-reg",
                          type_value='multiple_choice',
                          description="Apply a Bspline registration using the spinal cord edges target --> model first",
                          mandatory=False,
                          default_value=0,
                          example=['0', '1'])
        parser.add_option(name="-z",
                          type_value='multiple_choice',
                          description="1: Z regularisation, 0: no ",
                          mandatory=False,
                          default_value=0,
                          example=['0', '1'])
        parser.add_option(name="-weighted-label-fusion",
                          type_value='multiple_choice',
                          description="Use the similarities as a weights for the label fusion",
                          mandatory=False,
                          default_value=0,
                          example=['0', '1'])
        parser.add_option(name="-weighted-similarity",
                          type_value='multiple_choice',
                          description="Use a PCA mode weighted norm for the computation of the similarities instead of the euclidean square norm",
                          mandatory=False,
                          default_value=0,
                          example=['0', '1'])
        '''


        arguments = parser.parse(sys.argv[1:])

        if "-i" in arguments:
            input_target_fname = arguments["-i"]
        if "-o" in arguments:
            param.output_name = arguments["-o"]
        if "-model" in arguments:
            param.path_model = arguments["-model"]
        if "-todo-model" in arguments:
            param.todo_model = arguments["-todo-model"]
        if "-reg" in arguments:
            param.reg = arguments["-reg"]
        if "-l" in arguments:
            input_level_fname = arguments["-l"]
        if "-weight" in arguments:
            param.weight_gamma = arguments["-weight"]
        if "-use-levels" in arguments:
            param.use_levels = bool(int(arguments["-use-levels"]))
        if "-denoising" in arguments:
            param.target_denoising = bool(int(arguments["-denoising"]))
        if "-normalize" in arguments:
            param.target_normalization = bool(int(arguments["-normalize"]))
        if "-means" in arguments:
            param.target_means = arguments["-means"]
        if "-res-type" in arguments:
            param.res_type = arguments["-res-type"]
        if "-v" in arguments:
            param.verbose = int(arguments["-v"])
        '''
        if "-first-reg" in arguments:
            param.first_reg = bool(int(arguments["-first-reg"]))
        if "-z" in arguments:
            param.z_regularisation = bool(int(arguments["-z"]))
        if "-weighted-label-fusion" in arguments:
            param.weight_label_fusion = bool(int(arguments["-weighted-label-fusion"]))
        if "-weighted-similarity" in arguments:
            param.mode_weight_similarity = bool(int(arguments["-weighted-similarity"]))
        '''

    model = Model(model_param=param, k=0.8)
    if input_target_fname is not None:
        gm_seg_method = GMsegSupervisedMethod(input_target_fname, input_level_fname, model, gm_seg_param=param)
        if param.verbose == 2:
            gm_seg_method.show()