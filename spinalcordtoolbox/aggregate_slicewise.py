#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with metrics aggregation (mean, std, etc.) across slices and/or vertebral levels

import numpy as np
import sct_utils as sct
import msct_image
from spinalcordtoolbox.utils import parse_num_list
from spinalcordtoolbox.template import get_slices_from_vertebral_levels


def aggregate_per_slice_or_level(metrics, slices=[], levels=[], perslice=True, perlevel=False, im_vert_level=None,
                                 group_funcs=(('mean', np.mean),)):
    """
    It is assumed that each element of a metric's vector correspond to a slice. E.g., index #2 corresponds to slice #2.
    :param metrics:
    :param slices:
    :param levels:
    :param perslice:
    :param perlevel:
    :param im_vert_level:
    :param group_funcs:
    :return:
    """

    # TODO: remove the csv writing part from this function and create wrapper average_per_slice_or_level_to_file().
    # TODO: check last dimension is S-I
    # TODO: move the parse_num_list to the front-end
    # agg_metrics = dict((metric, ()) for metric in metrics.keys())
    agg_metrics = dict((metric, dict()) for metric in metrics.keys())
    # # nz = len(metrics[0])  # retrieve number of slices from the first metric (assuming they all have the same shape)
    # if slices:
    #     # if user specified slices of interest, convert to comma-separated string: '2:5,6' -> '2,3,4,5,6'
    #     list_slices = parse_num_list(slices)
    # else:
    #     # if no slices is specified, use all slices in the image
    #     list_slices = np.arange(nz).tolist()
    # list_slices.reverse()  # more intuitive to list slices in descending mode (i.e. from head to toes)
    # if perslice with slices: ['1', '2', '3', '4']
    # important: each slice number should be separated by "," not ":"
    # slicegroups = [str(i) for i in list_slices]
    # if user does not want to output metric per slice, then create a single element in slicegroups
    # if not perslice:
        # ['1', '2', '3', '4'] -> ['1,2,3,4']
        # slicegroups = [','.join(slicegroups)]

    # if user selected vertebral levels
    if levels:
        im_vert_level.change_orientation("RPI")  # last dim should be k
        slicegroups = [tuple(get_slices_from_vertebral_levels(im_vert_level, level)) for level in levels]
        if perlevel:
            # slicegroups = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
            # vertgroups = [(2,), (3,), (4,)]
            vertgroups = [tuple([level]) for level in levels]
        # output aggregate metrics across levels
        else:
            # slicegroups = [(0, 1, 2, 3, 4, 5, 6, 7, 8)]
            slicegroups = [tuple([val for sublist in slicegroups for val in sublist])]  # flatten list_slices
            # vertgroups = [(2, 3, 4)]
            vertgroups = [tuple([level for level in levels])]
        #
        # # Re-define slices_of_interest according to the vertebral levels selected by user
        # list_levels = parse_num_list(levels)
        # slicegroups = []
        # vertgroups = [str(i) for i in list_levels]
        # # for each level, find the matching slices and group them
        # for level in list_levels:
        #     list_slices = get_slices_from_vertebral_levels(im_vert_level, level)
        #     list_slices.reverse()
        #     slicegroups.append(','.join([str(i) for i in list_slices]))
        # # if user does not want to output metric per vert level, create a single element in vertgroups
        # if not perlevel:
        #     # ['2', '3', '4'] -> ['2,3,4']
        #     vertgroups = [','.join(vertgroups)]
        #     slicegroups = [','.join(slicegroups)]

    # loop across slice group
    for slicegroup in slicegroups:
        try:
            # # convert list of strings into list of int to use as index
            # ind_slicegroup = [int(i) for i in slicegroup.split(',')]
            # if levels:
            #     vertgroup = vertgroups[slicegroups.index(slicegroup)]
            # else:
            #     vertgroup = ''
            # # average metrics within slicegroup
            # # TODO: ADD STD
            # # change "," for ";" otherwise it will be parsed by the CSV format
            # # TODO: instead of having a long list of ;-separated numbers, it would be nicer to separate long number
            # # TODO (cont.) suites with ":". E.g.: '1,2,3,4,5' -> '1:5'. See #1932
            # slicegroup = slicegroup.replace(",", ";")
            # vertgroup = vertgroup.replace(",", ";")
            # # encode slice and levels

            # add level info
            agg_metrics['VertLevel'][slicegroup] = vertgroups[slicegroups.index(slicegroup)]

            for metric in metrics.keys():
                metric_data = [metrics[metric][slice] for slice in slicegroup]
                agg_metrics[metric][slicegroup] = dict((name, func(metric_data)) for (name, func) in group_funcs)
        except IndexError:
            sct.log.warning('The slice request is out of the range of the image')
    return agg_metrics
#
#     # Create output csv file
#     fname_out = file_out + '.csv'
#     file_results = open(fname_out, 'w')
#     file_results.write(','.join(["Slice [z]", "Vertebral level"] + header) + '\n')
#     # loop across slice group
#     for slicegroup in slicegroups:
#         try:
#             # convert list of strings into list of int to use as index
#             ind_slicegroup = [int(i) for i in slicegroup.split(',')]
#             if vert_levels:
#                 vertgroup = vertgroups[slicegroups.index(slicegroup)]
#             else:
#                 vertgroup = ''
#             # average metrics within slicegroup
#             # TODO: ADD STD
#             # change "," for ";" otherwise it will be parsed by the CSV format
#             # TODO: instead of having a long list of ;-separated numbers, it would be nicer to separate long number
#             # TODO (cont.) suites with ":". E.g.: '1,2,3,4,5' -> '1:5'. See #1932
#             slicegroup = slicegroup.replace(",", ";")
#             vertgroup = vertgroup.replace(",", ";")
#             # build csv file
#             file_results.write(','.join([slicegroup, vertgroup] + [str(np.mean(i[ind_slicegroup])) for i in metrics])
#                                + '\n')
#         except ValueError:
#             # the slice request is out of the range of the image
#             sct.printv('The slice(s) requested is out of the range of the image', type='warning')
#     file_results.close()
#     # TODO: printout csv
#     # TODO: return dict or panda structure instead of writing csv file
#
#
# def aggregate_metrics_by_vertebral_level(metrics, im_vert_levels, levels, group_funcs=None):
#     """
#     Aggregate metrics by vertebral level.
#     :param dict metrics: dict of (metric name, sequence of unidimensional metric values)
#     :param Image im_vert_levels: image from which vertebral levels are figured out
#     :param list(int) levels: list of int: levels to aggregate metrics from
#     :param group_funcs: list of (name, func) of functions applied on values of a metric on the slices of a same level,
#     for each vertebral level
#     :return: out: dict of aggregated metrics
#     """
#     if group_funcs is None:
#         group_funcs = (('mean', np.mean),)
#     agg_metrics = dict((metric, dict()) for metric in metrics.keys())
#     for level in levels:
#         idx_slices = get_slices_from_vertebral_levels(im_vert_levels, level)
#         for metric in metrics.keys():
#             metric_data = metrics[metric]
#             level_data = [metric_data[idx_slice] for idx_slice in idx_slices if 0 <= idx_slice < len(metric_data) ]
#             if not level_data:
#                 break
#             agg_metrics[metric][level] = dict((name, func(level_data)) for (name, func) in group_funcs)
#     return agg_metrics