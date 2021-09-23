# s, c, e_c, k_p, k_l, k_e, p_v = pruned_measures(prune_type='percentage',
#                                                 map=lambda x: 1 /
#                                                 (np.log10(1 + x)),
#                                                 map_name='inv_log',
#                                                 k_vals=[1, 2, 5, 10, 20, 50],
#                                                 prune_vals_given=None,
#                                                 prune_vals_number=10,
#                                                 prune_vals_max=0.85,
#                                                 prune_vals_offset=0.01,
#                                                 filter_att='GRUPO',
#                                                 filter_val='control sano',
#                                                 op=op.eq,
#                                                 remove_nodes=[34, 83],
#                                                 trace=True,
#                                                 trace_k_paths=False,
#                                                 only_plot=True)

# k_shortest_path_save(prune_type='percentage',
#                      map=lambda x: 1 / x,
#                      map_name='inv',
#                      k=50,
#                      prune_vals_given=None,
#                      prune_vals_number=10,
#                      prune_vals_max=0.85,
#                      prune_vals_offset=0.01,
#                      filter_att='GRUPO',
#                      filter_val='migraña crónica',
#                      op=op.eq,
#                      remove_nodes=[34, 83],
#                      trace=True,
#                      trace_k_paths=False,
#                      prune_start=1,
#                      case_start=34,
#                      prune_finish=1,
#                      case_finish=34)

# c_dir = os.getcwd()
# os.chdir(
#     "C:\\Users\Tomas\Desktop\Tesis\Programacion\\results\pruning\k_paths_txt\inv_log_migraña_crónica"
# )
# f_list = os.listdir()
# f_list.reverse()
# # f_list = f_list[:30]
# for file in f_list:
#     # print('prev:\t', file)
#     # name_split = file.split('.t')
#     # name_split[0] = name_split[0][:-4]
#     # name_split[1] = "txt"
#     # ipdb.set_trace()
#     # name_split[0] = name_split[0][1:]
#     # name_split[1] = name_split[1].split('_', 1)
#     # k = int(name_split[1][0]) + 1
#     # name_split[1][0] = str(k)
#     # name_split[1] = '_'.join(name_split[1])
#     # f_name = '='.join(name_split)
#     # print("post:\t", f_name)
#     f_name = file.replace("_log", "")  #".".join(name_split)
#     os.rename(file, f_name)
# os.chdir(c_dir)

# def PCA(data, correlation=False, ndim=1, groups=[0, 1, 2]):
#     # ipdb.set_trace()
#     data = np.copy(data[..., groups])
#     if correlation:
#         data_mat = data_correlation(data)
#     else:
#         data_mat = data_covariance(data)
#     e_vals, e_vecs = np.zeros(shape=(2,) + data.shape[:-1], dtype=object)
#     for k, measure_mat in enumerate(data_mat):
#         for i in range(data.shape[1]):
#             e_val, e_vec = np.linalg.eigh(measure_mat[i])
#             e_vals[k, i], e_vecs[k, i] = (
#                 e_val[::-1][:ndim],
#                 e_vec.T[::-1][:ndim],
#             )
#     data_scores = np.empty(shape=data.shape, dtype=object)
#     for k, measure in enumerate(data):
#         for i in range(data.shape[1]):
#             for j in range(data.shape[2]):
#                 scores = np.zeros(shape=(measure[i, j].shape[1], ndim))
#                 for dim in range(ndim):
#                     scores[:, dim] = e_vecs[k, i][dim].dot(measure[i, j])
#                 data_scores[k, i, j] = scores

#     return e_vals, e_vecs, data_scores


# def simple_PCA(data, ndim=1, rowvar=True):
#     if rowvar:
#         data = data.T
#     cov_mat = np.cov(data, rowvar=False)
#     e_val, e_vec = np.linalg.eigh(cov_mat)
#     e_val = e_val[::-1][:ndim]
#     e_vec = e_vec[:, ::-1][:, :ndim]
#     score = (data - data.mean(axis=0)).dot(e_vec)
# return e_val, e_vec, score

# def data_covariance(data):
#     data_cov = np.empty(shape=(data.shape[:-1]), dtype=object)
#     for k, measure in enumerate(data):
#         for i in range(data.shape[1]):
#             cov_matrix = np.cov(np.hstack(measure[i]))
#             data_cov[k, i] = cov_matrix
#         # cov_matrix[np.triu_indices_from(cov_matrix)]
#     return data_cov


# def data_correlation(data):
#     data_corr = np.empty(shape=(data.shape), dtype=object)
#     for k, measure in enumerate(data):
#         for i in range(data.shape[1]):
#             corr_matrix = np.corrcoef(np.hstack(measure[i]))
#             data_corr[k, i] = corr_matrix
#         # cov_matrix[np.triu_indices_from(cov_matrix)]
#     return data_corr
