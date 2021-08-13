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