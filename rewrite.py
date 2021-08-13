import networkx as nx
import numpy as np
import time
import ipdb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import operator as op
import os
from networkx.classes.function import path_weight
from networkx.algorithms.flow import shortest_augmenting_path


def connectomes_filtered(filter_attr,
                         filter_value,
                         operator,
                         all=False,
                         ret_cases=False,
                         remove_nodes=[34, 83]):
    # esta funcion devuelve todas las matrices de conexión que cumplan con que
    # operator(filter_attr,filter_value), donde operator es un operador booleano

    # los path para la lista de conectomas y los conectomas (matrices)
    list_path = r'C:\Users\Tomas\Desktop\Tesis\datos\conjunto_datos_conectomica_migranya\lista_de_casos.xlsx'
    connectomes_path = r'C:\Users\Tomas\Desktop\Tesis\datos\conjunto_datos_conectomica_migranya\matrices_conectividad\\'
    # cargo la lista de conectomas a un panda dataframe
    cases = pd.read_excel(list_path)
    # si quiero todos, paso todos
    if all:
        cases = cases['NOMBRE '].values
    # si no, me fijo que conectomas cumplen con la condición
    else:
        # lo paso a minuscula para no tener problema
        cases[filter_attr] = cases[filter_attr].str.lower()
        cases = cases[operator(cases[filter_attr], filter_value)]['NOMBRE ']
        # devuelvo un array con los nombres de los casos
        cases = cases.values
    # ahora voy a cargar los conectomas
    connectomes = np.zeros(shape=(len(cases), 84, 84))
    for i, case in enumerate(cases):
        # contruyo el path del conectoma que quiero cargar
        connectome_path = connectomes_path + case + '_conectomica_fiber_count.csv'
        # cargo el conectoma
        connectome = np.genfromtxt(connectome_path)
        # borro la diagonal
        connectome[np.tril_indices(84)] = 0
        connectomes[i] = connectome

    # hay un problema con los nodos 34 y 83 asi que los borramos
    if remove_nodes != None:
        connectomes = np.delete(connectomes, remove_nodes, axis=1)
        connectomes = np.delete(connectomes, remove_nodes, axis=2)

    # por si alguna razon alguien quiere el nombre de los casos
    if ret_cases:
        return connectomes, cases
    else:
        return connectomes


def prune_connectomes(connectomes_original, prune_type, val):
    # hace prunning a los conectomas
    connectomes = np.copy(connectomes_original)
    # percentage es borrar el porcentage mas bajo de las aristas
    if prune_type == 'percentage':
        for connectome in connectomes:
            # ordeno las aristas (cuento las que son 0 tmb)
            mat = np.sort(connectome[np.triu_indices(connectome.shape[0], 1)])
            n_elements = len(mat)
            # elijo el valor que determine el porcentaje que le pase
            percentile_value = mat[int(val * (n_elements - 1))]
            # todo lo que este debajo de eso lo hago 0
            connectome[connectome <= percentile_value] = 0
        return connectomes
    # aca borro todas las conexiones cuya media sea mas baja que el valor
    # de threshols
    elif prune_type == 'threshold':
        mask = connectomes.mean(axis=0)
        mask = mask <= val
        mask = mask[np.newaxis, :, :]
        mask = np.repeat(mask, len(connectomes), axis=0)
        connectomes[mask] = 0
        return connectomes
    # aca borro todas las conexiones cuyo coeficiente de variación
    # sea mas alto que cierto valor
    elif prune_type == 'var_coeff':
        mask = connectomes.std(axis=0) / connectomes.mean(axis=0)
        mask[np.isnan(mask)] = 0
        mask = mask >= val
        mask = mask[np.newaxis, :, :]
        mask = np.repeat(mask, len(connectomes), axis=0)
        connectomes[mask] = 0
        return connectomes
    else:
        raise Exception(
            'Invalid prune_type, choose percentage,threshold or var_coeff')
        pass


def add_edge_map(G, map, map_name):
    # agrego un mapeo a los ejes
    for u, v in G.edges():
        G.edges[u, v][map_name] = map(G.edges[u, v]['weight'])
    pass


prune_type = 'percentage'
map = lambda x: 1 / np.log10(1 + x)
map_name = 'inv_log'
k = 50
prune_vals_given = None
prune_vals_number = 10
prune_vals_max = 0.85
prune_vals_offset = 0.01
filter_att = 'GRUPO'
filter_val = 'migraña episódica'
op = op.eq
remove_nodes = [34, 83]
trace = True
trace_k_paths = False
prune_start = 0
case_start = 0
prune_finish = None
case_finish = None

# cargo solo los conectomas que me interesan
connectomes_original, cases = connectomes_filtered(filter_att,
                                                   filter_val,
                                                   op,
                                                   remove_nodes=remove_nodes,
                                                   ret_cases=True)
n_connectomes = len(connectomes_original)
c_size = connectomes_original.shape[1]
if case_finish == None:
    case_finish = n_connectomes
print('Loading {} connectomes'.format(n_connectomes))

if prune_vals_given == None:
    assert prune_vals_max != None, "prune_vals_max should be max prune value"
    assert prune_vals_number != None, "prune_vals_number should be number of prune values"
    assert prune_vals_offset != None, "prune_vals_offset should be offset from lowest prune value"
    # voy a calcular el numero minimo de conexiónes que son 0 para algun conectoma
    con_ref = connectomes_filtered(filter_att,
                                   'control sano',
                                   op,
                                   remove_nodes=remove_nodes,
                                   ret_cases=False)
    nn = con_ref.shape[1]
    # solo me importa la parte superior de la matriz, sin diagonal
    min_connectomes_zeros = np.min(
        (con_ref == 0).sum(axis=2).sum(axis=1)) - nn - (nn * (nn - 1) / 2)
    # el porcentaje lo saco dividiendo por la cantidad total de elementos en la parte superior de la matriz
    min_zero_percentage = min_connectomes_zeros / (nn * (nn - 1) / 2)
    prune_vals = np.linspace(min_zero_percentage - prune_vals_offset,
                             prune_vals_max, prune_vals_number)
    if prune_finish == None:
        prune_finish = prune_vals_number
else:
    prune_vals = prune_vals_given

for i, prune_val in enumerate(prune_vals):
    # les aplico pruning a los conectomas
    connectomes = prune_connectomes(connectomes_original, prune_type,
                                    prune_val)
    current_dir = os.getcwd()
    os.chdir(
        "C:\\Users\Tomas\Desktop\Tesis\programacion\\results\pruning\k_paths_txt\inv_log_migraña_episódica"
    )
    for j, connectome in enumerate(connectomes):
        # mido el tiempo
        start = time.time()
        # creo un grafo del conectoma
        G = nx.from_numpy_matrix(connectome)
        # le agrego un atributo
        add_edge_map(G, map, map_name)
        # longitud, aristas usadas y longitud en aristas

        for k_v in range(k):
            fname = "{}_k={}_{}_prune_val={}_{}.txt".format(
                cases[j], k_v + 1, prune_type, prune_val, map_name)
            f_read = open(fname, "r")
            f_read.close()
        #     write_lines = []

        #     for i, line in enumerate(f_read):
        #         if (i == 0):
        #             write_lines.append(
        #                 'Inicio,Fin,Camino(largo variable),Distancia\n')
        #         if (i > 0):
        #             split = line.split(',')
        #             if (split[2] == 'None\n'):
        #                 write_lines.append(','.join([line.strip(), 'None\n']))
        #             else:
        #                 path = [int(elem) for elem in split[2:]]
        #                 path_len = path_weight(G, path, map_name)
        #                 write_lines.append(','.join(
        #                     [line.strip(), ''.join([str(path_len), '\n'])]))
        #     f_write = open(fname, "w")
        #     f_write.writelines(write_lines)
        # end = time.time()
        # print('Prune val: {:.3f}/{:.3f} - {} - Connectome: {}/{} - {:.5f} s'.
        #       format(prune_val, np.max(prune_vals), cases[j], j + 1,
        #              n_connectomes, end - start))

    os.chdir(current_dir)
