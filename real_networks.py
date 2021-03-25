import networkx as nx
import numpy as np
import time
import ipdb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import operator as op
from networkx.algorithms.flow import shortest_augmenting_path


def path_length(G, path, weight='weight'):
    # esto sirve para calcular la longitud de un camino
    # primero vemos si el grafo es pesado o no
    if nx.is_weighted(G, weight=weight):
        # creamos una lista donde cada elemento son los pesos de la arista
        costs = [G[path[i]][path[i + 1]][weight] for i in range(len(path) - 1)]
        # sumamos
        length = np.sum(costs)
    else:
        # si no es pesado el grafo, es el numero de nodos del camino menos 1
        length = len(path) - 1
    return length


def k_shortest_paths(G, source, target, k, dists_calc=True, weight=None):
    # esto es un generador de caminos, los devuelve del mas corto al mas largo
    gen = nx.shortest_simple_paths(G, source, target, weight)
    paths = np.empty(shape=(k), dtype=object)
    if dists_calc:
        dists = np.zeros(shape=(k))
    try:
        for i, path in enumerate(gen):
            paths[i] = path
            if dists_calc:
                dists[i] = path_length(G, path, weight=weight)
            if i == k - 1:
                break
        if i < k - 1:
            print('Hay solo {} caminos'.format(i + 1))
    except:
        print('Nodos {}-{} no conectados'.format(source, target))
        paths = np.array([[]])
        dists = np.repeat(np.inf, k)
    if dists_calc:
        return dists, paths
    else:
        return paths


def global_k_shortest_paths(G, K, weight=None, trace=False, dists=True):
    # primero creo una lista con todos los nodos para poder recorrerlos
    nodes = list(G.nodes)
    size = len(nodes)
    # vamos a crear una matriz donde guardar los k_paths
    k_paths_mat = np.zeros(shape=(size, size), dtype=object)
    if dists:
        k_dists_mat = np.zeros(shape=(size, size, K))
    # vamos a recorrer todos los nodos, sabiendo cual es en el que estamos
    for index, node in enumerate(nodes):
        # para no calcular dos veces las cosas, solo recorro a partir del siguiente nodo
        for i in range(index + 1, len(nodes)):
            if trace:
                print('{}-{}'.format(node, i))
            # obtengo los paths y sus distancias
            if dists:
                k_dists, k_paths = k_shortest_paths(G, node, nodes[i], K,
                                                    dists, weight)
                k_paths_mat[node, i] = k_paths
                k_dists_mat[node, i] = k_dists
            else:
                k_paths = k_shortest_paths(G, node, nodes[i], K, dists, weight)
                k_paths_mat[node, i] = k_paths
    if dists:
        return k_dists_mat, k_paths_mat
    else:
        return k_paths_mat


def path_edges_length(G, paths, k_vals):
    # esta función me devuelve una matriz con las conexiónes presentes en un camino dado
    # creo la matriz de conexiones
    edges = np.zeros(shape=(len(k_vals), G.number_of_nodes(),
                            G.number_of_nodes()))
    lengths = np.zeros(shape=(len(k_vals), G.number_of_nodes(),
                              G.number_of_nodes()))
    #recorro todos los pares de nodos
    for row in range(G.number_of_nodes()):
        for col in range(row + 1, G.number_of_nodes()):
            # los diferentes caminos entre dos nodos
            for i, k in enumerate(k_vals):
                lengths[i, row, col] = len(paths[row, col][k - 1])
                for path in paths[row, col][:k]:
                    l = len(path)
                    # para recorrer desde el primer hasta el anteultimo nodo del camino
                    for n in range(l - 1):
                        node_1 = path[n]
                        node_2 = path[n + 1]
                        n_min = min(node_1, node_2)
                        n_max = max(node_1, node_2)
                        edges[i, n_min, n_max] += 1
    return edges, lengths


def mean_k_shortest_path_length(G, k_vals, weight=None, trace=False):
    dists, paths = global_k_shortest_paths(G,
                                           np.max(k_vals),
                                           weight=weight,
                                           trace=trace)
    path_edges, path_lenghts = path_edges_length(G, paths, k_vals)
    n = G.number_of_nodes()
    e = G.number_of_edges()
    edges_percentage = (path_edges != 0).sum(axis=2).sum(axis=1) / e
    path_lenghts = path_lenghts.mean(axis=2).mean(axis=1)
    means = np.zeros(shape=len(k_vals))
    for i, k in enumerate(k_vals):
        k_dists = dists[:, :, i][np.triu_indices(n, 1)]
        means[i] = k_dists[k_dists != np.inf].mean()
    return means, path_lenghts, edges_percentage


def connectomes_filtered(filter_attr,
                         filter_value,
                         operator,
                         all=False,
                         ret_cases=False):
    list_path = '/home/tomas/Desktop/Tesis/conjunto_datos_conectomica_migranya/conjunto_datos_conectomica_migranya/lista_de_casos.xlsx'
    connectomes_path = '/home/tomas/Desktop/Tesis/conjunto_datos_conectomica_migranya/conjunto_datos_conectomica_migranya/matrices_conectividad/'
    cases = pd.read_excel(list_path)
    if all:
        cases = cases['NOMBRE '].values
    else:
        cases[filter_attr] = cases[filter_attr].str.lower()
        cases = cases[operator(cases[filter_attr], filter_value)]['NOMBRE ']
        cases = cases.values
    connectomes = np.zeros(shape=(len(cases), 84, 84))
    for i, case in enumerate(cases):
        connectome_path = connectomes_path + case + '_conectomica_fiber_count.csv'
        connectome = np.genfromtxt(connectome_path)
        connectomes[i] = connectome
    if ret_cases:
        return connectomes, cases
    else:
        return connectomes


def connectomes_components(connectomes, cases):
    disconnected = []
    components = np.zeros(shape=len(connectomes))
    max_components_size = np.zeros(shape=len(connectomes))
    for i, connectome in enumerate(connectomes):
        g = nx.from_numpy_matrix(connectome)
        if not (nx.is_connected(g)):
            disconnected.append(cases[i])
        components[i] = nx.number_connected_components(g)
        max_components_size[i] = np.max(
            [len(c) for c in nx.connected_components(g)])
    return disconnected, components, max_components_size


def check_prune_disconnect(prune_type,
                           prune_vals,
                           filter_attr=None,
                           filter_val=None,
                           operator=None,
                           all=True,
                           remove_nodes=None):
    connectomes, cases = connectomes_filtered(filter_attr,
                                              filter_val,
                                              operator,
                                              all=all,
                                              ret_cases=True)

    if remove_nodes != None:
        connectomes = np.delete(connectomes, remove_nodes, axis=1)
        connectomes = np.delete(connectomes, remove_nodes, axis=2)

    disconnected = []
    components = np.zeros(shape=(len(prune_vals), len(connectomes)))
    max_components = np.zeros(shape=(len(prune_vals), len(connectomes)))
    for i, prune_val in enumerate(prune_vals):
        pruned_connectomes = prune_connectomes(connectomes, prune_type,
                                               prune_val)
        disc, comp, max = connectomes_components(pruned_connectomes, cases)
        disconnected.append(disc)
        components[i] = comp
        max_components[i] = max

    return disconnected, components, max_components


def plot(x,
         y,
         xlabel,
         ylabel,
         title,
         savefig,
         legends=None,
         fs=16,
         lw=3,
         alpha=1,
         fmt='-o'):
    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 6))
    if legends != None:
        for i in range(len(legends)):
            plt.errorbar(x[i],
                         y[i].mean(axis=1),
                         yerr=y[i].std(axis=1) / np.sqrt(y[i].shape[1]),
                         label=legends[i],
                         lw=lw,
                         alpha=alpha,
                         fmt=fmt)
        plt.legend(fontsize=fs)
    else:
        plt.plot(x,
                 y.mean(axis=1),
                 yerr=y.std(axis=1) / np.sqrt(y.shape[1]),
                 lw=lw,
                 alpha=alpha,
                 fmt=fmt)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel(ylabel, fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.title(title, fontsize=fs)
    plt.tight_layout()
    plt.savefig(savefig)
    plt.show()


def prune_connectomes(connectomes_original, prune_type, val):
    connectomes = np.copy(connectomes_original)
    if prune_type == 'percentage':
        for connectome in connectomes:
            mat = np.sort(connectome[np.triu_indices(connectome.shape[0], 1)])
            n_elements = len(mat)
            percentile_value = mat[int(val * (n_elements - 1))]
            connectome[connectome <= percentile_value] = 0
        return connectomes
    elif prune_type == 'threshold':
        mask = connectomes.mean(axis=0)
        mask = mask <= val
        mask = mask[np.newaxis, :, :]
        mask = np.repeat(mask, len(connectomes), axis=0)
        connectomes[mask] = 0
        return connectomes
    elif prune_type == 'var_coeff':
        # ipdb.set_trace()
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
    for u, v in G.edges():
        G.edges[u, v][map_name] = map(G.edges[u, v]['weight'])
    pass


def pruned_measures(prune_type,
                    map,
                    map_name,
                    k_vals,
                    prune_vals_given=None,
                    prune_vals_number=None,
                    prune_vals_max=None,
                    prune_vals_offset=None,
                    filter_att='GRUPO',
                    filter_val='control sano',
                    op=op.eq,
                    remove_nodes=[34, 83],
                    trace=False,
                    trace_k_paths=False,
                    only_plot=False):
    # cargo solo los conectomas que me interesan
    connectomes_original = connectomes_filtered(filter_att, filter_val, op)
    n_connectomes = len(connectomes_original)

    print('Loading {} connectomes'.format(n_connectomes))

    # borro algunos nodos si es necesario
    if remove_nodes != None:
        connectomes_original = np.delete(connectomes_original,
                                         remove_nodes,
                                         axis=1)
        connectomes_original = np.delete(connectomes_original,
                                         remove_nodes,
                                         axis=2)
    if prune_vals_given == None:
        # voy a calcular el numero minimo de conexiónes que son 0 para algun conectoma
        nn = connectomes_original.shape[1]
        # solo me importa la parte superior de la matriz, sin diagonal
        min_connectomes_zeros = np.min(
            (connectomes_original
             == 0).sum(axis=2).sum(axis=1)) - nn - (nn * (nn - 1) / 2)
        # el porcentaje lo saco dividiendo por la cantidad total de elementos en la parte superior de la matriz
        min_zero_percentage = min_connectomes_zeros / (nn * (nn - 1) / 2)
        prune_vals = np.linspace(min_zero_percentage - prune_vals_offset,
                                 prune_vals_max, prune_vals_number)
    else:
        prune_vals = prune_vals_given

    if not (only_plot):
        #creo los arrays donde vamos a guardar las cosas
        sh_paths = np.zeros(shape=(prune_vals_number, n_connectomes))
        k_paths = np.zeros(shape=(len(k_vals), prune_vals_number,
                                  n_connectomes))
        k_lengths = np.zeros(shape=(len(k_vals), prune_vals_number,
                                    n_connectomes))
        k_edges = np.zeros(shape=(len(k_vals), prune_vals_number,
                                  n_connectomes))
        clustering = np.zeros(shape=(prune_vals_number, n_connectomes))
        edge_connectivity = np.zeros(shape=(prune_vals_number, n_connectomes))

        # ahora recorro todos los prune val y los conetomas
        for i, prune_val in enumerate(prune_vals):
            # les aplico pruning a los conectomas
            connectomes = prune_connectomes(connectomes_original, prune_type,
                                            prune_val)
            for j, connectome in enumerate(connectomes):
                # mido el tiempo
                start = time.time()
                # creo un grafo del conectoma
                G = nx.from_numpy_matrix(connectome)
                # le agrego un atributo
                add_edge_map(G, map, map_name)
                # clustering pesado
                clustering[i, j] = nx.average_clustering(G, weight='weight')
                # sh paths siguiendo el atributo que agregue
                sh_paths[i,
                         j] = nx.average_shortest_path_length(G,
                                                              weight=map_name)
                # edge connectivity
                edge_connectivity[i, j] = nx.edge_connectivity(
                    G, flow_func=shortest_augmenting_path)
                # longitud, aristas usadas y longitud en aristas
                path_means, e_percent, path_lengths = mean_k_shortest_path_length(
                    G, k_vals, weight=map_name, trace=trace_k_paths)

                k_paths[:, i, j] = path_means
                k_lengths[:, i, j] = path_lengths
                k_edges[:, i, j] = e_percent

                end = time.time()
                # printeo el tiempo
                if trace:
                    print(
                        'Prune val: {:3f}/{:3f} - Connectome:{}/{} - {:5f} s'.
                        format(prune_val, np.max(prune_vals), j + 1,
                               n_connectomes, end - start))

        np.save(
            'sh_path_{}_prune_{:0f}_{:0f}_{}'.format(prune_type,
                                                     np.min(prune_vals) * 100,
                                                     np.max(prune_vals) * 100,
                                                     map_name), sh_paths)
        np.save(
            'clustering_{}_prune_{:0f}_{:0f}_{}'.format(
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100, map_name), clustering)
        np.save(
            'edge_conn_{}_prune_{:0f}_{:0f}_{}'.format(
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100, map_name), edge_connectivity)
        np.save(
            'k_paths_{}_{}_prune_{:0f}_{:0f}_{}'.format(
                k_vals, prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100, map_name), k_paths)

        np.save(
            'k_lengths_{}_{}_prune_{:0f}_{:0f}_{}'.format(
                k_vals, prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100, map_name), k_lengths)

        np.save(
            'k_edges_{}_{}_prune_{:0f}_{:0f}_{}'.format(
                k_vals, prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100, map_name), k_edges)

    else:
        sh_paths = np.load('sh_path_{}_prune_{:0f}_{:0f}_{}.npy'.format(
            prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100, map_name))

        clustering = np.load('clustering_{}_prune_{:0f}_{:0f}_{}.npy'.format(
            prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100, map_name))

        edge_connectivity = np.load(
            'edge_conn_{}_prune_{:0f}_{:0f}_{}.npy'.format(
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100, map_name))

        k_paths = np.load('k_paths_{}_{}_prune_{:0f}_{:0f}_{}.npy'.format(
            k_vals, prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100, map_name))

        k_lengths = np.load('k_lengths_{}_{}_prune_{:0f}_{:0f}_{}.npy'.format(
            k_vals, prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100, map_name))

        k_edges = np.load('k_edges_{}_{}_prune_{:0f}_{:0f}_{}.npy'.format(
            k_vals, prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100, map_name))

    if map_name == 'inv_log':
        map_str = '1/log(1+w)'
    elif map_name == 'inv':
        map_str = '1/w'

    plot(
        prune_vals * 100, sh_paths, 'Aristas eliminadas [%]', 'Longitud media',
        r'Shortest path - Pacientes sanos - $f\left(w\right)={}',
        'sh_path_{}_prune_{:0f}_{:0f}_{}.pdf'.format(prune_type,
                                                     np.min(prune_vals) * 100,
                                                     np.max(prune_vals) * 100,
                                                     map_name))

    plot(
        prune_vals * 100, clustering, 'Aristas eliminadas [%]',
        'Coeficiente de clustering medio',
        'Clustering coefficient - Pacientes sanos',
        'clustering_{}_prune_{:0f}_{:0f}_{}.pdf'.format(
            prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100, map_name))

    plot(
        prune_vals * 100, edge_connectivity, 'Aristas eliminadas [%]',
        'Edge connectivity ', 'Edge connectivity - Pacientes sanos',
        'edge_conn_{}_prune_{:0f}_{:0f}_{}.pdf'.format(
            prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100, map_name))

    k_legends = [r'$k={}$'.format(k) for k in k_vals]

    plot(prune_vals * 100,
         k_paths,
         'Aristas eliminadas [%]',
         'Longitud media',
         'K shortest paths - Pacientes sanos',
         'k_paths_{}_{}_prune_{:0f}_{:0f}_{}.pdf'.format(
             k_vals, prune_type,
             np.min(prune_vals) * 100,
             np.max(prune_vals) * 100, map_name),
         legends=k_legends)

    plot(prune_vals * 100,
         k_lengths,
         'Aristas eliminadas [%]',
         'Numero de aristas medio',
         'K shortest paths - Pacientes sanos',
         'k_lengths_{}_{}_prune_{:0f}_{:0f}_{}.pdf'.format(
             k_vals, prune_type,
             np.min(prune_vals) * 100,
             np.max(prune_vals) * 100, map_name),
         legends=k_legends)

    plot(prune_vals * 100,
         k_edges,
         'Aristas eliminadas [%]',
         'Valor medio de aristas usadas [%]',
         'K shortest paths - Pacientes sanos',
         'k_edges_{}_{}_prune_{:0f}_{:0f}_{}.pdf'.format(
             k_vals, prune_type,
             np.min(prune_vals) * 100,
             np.max(prune_vals) * 100, map_name),
         legends=k_legends)

    return sh_paths, clustering, edge_connectivity, k_paths, k_lengths, k_edges, prune_vals


sh_paths, clustering, edge_connectivity, k_paths, k_lengths, k_edges, prune_vals = pruned_measures(
    'percentage',
    map=lambda x: 1 / x,
    map_name='inv',
    prune_vals_number=10,
    prune_vals_max=0.85,
    prune_vals_offset=0.01,
    k_vals=[1, 2],
    trace=True)

# np.save('sh_path_percentage_prune_37_85_inv_log', sh_paths)
# np.save('clustering_percentage_prune_37_85_inv_log', clustering)
# np.save('edge_conn_percentage_prune_37_85_inv_log', edge_connectivity)
# np.save('k_paths_percentage_prune_37_85_inv_log', k_paths)

# sh_paths = np.load('sh_path_percentage_prune_37_85_inv_log.npy')
# clustering = np.load('clustering_percentage_prune_37_85_inv_log.npy')
# edge_connectivity = np.load('edge_conn_percentage_prune_37_85_inv_log.npy')
# k_paths = np.load('k_paths_percentage_prune_37_85_inv_log.npy')

# plot(np.linspace(0, 85, 100),
#      (edge_connectivity.mean(axis=-1) - edge_connectivity[0].mean()) /
#      edge_connectivity[0].mean(), 'Aristas eliminadas [%]', 'Variación [%]',
#      'Edge connectivity - Pacientes sanos',
#      'edge_connectivity_prune_0_85_inv_log.pdf')

# map_name = '1/log(1+w)'
# map = lambda x: 1 / np.log10(1 + x)
# prune_type = 'var_coeff'
# prune_vals = 10**np.linspace(1, -1, 10)
# k_vals = np.array([2, 5, 10])
# trace = True

# filter_atts = ['control sano', 'migraña crónica', 'migraña episódica']
# legends = ['Control sano', 'Migraña crónica', 'Migraña episódica']
# discs = []
# lens = []
# for i, filter_att in enumerate(filter_atts):
#     discs.append(
#         check_prune_disconnect(prune_type='percentage',
#                                prune_vals=np.linspace(0, 1, 100),
#                                filter_attr='GRUPO',
#                                filter_val=filter_att,
#                                operator=op.eq,
#                                all=False,
#                                remove_nodes=[34, 83]))
#     lens.append([len(disc) for disc in discs[i]])
# plot(x=np.tile(np.linspace(0, 100, 100), 3).reshape(3, 100)[:, 70:],
#      y=np.array(lens)[:, 70:],
#      xlabel='Porcentaje de aristas totales eliminadas',
#      ylabel='Número de conectomas desconectados',
#      title='',
#      savefig='disconnected_connectomes_70_up.pdf',
#  legends=legends)

# filter_atts = ['control sano', 'migraña crónica', 'migraña episódica']
# legends = ['Control sano', 'Migraña crónica', 'Migraña episódica']
# discs = []
# comps = np.zeros(shape=(len(legends), 100))
# lens = []
# for i, filter_att in enumerate(filter_atts):
#     disc, comp = check_prune_disconnect(prune_type='percentage',
#                                         prune_vals=np.linspace(0, 1, 100),
#                                         filter_attr='GRUPO',
#                                         filter_val=filter_att,
#                                         operator=op.eq,
#                                         all=False,
#                                         remove_nodes=[34, 83])
#     comps[i] = comp.mean(axis=-1)
#     discs.append(disc)
#     lens.append([len(disc) for disc in discs[i]])

# plot(x=np.tile(np.linspace(0, 100, 100), 3).reshape(3, 100)[:, 70:],
#      y=comps[:, 70:],
#      xlabel='Porcentaje de aristas totales eliminadas',
#      ylabel='Numero medio de componentes',
#      title='',
#      savefig='components_connectomes_70_up.pdf',
#      legends=legends)
# def plot_prunned_evolution(save, remove_nodes=[34, 83], offset=0):

# remove_nodes = [34, 83]
# offset = 0

# if remove_nodes == None:
#     title = 'Conectomas completos'
# else:
#     l = remove_nodes[:-1]
#     last_val = remove_nodes[-1]
#     l += (['y', last_val])
#     title = 'Removiendo nodos {}'.format(' '.join([str(elem) for elem in l]))
# filter_atts = ['control sano', 'migraña crónica', 'migraña episódica']
# legends = ['Control sano', 'Migraña crónica', 'Migraña episódica']
# discs = []
# comps = np.zeros(shape=(len(legends), 100))
# maxs = np.zeros(shape=(len(legends), 100))
# lens = []
# for i, filter_att in enumerate(filter_atts):
#     disc, comp, max = check_prune_disconnect(prune_type='percentage',
#                                              prune_vals=np.linspace(0, 1, 100),
#                                              filter_attr='GRUPO',
#                                              filter_val=filter_att,
#                                              operator=op.eq,
#                                              all=False,
#                                              remove_nodes=remove_nodes)
#     comps[i] = comp.mean(axis=-1)
#     maxs[i] = max.mean(axis=-1) / np.max(max)
#     discs.append(disc)
#     lens.append([len(disc) for disc in discs[i]])

# plot(x=np.tile(np.linspace(0, 100, 100), 3).reshape(3, 100)[:, offset:],
#      y=maxs[:, offset:],
#      xlabel='Porcentaje de aristas totales eliminadas',
#      ylabel='Tamaño medio de la máxima componente (norm.)',
#      title=title,
#      savefig='max_component_size_mean_{}_up.pdf'.format(offset),
#      legends=legends,
#      marker=None)

# plot(np.linspace(0, 85, 100),
#      (sh_paths.mean(axis=-1) - sh_paths[0].mean()) / sh_paths[0].mean(),
#      'Aristas eliminadas [%]', 'Variación [%]',
#      r'Shortest path - Pacientes sanos - $f\left(w\right)=1/log(1+w)$',
#      'sh_path_percentage_prune_0_85_inv_log.pdf')

# plot(np.linspace(0, 85, 100),
#      (clustering.mean(axis=-1) - clustering[0].mean()) / clustering[0].mean(),
#      'Aristas eliminadas [%]', 'Variación [%]',
#      'Clustering coefficient - Pacientes sanos',
#      'clustering_percentage_prune_0_85_inv_log.pdf')

# plot(np.linspace(0, 85, 100),
#      (edge_connectivity.mean(axis=-1) - edge_connectivity[0].mean()) /
#      edge_connectivity[0].mean(), 'Aristas eliminadas [%]', 'Variación [%]',
#      'Edge connectivity - Pacientes sanos',
#      'edge_connectivity_prune_0_85_inv_log.pdf')

# np.save('sh_path_percentage_prune_0_85_inv_log', sh_paths)
# np.save('clustering_percentage_prune_0_85_inv_log', clustering)
