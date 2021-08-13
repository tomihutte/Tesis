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


def inv_log(x):
    return 1 / np.log10(1 + x)


def inv(x):
    return 1 / x


def k_shortest_paths(G, source, target, k, dists_calc=True, weight=None):
    # esto es un generador de caminos, los devuelve del mas corto al mas largo
    gen = nx.shortest_simple_paths(G, source, target, weight)
    # los caminos los voy a guardar en un array de objetos (inicializados como None)
    paths = np.empty(shape=(k), dtype=object)
    # si me piden calcular las distancias tmb
    if dists_calc:
        dists = np.zeros(shape=(k))
    try:
        # recorro el generador
        for i, path in enumerate(gen):
            # agrego el camino
            paths[i] = path
            if dists_calc:
                # agrego la distancia
                dists[i] = path_weight(G, path, weight=weight)
            if i == k - 1:
                # si llegue a k caminos, dejo de recorrerF
                break
        if i < k - 1:
            # si calcule menos que k caminos, aviso
            print("Hay solo {} caminos entre {} y {}".format(i + 1, source, target))
    except nx.NetworkXNoPath:
        # si no estan conectados los nodos tengo que avisar
        print("Nodos {}-{} no conectados".format(source, target))
        dists = np.repeat(np.inf, k)
    # retorno lo calculado
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
                print("{}-{}".format(node + 1, i + 1))
            # obtengo los paths y sus distancias
            if dists:
                k_dists, k_paths = k_shortest_paths(G, node, nodes[i], K, dists, weight)
                k_paths_mat[node, i] = k_paths
                k_dists_mat[node, i] = k_dists
            else:
                k_paths = k_shortest_paths(G, node, nodes[i], K, dists, weight)
                k_paths_mat[node, i] = k_paths
    if dists:
        return k_dists_mat, k_paths_mat
    else:
        return k_paths_mat


def k_shortest_paths_load_txt(
    filter_val, case, k_max, prune_type, prune_val, weight, dists=True, trace=False
):
    n_nodes = 82
    paths = np.empty(shape=(n_nodes, n_nodes, k_max), dtype=object)
    if dists:
        dists_mat = np.zeros(shape=(n_nodes, n_nodes, k_max))
    current_dir = os.getcwd()
    group_str = "_".join(filter_val.split())
    os.chdir(
        "C:\\Users\Tomas\Desktop\Tesis\Programacion\\results\pruning\k_paths_txt\{}_{}".format(
            weight, group_str
        )
    )
    for k_v in range(k_max):
        if trace:
            print("k={}".format(k_v))
        fname = "{}_k={}_{}_prune_val={}_{}.txt".format(
            case, k_v + 1, prune_type, prune_val, weight
        )
        f = open(fname, "r")
        for i, line in enumerate(f):
            if i > 0:
                split = line.split(",")
                i = int(split[0])
                j = int(split[1])
                if split[2] == "None":
                    paths[i, j, k_v] = None
                    if dists:
                        dists_mat[i, j, k_v] = np.inf
                else:
                    paths[i, j, k_v] = [int(elem) for elem in split[2:-1]]
                    if dists:
                        dists_mat[i, j, k_v] = float(split[-1])
                        # dists_mat[i, j,
                        # 			k_v] = path_weight(G, paths[i, j, k_v],
                        # 								weight)
        f.close()
    os.chdir(current_dir)
    if dists:
        return paths, dists_mat
    else:
        return paths


def k_shortest_paths_load_npy(prune_type, prune_val, map_name, group, k, dists=True):
    current_dir = os.getcwd()
    os.chdir(
        "C:\\Users\Tomas\Desktop\Tesis\Programacion\\results\pruning\data\k_paths\{}_{}".format(
            map_name, "_".join(group.split())
        )
    )
    paths_name = "k_paths_saved_k=50_{}_prune_{}_{}.npy".format(
        prune_type, prune_val * 100, map_name
    )
    k_paths = np.load(paths_name, allow_pickle=True)
    if dists:
        dists_name = "k_dists_saved_k=50_{}_prune_{}_{}.npy".format(
            prune_type, prune_val * 100, map_name
        )
        k_dists = np.load(dists_name, allow_pickle=True)
        os.chdir(current_dir)
        return k_paths[:, :, :, :k], k_dists[:, :, :, :k]
    os.chdir(current_dir)
    return k_paths[:, :, :, :k]


def path_edges_length(G, paths, k_vals):
    # esta funcion devuelve una matriz de aristas (edges)
    # donde en cada elemento i,j esta cuantas veces fue usada
    # la arista i,j en los k_shortest_paths para los valores de k presentes en k_vals

    # tambien devuelve una matriz de nodos (lengths) donde en el elemento i,j
    # esta calculada la distancia, medida como numero de aristas, que hay entre el
    # nodo i y el nodo j, en el k_shortest_path para valores de k en k_vals

    # creo las matrices
    edges = np.zeros(shape=(len(k_vals), G.number_of_nodes(), G.number_of_nodes()))
    lengths = np.zeros(shape=(len(k_vals), G.number_of_nodes(), G.number_of_nodes()))
    # recorro todos los pares de nodos
    for row in range(G.number_of_nodes()):
        for col in range(row + 1, G.number_of_nodes()):
            # recorro los valores presentes en k_vals
            for i, k in enumerate(k_vals):
                # me paro en el camino k_esimo
                # ipdb.set_trace()
                if paths[row, col][k - 1] != None:
                    # el numero de aristas es el largo del camino -1
                    lengths[i, row, col] = len(paths[row, col][k - 1]) - 1
                else:
                    # si el camino es None, le pongo longitud inf
                    lengths[i, row, col] = np.inf
                for path in paths[row, col][:k]:
                    if path != None:
                        l = len(path)
                        # para recorrer desde el primer hasta el anteultimo nodo del camino
                        for n in range(l - 1):
                            # aca le sumo uno a la arista que correspone, la que conecta
                            # el nodo n con el nodo n+1 del camino
                            node_1 = path[n]
                            node_2 = path[n + 1]
                            n_min = min(node_1, node_2)
                            n_max = max(node_1, node_2)
                            edges[i, n_min, n_max] += 1
    return edges, lengths


def mean_k_shortest_path_length(G, k_vals, paths, dists):
    # calculo la media de distancia, uso de aristas y distancia en aristas
    # obtengo las distancias en aristas y las aristas usadas
    path_edges, path_lengths = path_edges_length(G, paths, k_vals)
    # numero de nodos y aristas presentes en el grafo
    n = G.number_of_nodes()
    e = G.number_of_edges()
    # saco el porcentaje de aristas que se estan usando (de las que existen)
    edges_percentage = (path_edges != 0).sum(axis=2).sum(axis=1) / e
    # voy a calcular la media de las longitudes
    p_lengths = np.zeros(shape=(len(k_vals)))
    for i in range(len(path_lengths)):
        # hago la media por sobre los valores diferentes de inf y mayores
        # que 0, los que representan nodos conectados
        p_lengths[i] = (
            path_lengths[i][(path_lengths[i] != np.inf) * (path_lengths[i] > 0)]
        ).mean()
    # calculo la media de distancias
    means = np.zeros(shape=len(k_vals))
    # lo hago para cada valor de k elegido
    for i, k in enumerate(k_vals):
        # calculo la media sobre los valores de la parte triangular superior
        # ya que la inferior son cero, no los toque, y no uso los inf
        k_dists = dists[:, :, k - 1][np.triu_indices(n, 1)]
        means[i] = k_dists[k_dists != np.inf].mean()
    return means, p_lengths, edges_percentage


def connectomes_filtered(
    filter_attr,
    filter_value,
    operator,
    all=False,
    ret_cases=False,
    remove_nodes=[34, 83],
):
    # esta funcion devuelve todas las matrices de conexión que cumplan con que
    # operator(filter_attr,filter_value), donde operator es un operador booleano

    # los path para la lista de conectomas y los conectomas (matrices)
    list_path = r"C:\Users\Tomas\Desktop\Tesis\datos\conjunto_datos_conectomica_migranya\lista_de_casos.xlsx"
    connectomes_path = r"C:\Users\Tomas\Desktop\Tesis\datos\conjunto_datos_conectomica_migranya\matrices_conectividad\\"
    # cargo la lista de conectomas a un panda dataframe
    cases = pd.read_excel(list_path)
    # si quiero todos, paso todos
    if all:
        cases = cases["NOMBRE "].values
    # si no, me fijo que conectomas cumplen con la condición
    else:
        # lo paso a minuscula para no tener problema
        cases[filter_attr] = cases[filter_attr].str.lower()
        cases = cases[operator(cases[filter_attr], filter_value)]["NOMBRE "]
        # devuelvo un array con los nombres de los casos
        cases = cases.values
    # ahora voy a cargar los conectomas
    connectomes = np.zeros(shape=(len(cases), 84, 84))
    for i, case in enumerate(cases):
        # contruyo el path del conectoma que quiero cargar
        connectome_path = connectomes_path + case + "_conectomica_fiber_count.csv"
        # cargo el conectoma
        connectome = np.genfromtxt(connectome_path)
        # borro la diagonal y la parte de abajo
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


def connectomes_components(connectomes, cases):
    # calcula por cuantos componentes esta formado el grafo
    # si es mas que uno, quiere decir que esta desconectado
    disconnected = []
    components = np.zeros(shape=len(connectomes))
    max_components_size = np.zeros(shape=len(connectomes))
    for i, connectome in enumerate(connectomes):
        g = nx.from_numpy_matrix(connectome)
        if not (nx.is_connected(g)):
            disconnected.append(cases[i])
        components[i] = nx.number_connected_components(g)
        max_components_size[i] = np.max([len(c) for c in nx.connected_components(g)])
    return disconnected, components, max_components_size


def check_prune_disconnect(
    prune_type,
    prune_vals,
    filter_attr=None,
    filter_val=None,
    operator=None,
    all=True,
    remove_nodes=None,
):
    # calcula cuantos grafos estan desconectados y sus componentes
    # para diferentes valores de prunning
    connectomes, cases = connectomes_filtered(
        filter_attr,
        filter_val,
        operator,
        all=all,
        ret_cases=True,
        remove_nodes=remove_nodes,
    )

    disconnected = []
    components = np.zeros(shape=(len(prune_vals), len(connectomes)))
    max_components = np.zeros(shape=(len(prune_vals), len(connectomes)))
    for i, prune_val in enumerate(prune_vals):
        pruned_connectomes = prune_connectomes(connectomes, prune_type, prune_val)
        disc, comp, max = connectomes_components(pruned_connectomes, cases)
        disconnected.append(disc)
        components[i] = comp
        max_components[i] = max

    return disconnected, components, max_components


def plot(
    x, y, xlabel, ylabel, title, savefig, legends=None, fs=16, lw=3, alpha=1, fmt="-o"
):
    sns.set(style="whitegrid")
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"] = 1.25
    plt.figure(figsize=(8, 6))
    if legends != None:
        for i in range(len(legends)):
            plt.errorbar(
                x,
                np.nanmean(y[i], axis=1),
                yerr=np.nanstd(y[i], axis=1) / np.sqrt(y[i].shape[1]),
                label=legends[i],
                lw=lw,
                alpha=alpha,
                fmt=fmt,
                barsabove=True,
                ecolor="black",
            )
        plt.legend(fontsize=fs)
    else:
        plt.errorbar(
            x,
            y.mean(axis=1),
            yerr=y.std(axis=1) / np.sqrt(y.shape[1]),
            lw=lw,
            alpha=alpha,
            fmt=fmt,
            ecolor="black",
        )
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel(ylabel, fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.title(title, fontsize=fs)
    plt.tight_layout()
    plt.savefig(savefig)
    plt.show()


def prune_connectomes(connectomes_original, prune_type, val):
    # hace prunning a los conectomas
    connectomes = np.copy(connectomes_original)
    # percentage es borrar el porcentage mas bajo de las aristas
    if prune_type == "percentage":
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
    elif prune_type == "threshold":
        mask = connectomes.mean(axis=0)
        mask = mask <= val
        mask = mask[np.newaxis, :, :]
        mask = np.repeat(mask, len(connectomes), axis=0)
        connectomes[mask] = 0
        return connectomes
    # aca borro todas las conexiones cuyo coeficiente de variación
    # sea mas alto que cierto valor
    elif prune_type == "var_coeff":
        mask = connectomes.std(axis=0) / connectomes.mean(axis=0)
        mask[np.isnan(mask)] = 0
        mask = mask >= val
        mask = mask[np.newaxis, :, :]
        mask = np.repeat(mask, len(connectomes), axis=0)
        connectomes[mask] = 0
        return connectomes
    else:
        raise Exception("Invalid prune_type, choose percentage,threshold or var_coeff")
        pass


def prune_vals_calc(
    connectomes, prune_vals_number=10, prune_vals_max=0.85, prune_vals_offset=0.01
):
    # voy a calcular el numero minimo de conexiónes que son 0 para algun conectoma
    nn = connectomes.shape[1]
    # solo me importa la parte superior de la matriz, sin diagonal
    min_connectomes_zeros = (
        np.min((connectomes == 0).sum(axis=2).sum(axis=1)) - nn - (nn * (nn - 1) / 2)
    )
    # el porcentaje lo saco dividiendo por la cantidad total de elementos en la parte superior de la matriz
    min_zero_percentage = min_connectomes_zeros / (nn * (nn - 1) / 2)
    prune_vals = np.linspace(
        min_zero_percentage - prune_vals_offset, prune_vals_max, prune_vals_number
    )
    return prune_vals


def add_edge_map(G, map, map_name):
    # agrego un mapeo a los ejes
    for u, v in G.edges():
        G.edges[u, v][map_name] = map(G.edges[u, v]["weight"])
    pass


def k_shortest_path_save(
    prune_type,
    map,
    map_name,
    k,
    prune_vals_given=None,
    prune_vals_number=None,
    prune_vals_max=None,
    prune_vals_offset=None,
    filter_att="GRUPO",
    filter_val="control sano",
    op=op.eq,
    remove_nodes=[34, 83],
    trace=False,
    trace_k_paths=False,
    prune_start=0,
    prune_finish=None,
    case_start=0,
    case_finish=None,
):
    # cargo solo los conectomas que me interesan
    connectomes_original, cases = connectomes_filtered(
        filter_att, filter_val, op, remove_nodes=remove_nodes, ret_cases=True
    )
    n_connectomes = len(connectomes_original)
    c_size = connectomes_original.shape[1]
    if case_finish == None:
        case_finish = n_connectomes
    print("Loading {} connectomes".format(n_connectomes))

    if prune_vals_given == None:
        assert prune_vals_max != None, "prune_vals_max should be max prune value"
        assert (
            prune_vals_number != None
        ), "prune_vals_number should be number of prune values"
        assert (
            prune_vals_offset != None
        ), "prune_vals_offset should be offset from lowest prune value"
        connectomes_control_sano = connectomes_filtered(
            "GRUPO", "control sano", op, remove_nodes=remove_nodes, ret_cases=False
        )
        # voy a calcular el numero minimo de conexiónes que son 0 para algun conectoma
        nn = connectomes_control_sano.shape[1]
        # solo me importa la parte superior de la matriz, sin diagonal
        min_connectomes_zeros = (
            np.min((connectomes_control_sano == 0).sum(axis=2).sum(axis=1))
            - nn
            - (nn * (nn - 1) / 2)
        )
        # el porcentaje lo saco dividiendo por la cantidad total de elementos en la parte superior de la matriz
        min_zero_percentage = min_connectomes_zeros / (nn * (nn - 1) / 2)
        prune_vals = np.linspace(
            min_zero_percentage - prune_vals_offset, prune_vals_max, prune_vals_number
        )
        if prune_finish == None:
            prune_finish = prune_vals_number
    else:
        prune_vals = prune_vals_given
    start_flag = False
    save_path = "C:\\Users\Tomas\Desktop\Tesis\Programacion\\results\pruning\k_paths_txt\{}_{}".format(
        map_name, "_".join(filter_val.split())
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, prune_val in enumerate(prune_vals):
        # les aplico pruning a los conectomas
        connectomes = prune_connectomes(connectomes_original, prune_type, prune_val)
        for j, connectome in enumerate(connectomes):
            if (i >= prune_start) and ((j >= case_start) | start_flag):
                start_flag = True
                if (i <= prune_finish) and (j <= case_finish):
                    if trace:
                        print(
                            "Prune val: {:.3f}/{:.3f} - {} - Connectome: {}/{} - ".format(
                                prune_val,
                                np.max(prune_vals),
                                cases[j],
                                j + 1,
                                n_connectomes,
                            ),
                            end="",
                        )
                    # mido el tiempo
                    start = time.time()
                    # creo un grafo del conectoma
                    G = nx.from_numpy_matrix(connectome)
                    # le agrego un atributo
                    add_edge_map(G, map, map_name)
                    # longitud, aristas usadas y longitud en aristas
                    paths = global_k_shortest_paths(
                        G, k, trace=trace_k_paths, weight=map_name, dists=False
                    )
                    for k_v in range(k):
                        fname = "{}_k={}_{}_prune_val={}_{}.txt".format(
                            cases[j], k_v + 1, prune_type, prune_val, map_name
                        )
                        fname = os.path.join(save_path, fname)
                        f = open(fname, "w")
                        f.truncate(0)
                        f.write(
                            "Inicio,Fin,Camino(separado por comas),Distancia del camino\n"
                        )
                        for i_idx in range(paths.shape[0]):
                            for j_idx in range(i_idx + 1, paths.shape[1]):
                                if paths[i_idx, j_idx][k_v] != None:
                                    f.write(
                                        "{},{},{},{}\n".format(
                                            i_idx,
                                            j_idx,
                                            ",".join(
                                                [
                                                    str(e)
                                                    for e in paths[i_idx, j_idx][k_v]
                                                ]
                                            ),
                                            path_weight(
                                                G, paths[i_idx, j_idx][k_v], map_name
                                            ),
                                        )
                                    )
                                else:
                                    f.write("{},{},None,None\n".format(i_idx, j_idx))
                        f.close()

                    end = time.time()
                    # printeo el tiempo
                    if trace:
                        print("{:.5f} s".format(end - start))
    pass

