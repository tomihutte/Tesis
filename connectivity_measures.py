from real_networks import *
import scipy as sp


# Calcula la probabilidad de un camino teniendo en cuenta los pesos del conectoma
# Argumentos:
# G: NetworkX graph asociado al conectoma
# weight_matrix: array de (n,n) con los pesos de las conexiones
# path: array o lista unidimensional que tiene los nodos del camino
# weight: por si se quiere usar otro parametro como peso que el por defecto
# Devuelve:
# numero que indica la probabilidad de ese camino
def path_probability(G, weight_matrix, path, weight="weight"):
    # la probabilidad de cada salto es el peso hacia el siguiente nodo sobre la suma de todos los pesos
    # del nodo (su grado pesado)
    probs = [
        weight_matrix[path[i], path[i + 1]] / G.degree(path[i], weight=weight)
        for i in range(len(path) - 1)
    ]
    # la probabilidad del camino es el producto de las proobabilidades de cada uno de sus saltos
    return np.prod(probs)


# Calcula la distancia entre pares de nodos de conectomas pesando la distancia de los k caminos mas cortos por su probabilidad
# Argumentos:
# connectomas: array de (m,n,n) de los m conectomas (matrices de peso) con n nodos cada uno
# paths: array de (m,n,n,k) con los k caminos mas cortos para cada uno de los pares de nodos de los m conectomas
# dists: array de (m,n,n,k) con la distancia de los k caminos mas cortos especificados en paths
# Devuelve:
# D_k: array de (m,n,n) con la distancia entre cada uno de los n(n-1)/2 pares de nodos de los m conectomas
def k_shortest_paths_length(connectomes, paths, dists, trace=False):
    # la probabilidad de un camino no es simetrica (i->j != i<-j)
    # probabilidad de ida y de vuelta
    paths_prob_a = np.zeros(shape=paths.shape)
    paths_prob_b = np.zeros(shape=paths.shape)
    if trace:
        nn = connectomes.shape[0]
        total_start = time.time()
        print("D_k")
    for case, connectome in enumerate(connectomes):
        if trace:
            start = time.time()
            print("Caso {}/{}".format(case + 1, nn), end="")
        # creamos el grafo
        G = nx.from_numpy_matrix(connectome)
        # la matriz de pesos es simétrica
        connectome += connectome.T
        for node_i in range(paths.shape[1]):
            for node_j in range(node_i + 1, paths.shape[2]):
                for k in range(paths.shape[3]):
                    # cargo el camino para sacar su probabilidad
                    path = paths[case, node_i, node_j, k]
                    if path is None:
                        break
                    else:
                        # calculo probabilidad de ida
                        paths_prob_a[case, node_i, node_j, k] = path_probability(
                            G, connectome, path
                        )
                        # revierto el orden del camino
                        path.reverse()
                        # probabilidad de vuelta
                        paths_prob_b[case, node_i, node_j, k] = path_probability(
                            G, connectome, path
                        )
        if trace:
            print(" - {:.3f} s".format(time.time() - start))
    # normalizo las probabilidades por la suma para todos los k caminos
    p_sum_a = paths_prob_a.sum(axis=-1)[:, :, :, np.newaxis]
    p_sum_b = paths_prob_b.sum(axis=-1)[:, :, :, np.newaxis]
    paths_prob_a = np.divide(
        paths_prob_a, p_sum_a, out=np.zeros_like(paths_prob_a), where=p_sum_a != 0
    )
    paths_prob_b = np.divide(
        paths_prob_b, p_sum_b, out=np.zeros_like(paths_prob_b), where=p_sum_b != 0
    )
    # simetrizo la probabilidad (promedio entre ida y vuelta)
    paths_prob = (paths_prob_a + paths_prob_b) / 2
    # multiplico cada camino por su probabilidad normalizada y obtendo la distancia
    # entre los dos nodos teniendo en cuenta los k caminos mas cortos
    D_k = np.multiply(
        paths_prob, dists, out=np.zeros(shape=paths_prob.shape), where=dists != np.inf,
    ).sum(axis=-1)
    if trace:
        print("Total D_k run time: {:.3f}".format(time.time() - total_start))
    return D_k


# Calcula la centralidad de aristas y nodos en los k caminos mas cortos del conectoma
# Argumentos
# paths: array de (m,n,n,k) con los k caminos mas cortos para los n(n-1)/2 pares de nodos de los m conectomas
# Devuelve:
# nodes: array de (m,n) con la centralidad de los n nodos de cada uno de los m conectomas
# edges: array de (m,n,n) con la centralidad de las n(n-1)/2 aristas de cada uno de los m conectomas
def k_nodes_edges_centrality(paths, trace=False):
    nodes = np.zeros(shape=paths.shape[:2])  # shape = conectomas x nodos
    edges = np.zeros(
        shape=paths.shape[:3]
    )  # shape =  conectomas x nodos x nodos (conectomas x aristas)
    n_cases = paths.shape[0]
    n_nodes = paths.shape[1]
    n_k_vals = paths.shape[-1]
    if trace:
        total_start = time.time()
        print("Nodes-Edges centrality")
    for case in range(n_cases):
        if trace:
            start = time.time()
            print("Caso {}/{}".format(case + 1, n_cases), end="")
        for node_i in range(n_nodes):
            for node_j in range(node_i + 1, n_nodes):
                for k in range(n_k_vals):
                    path = paths[case, node_i, node_j, k]
                    if path is None:
                        break
                    else:
                        # hago una mascara con los nodos que aparecen en el camino
                        nodes_mask = np.array(paths[case, node_i, node_j, k])
                        # ipdb.set_trace()
                        # una mascara con las aristas del camino [[nodo_1,nodo_2],[nodo_2,nodo_3],..,[nodo_n-1,nodo_n]]
                        edges_mask = np.column_stack((nodes_mask[:-1], nodes_mask[1:]))
                        # sumo 1 a los nodos y aristas que aparecen en el camino
                        nodes[case, nodes_mask] += 1
                        edges[case, edges_mask[:, 0], edges_mask[:, 1]] += 1
        if trace:
            print(" - {:.3f} s".format(time.time() - start))
    # normalizo por la cantidad total de caminos que hay
    norm_val = (n_k_vals * n_nodes * (n_nodes - 1)) / 2
    nodes /= norm_val
    edges = np.triu(edges + np.transpose(edges, axes=(0, 2, 1))) / norm_val
    if trace:
        print("Total centrality run time: {:.3f}".format(time.time() - total_start))
    return nodes, edges


# Calcula la medida search information.
# Argumentos:
# connectomes: np.array de (m,n,n) donde m es la cantidad de conectomas y n es la cantidad de nodos
# paths: np.array de (m,n,n,k) donde k es la cantidad de caminos mas cortos a tener en cuenta
# k_vals: np.array de (l) donde l es la cantidad de valores para los que se busca calcular la medida
# trace: bool para imprimir o no el avance del calculo
# Devuelve:
# S: np.array de (l,m,n,n) donde esta calculada la medida para los diferentes l valores especificados en k_vals,
# paara los m conectomas y sus n(n-1)/2 pares de nodos
def search_information(connectomes, paths, k_vals=[1, 2, 5, 10, 20, 50], trace=False):
    paths_prob_a = np.zeros(shape=paths.shape)
    paths_prob_b = np.zeros(shape=paths.shape)
    nn = connectomes.shape[0]
    if trace:
        total_start = time.time()
        print("Search information")
    # recorremos cada conectoma
    for case, connectome in enumerate(connectomes):
        if trace:
            start = time.time()
            print("Caso {}/{}".format(case + 1, nn), end="")
        G = nx.from_numpy_matrix(connectome)
        # hacemos simetricos las matrices adyacencia
        connectome += connectome.T
        # recorremos cada par de nodos
        for node_i in range(paths.shape[1]):
            for node_j in range(node_i + 1, paths.shape[2]):
                # recorremos cada uno de los caminos
                for k in range(paths.shape[3]):
                    path = paths[case, node_i, node_j, k]
                    if path is None:
                        break
                    else:
                        paths_prob_a[case, node_i, node_j, k] = path_probability(
                            G, connectome, path
                        )
                        path.reverse()
                        paths_prob_b[case, node_i, node_j, k] = path_probability(
                            G, connectome, path
                        )
        if trace:
            print(" - {:.3f} s".format(time.time() - start))
    S = np.zeros(shape=((len(k_vals),) + connectomes.shape))
    # ahora sacamos la search information para cada uno de los valores de k especifciados
    for index, k_val in enumerate(k_vals):
        p_a = paths_prob_a[..., :k_val]
        p_b = paths_prob_b[..., :k_val]
        p_a = p_a.sum(axis=-1)
        p_b = p_b.sum(axis=-1)
        # evitamos dividir por cero
        S[index] = (
            -np.log2(p_a, out=np.zeros_like(p_a), where=p_a != 0)
            - np.log2(p_b, out=np.zeros_like(p_b), where=p_b != 0)
        ) / 2
    if trace:
        print(
            "Total search information run time: {:.3f}".format(
                time.time() - total_start
            )
        )
    return S


# Calcula el matching index para cada par de nodos del grafo indicado
# Argumentos:
# G: NetworkX graph, grafo que contiene los nodos
# weight_matrix: array (n,n), matriz de peso asociada al grafo G
# Devuelve:
# M: array (n,n), matriz triangular superior con cada uno de los valores de matching index para los pares de nodos del grafo
def matching_index(G, weight_matrix):
    M = np.zeros(shape=weight_matrix.shape)
    n_nodes = G.number_of_nodes()
    for node_i in range(n_nodes):
        for node_j in range(node_i + 1, n_nodes):
            numerator = np.sum(
                [
                    weight_matrix[node_i, node_k] + weight_matrix[node_k, node_j]
                    for node_k in nx.common_neighbors(G, node_i, node_j)
                ]
            )
            denominator = (
                G.degree(node_i, weight="weight")
                + G.degree(node_j, weight="weight")
                - 2 * weight_matrix[node_i, node_j]
            )
            M[node_i, node_j] = numerator / denominator
    return M


# Calcula la medida path transitivity de un grafo
# Argumentos:
# connectomes: array de (m,n,n), matriz de peso de los m conectomas con sus n nodos
# paths: array de (m,n,n,k), contiene los k caminos mas cortos entre cada uno de los pares de nodos de los m conectomas
# Devuelve
# path_transitiviy_weighted: array de (m,n,n), path transitiviy para cada uno de los pares de nodos de los m conectomas, teniendo en cuenta los k caminos mas cortos pesados por su probabilidad
def k_path_transitivity(connectomes, paths, trace=False):
    n_nodes = connectomes.shape[1]
    paths_prob_a = np.zeros(shape=paths.shape)
    paths_prob_b = np.zeros(shape=paths.shape)
    paths_transitivy = np.zeros(shape=paths.shape)
    nn = connectomes.shape[0]
    if trace:
        total_start = time.time()
        print("Path Transitivity")
    for case, connectome in enumerate(connectomes):
        G = nx.from_numpy_matrix(connectome)
        # simetrizamos la matriz de adyacencia
        connectome += connectome.T
        M = matching_index(G, connectome)
        if trace:
            start = time.time()
            print("Caso {}/{}".format(case + 1, nn), end="")
        for node_i in range(n_nodes):
            for node_j in range(node_i + 1, n_nodes):
                for k in range(paths.shape[3]):
                    path = paths[case, node_i, node_j, k]
                    if path is None:
                        break
                    else:
                        paths_prob_a[case, node_i, node_j, k] = path_probability(
                            G, connectome, path
                        )
                        path.reverse()
                        paths_prob_b[case, node_i, node_j, k] = path_probability(
                            G, connectome, path
                        )
                        omega = len(path)
                        matching_indexes = [
                            M[path[i], path[i + 1]] for i in range(omega - 1)
                        ]
                        paths_transitivy[case, node_i, node_j, k] = (
                            2 * np.sum(matching_indexes) / (omega * (omega - 1))
                        )
        if trace:
            print(" - {:.3f} s".format(time.time() - start))
    # tomo la probabilidad total como la media de la de ida y vuelta
    paths_prob = (paths_prob_a + paths_prob_b) / 2
    # normalizo las probabilidades porque las estoy usando como peso
    p_sum = paths_prob.sum(axis=-1)[..., np.newaxis]
    paths_prob = np.divide(
        paths_prob, p_sum, out=np.zeros_like(paths_prob), where=p_sum != 0
    )
    # aca tengo a transitividad pesada
    paths_transitivy_weighted = (paths_transitivy * paths_prob).sum(axis=-1)

    if trace:
        print(
            "Total path transitivity run time: {:.3f}".format(time.time() - total_start)
        )

    return paths_transitivy_weighted


# Calcula la routing efficiency los conectomas indicados
# Argumentos:
# connectomes: array de (m,n,n) con los m conectomas de n nodos
# Devuelve
# routing_efficiency: array de (m) con la eciciencia de cada uno de los m conectomas.
def routing_efficiency(connectomes, trace=False):
    n_connectomes = connectomes.shape[0]
    routing_efficiency = np.zeros(n_connectomes)
    if trace:
        total_start = time.time()
        print("Routing efficiency")
    for nn, connectome in enumerate(connectomes):
        g = nx.from_numpy_matrix(connectome)
        routing_efficiency[nn] = nx.global_efficiency(g)
    if trace:
        print(
            "Total routing efficiency run time: {:.3f}".format(
                time.time() - total_start
            )
        )
    return routing_efficiency


# Calcula la difussion efficiency para los conetcomas indicados
# Argumentos:
# connectomes: array de (m,n,n), matrices de peso de los m conectomas
# Devuelve
# diff_eff: array (m), array con los valores de difussion efficiency para todos los m conectomas
def diffussion_efficiency(connectomes, trace=False):
    X = np.zeros(shape=connectomes.shape)
    n_nodes = connectomes.shape[1]
    if trace:
        total_start = time.time()
        print("Diffusion Efficiency")
        n_conectomes = connectomes.shape[0]
    for case, W in enumerate(connectomes):
        if trace:
            start = time.time()
            print("Caso {}/{}".format(case + 1, n_conectomes), end="")
        g = nx.from_numpy_matrix(W)
        S = np.diag(np.array(g.degree(weight="weight"))[:, 1])
        U = W.dot(np.linalg.inv(S))
        for node_j in range(n_nodes):
            U_j = np.copy(U)
            U_j[:, node_j] = 0
            C = np.linalg.inv(np.identity(n_nodes) - U_j)
            X[case, :, node_j] = C.sum(axis=0)
        if trace:
            print(" - {:.3f} s".format(time.time() - start))
    diff_eff = np.divide(1, X, out=np.zeros_like(X), where=X != 0)
    diff_eff = diff_eff.sum(axis=-1).sum(axis=-1) / (n_nodes * (n_nodes - 1))
    if trace:
        print(
            "Total diffusion efficiency run time: {:.3f}".format(
                time.time() - total_start
            )
        )
    return diff_eff


# Calcula la communicability de las redes dadas
# Argumentos:
# connectomes: array de (m,n,n), matrices de peso triangulares de las m redes
# Devuelve:
# C: array de (m,n,n), matrices de comunicabilidad de las m redes
def communicability(connectomes, trace=False):
    # donde guardo la matriz de communicability
    C = np.zeros_like(connectomes)
    if trace:
        total_start = time.time()
        print("Communicability")
        n_conectomes = connectomes.shape[0]
    for case, W in enumerate(connectomes):
        if trace:
            start = time.time()
            print("Caso {}/{}".format(case + 1, n_conectomes), end="")
        # la matriz de pesos es simétrica
        W += W.T
        # creamos el grafo
        g = nx.from_numpy_matrix(W)
        # creamos matriz diagonal de grados de nodos
        K = np.diag(np.array(g.degree(weight="weight"))[:, 1])
        # esta matrix la usamos para normalizar W
        K1 = sp.linalg.fractional_matrix_power(K, -1 / 2)
        # normalizmos la matriz de acuerdo al grado de los nodos
        W1 = np.linalg.multi_dot([K1, W, K1])
        # autovalores y autovectores
        D, U = np.linalg.eigh(W1)
        # diagonal de exponencial de autovalores
        eD = np.diag(np.exp(D))
        C[case] = U.dot(eD.dot(np.linalg.inv(U)))
        if trace:
            print(" - {:.3f} s".format(time.time() - start))
    if trace:
        print(
            "Total communicability run time: {:.3f}".format(time.time() - total_start)
        )
    return C


# Calcula el segundo y mayor autovalor de las matrices de acoplamiento de las redes dadas
# Argumentos:
# connectomes: array de (m,n,n), matrices de peso triangulares de las m redes
# Devuelve:
# C: array de (m,2), los dos autovalores de cada red
def coupling_matrix_eigenvals(connectomes, trace=False):
    # donde guardo lambda_2 y lambda_n
    E = np.zeros(shape=(connectomes.shape[0], 2))
    if trace:
        total_start = time.time()
        print("Coupling matrix eigenvals")
        n_conectomes = connectomes.shape[0]
    for case, W in enumerate(connectomes):
        if trace:
            start = time.time()
            print("Caso {}/{}".format(case + 1, n_conectomes), end="")
        # creo un grafo de acuerdo a la matriz de pesos
        g = nx.from_numpy_matrix(W)
        # matriz con los grados de cada nodo en la diagonal y nada mas
        D = np.diag(np.array(g.degree(weight="weight"))[:, 1])
        # matriz de acoplamiento
        C = -(W + W.T) + D
        # autovalores y autovectores
        D, _ = np.linalg.eigh(C)
        E[case] = np.array([D[1], D[-1]])
        if trace:
            print(" - {:.3f} s".format(time.time() - start))
    if trace:
        print(
            "Total coupling matrix eigenvals run time: {:.3f}".format(
                time.time() - total_start
            )
        )
    return E


if __name__ == "__main__":

    k = 50
    maps_name = ["inv_log", "inv"]
    groups = ["control sano", "migraña crónica", "migraña episódica"]

    prune_type = "percentage"
    connectomes_control_sano = connectomes_filtered("GRUPO", "control sano", op.eq)
    prune_vals = prune_vals_calc(connectomes_control_sano)
    prune_val = prune_vals[-4]
    k_vals = [1, 2, 5, 10, 20, 50]

    for map_name in maps_name:
        print("Map: {}".format(map_name))
        for group in groups:
            print("Group: {}".format(group))
            if map_name == "inv_log" and group != "migraña episódica":
                continue
            connectomes_original = connectomes_filtered("GRUPO", group, op.eq)
            connectomes = prune_connectomes(connectomes_original, prune_type, prune_val)
            paths, dists = k_shortest_paths_load_npy(
                prune_type, prune_val, map_name, group, k, dists=True
            )

            # paths = paths[60:]
            # dists = dists[60:]
            # connectomes = connectomes[60:]
            # connectomes = np.array([connectomes[60]])

            current_dir = os.getcwd()
            save_dir = "C:\\Users\Tomas\Desktop\Tesis\Programacion\\results\pruning\data\{}_{}\connectivity".format(
                map_name, "_".join(group.split())
            )
            # save_dir = "C:\\Users\Tomas\Desktop\connectivity"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            os.chdir(save_dir)

            D_k = k_shortest_paths_length(connectomes, paths, dists, trace=True)
            np.save(
                "k_shortest_paths_length_k_{}prune_val_{}_{}_{}".format(
                    k, prune_val, map_name, "_".join(group.split())
                ),
                D_k,
            )

            n_centrality, e_centrality = k_nodes_edges_centrality(paths, trace=True)
            np.save(
                "nodes_k_centrality_k_{}_prune_val_{}_{}_{}".format(
                    k, prune_val, map_name, "_".join(group.split())
                ),
                n_centrality,
            )
            np.save(
                "edges_k_centrality_k_{}_prune_val_{}_{}_{}".format(
                    k, prune_val, map_name, "_".join(group.split())
                ),
                e_centrality,
            )

            S = search_information(connectomes, paths, k_vals=k_vals, trace=True)
            np.save(
                "search_information_k_{}_prune_val_{}_{}_{}".format(
                    k_vals, prune_val, map_name, "_".join(group.split())
                ),
                S,
            )

            P_transitivity = k_path_transitivity(connectomes, paths, trace=True)
            np.save(
                "k_path_transitivity_k_{}_prune_val_{}_{}_{}".format(
                    k, prune_val, map_name, "_".join(group.split())
                ),
                P_transitivity,
            )

            RE = routing_efficiency(connectomes, trace=True)
            np.save(
                "routing_efficiency_prune_val_{}_{}".format(
                    prune_val, "_".join(group.split())
                ),
                RE,
            )

            DE = diffussion_efficiency(connectomes, trace=True)
            np.save(
                "difussion_efficiency_prune_val_{}_{}".format(
                    prune_val, "_".join(group.split())
                ),
                DE,
            )

            C = communicability(connectomes, trace=True)
            np.save(
                "communicability_prune_val_{}_{}".format(
                    prune_val, "_".join(group.split())
                ),
                C,
            )

            EV = coupling_matrix_eigenvals(connectomes, trace=True)
            np.save(
                "coupling_matrix_eigenvals_prune_val_{}_{}".format(
                    prune_val, "_".join(group.split())
                ),
                EV,
            )

            os.chdir(current_dir)
