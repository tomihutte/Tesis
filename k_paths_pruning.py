from real_networks import *
import multiprocessing


def process_function_k_paths(
    connectomes,
    j_vals,
    prune_val,
    prune_vals,
    cases,
    map,
    map_name,
    k,
    prune_type,
    save_path,
    trace,
    trace_k_paths,
):
    for j in j_vals:
        connectome = connectomes[j]
        n_connectomes = len(connectomes)
        if trace:
            print(
                "Working with - Prune val: {:.3f}/{:.3f} - {} - Connectome: {}/{}".format(
                    prune_val, np.max(prune_vals), cases[j], j + 1, n_connectomes
                )
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
            f.write("Inicio,Fin,Camino(separado por comas),Distancia del camino\n")
            for i_idx in range(paths.shape[0]):
                for j_idx in range(i_idx + 1, paths.shape[1]):
                    if paths[i_idx, j_idx][k_v] != None:
                        f.write(
                            "{},{},{},{}\n".format(
                                i_idx,
                                j_idx,
                                ",".join([str(e) for e in paths[i_idx, j_idx][k_v]]),
                                path_weight(G, paths[i_idx, j_idx][k_v], map_name),
                            )
                        )
                    else:
                        f.write("{},{},None,None\n".format(i_idx, j_idx))
            f.close()

        end = time.time()
        # printeo el tiempo
        if trace:
            print(
                "Prune val: {:.3f}/{:.3f} - {} - Connectome: {}/{} - {:.5f} s".format(
                    prune_val,
                    np.max(prune_vals),
                    cases[j],
                    j + 1,
                    n_connectomes,
                    end - start,
                )
            )


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    prune_type = "percentage"
    map_name = "inv_log"
    if map_name == "inv":
        map = inv
    elif map_name == "inv_log":
        map = inv_log
    else:
        raise ValueError("Invalid map name")
    k = 50
    prune_vals_given = [0.35]
    prune_vals_number = 10
    prune_vals_max = 0.85
    prune_vals_offset = 0.01
    filter_att = "GRUPO"
    filter_val = "migraña episódica"
    op = op.eq
    remove_nodes = [34, 83]
    trace = True
    trace_k_paths = False
    prune_start = 0
    case_start = 64
    prune_finish = None
    case_finish = None
    connectomes_original, cases = connectomes_filtered(
        filter_att, filter_val, op, remove_nodes=remove_nodes, ret_cases=True
    )
    n_connectomes = len(connectomes_original)
    c_size = connectomes_original.shape[1]
    if case_finish == None:
        case_finish = n_connectomes - 1
    print("Loading {} connectomes".format(n_connectomes))
    print("Using {} map".format(map_name))
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
            prune_finish = prune_vals_number - 1
    else:
        prune_vals = prune_vals_given
        prune_finish = len(prune_vals) - 1

    start_flag = False
    save_path = "C:\\Users\Tomas\Desktop\Tesis\Programacion\\results\pruning\k_paths_txt\{}_{}".format(
        map_name, "_".join(filter_val.split())
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, prune_val in enumerate(prune_vals):
        if (i >= prune_start) and (i <= prune_finish):
            # les aplico pruning a los conectomas
            connectomes = prune_connectomes(connectomes_original, prune_type, prune_val)
            n_process = 4
            start = 0
            finish = n_connectomes
            if i == prune_start:
                start = case_start
            if i == prune_finish:
                finish = case_finish + 1
            all_connectomes = np.arange(start, finish)
            processes = []
            for l in range(n_process):
                # ipdb.set_trace()
                j_vals = all_connectomes[l::n_process]
                p = multiprocessing.Process(
                    target=process_function_k_paths,
                    args=(
                        connectomes,
                        j_vals,
                        prune_val,
                        prune_vals,
                        cases,
                        map,
                        map_name,
                        k,
                        prune_type,
                        save_path,
                        trace,
                        trace_k_paths,
                    ),
                )
                processes.append(p)
                p.start()
            for process in processes:
                process.join()
