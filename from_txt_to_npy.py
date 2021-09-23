from real_networks import *

if __name__ == "__main__":
    prune_type = "percentage"
    map_name = "inv"
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
    # cargo solo los conectomas que me interesan
    connectomes_original, cases = connectomes_filtered(
        filter_att, filter_val, op, remove_nodes=remove_nodes, ret_cases=True
    )
    n_connectomes = len(connectomes_original)
    c_size = connectomes_original.shape[1]
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
    else:
        prune_vals = prune_vals_given

    group_str = "_".join(filter_val.split())
    save_dir = "C:\\Users\Tomas\Desktop\Tesis\Programacion\\results\pruning\data\k_paths\{}_{}".format(
        map_name, group_str
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for prune_val in prune_vals:
        k_paths_save = np.empty(shape=(n_connectomes, c_size, c_size, k), dtype=object)
        k_dists_save = np.zeros(shape=(n_connectomes, c_size, c_size, k))
        # les aplico pruning a los conectomas
        connectomes = prune_connectomes(connectomes_original, prune_type, prune_val)
        for j, connectome in enumerate(connectomes):
            if trace:
                print(
                    "Prune val: {:.3f}/{:.3f} - {} - Connectome: {}/{} - ".format(
                        prune_val, np.max(prune_vals), cases[j], j + 1, n_connectomes
                    ),
                    end="",
                )
            # mido el tiempo
            start = time.time()
            k_paths_save[j], k_dists_save[j] = k_shortest_paths_load_txt(
                filter_val=filter_val,
                case=cases[j],
                k_max=k,
                prune_type=prune_type,
                prune_val=prune_val,
                weight=map_name,
                dists=True,
                trace=False,
            )
            end = time.time()
            # printeo el tiempo
            if trace:
                print("{:.5f} s".format(end - start))
        paths_name = "k_paths_saved_k={}_{}_prune_{}_{}".format(
            k, prune_type, prune_val * 100, map_name
        )
        dists_name = "k_dists_saved_k={}_{}_prune_{}_{}".format(
            k, prune_type, prune_val * 100, map_name
        )
        current_dir = os.getcwd()
        os.chdir(save_dir)
        np.save(paths_name, k_paths_save)
        np.save(dists_name, k_dists_save)
        os.chdir(current_dir)
