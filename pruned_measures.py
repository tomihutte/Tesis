from real_networks import *


def pruned_measures(
    prune_type,
    map,
    map_name,
    k_vals,
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
    only_plot=False,
):
    # esta funcion aplica prunning progresivo a un conjunto de conectomas
    # y calcula medidas a medida que avanza el prunning

    # cargo solo los conectomas que me interesan
    connectomes_original, cases = connectomes_filtered(
        filter_att, filter_val, op, remove_nodes=remove_nodes, ret_cases=True
    )
    n_connectomes = len(connectomes_original)
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

    if not (only_plot):
        # creo los arrays donde vamos a guardar las cosas
        e_number = np.zeros(shape=(prune_vals_number, n_connectomes))
        sh_paths = np.zeros(shape=(prune_vals_number, n_connectomes))
        k_paths = np.zeros(shape=(len(k_vals), prune_vals_number, n_connectomes))
        k_lengths = np.zeros(shape=(len(k_vals), prune_vals_number, n_connectomes))
        k_edges = np.zeros(shape=(len(k_vals), prune_vals_number, n_connectomes))
        clustering = np.zeros(shape=(prune_vals_number, n_connectomes))
        edge_connectivity = np.zeros(shape=(prune_vals_number, n_connectomes))

        # ahora recorro todos los prune val y los conetomas
        for i, prune_val in enumerate(prune_vals):
            # les aplico pruning a los conectomas
            connectomes = prune_connectomes(connectomes_original, prune_type, prune_val)
            start = time.time()
            k_paths_loaded, k_dists_loaded = k_shortest_paths_load_npy(
                prune_type, prune_val, map_name, filter_val, np.max(k_vals), dists=True
            )
            if trace:
                print(
                    "Loading k_paths and k_dists - {:.2f} s".format(time.time() - start)
                )
            for j, connectome in enumerate(connectomes):
                # mido el tiempo
                start = time.time()
                # creo un grafo del conectoma
                G = nx.from_numpy_matrix(connectome)
                # agrego el numero de aristas que tiene el grafo
                e_number[i, j] = G.number_of_edges()
                # le agrego un atributo
                add_edge_map(G, map, map_name)
                # clustering pesado
                clustering[i, j] = nx.average_clustering(G, weight="weight")
                # sh paths siguiendo el atributo que agregue
                try:
                    sh_paths[i, j] = nx.average_shortest_path_length(G, weight=map_name)
                except nx.NetworkXError:
                    sh_paths[i, j] = np.NaN

                # edge connectivity
                edge_connectivity[i, j] = nx.edge_connectivity(
                    G, flow_func=shortest_augmenting_path
                )
                # longitud, aristas usadas y longitud en aristas
                path_means, path_lengths, e_percent = mean_k_shortest_path_length(
                    G, k_vals, k_paths_loaded[j], k_dists_loaded[j]
                )

                k_paths[:, i, j] = path_means
                k_lengths[:, i, j] = path_lengths
                k_edges[:, i, j] = e_percent

                end = time.time()
                # printeo el tiempo
                if trace:
                    print(
                        "Prune val: {:.3f}/{:.3f} - Connectome:{}/{} - {:.5f} s".format(
                            prune_val,
                            np.max(prune_vals),
                            j + 1,
                            n_connectomes,
                            end - start,
                        )
                    )

        # cambio el directorio para guardar las cosas
        current_dir = os.getcwd()
        save_dir = "C:\\Users\Tomas\Desktop\Tesis\Programacion\\results\pruning\data\{}_{}".format(
            map_name, "_".join(filter_val.split())
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        os.chdir(save_dir)

        np.save(
            "e_number_{}_prune_{}_{}_{}_{}.npy".format(
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            ),
            e_number,
        )
        np.save(
            "sh_path_{}_prune_{}_{}_{}_{}".format(
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            ),
            sh_paths,
        )
        np.save(
            "clustering_{}_prune_{}_{}_{}_{}".format(
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            ),
            clustering,
        )
        np.save(
            "edge_conn_{}_prune_{}_{}_{}_{}".format(
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            ),
            edge_connectivity,
        )
        np.save(
            "k_paths_{}_{}_prune_{}_{}_{}_{}".format(
                k_vals,
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            ),
            k_paths,
        )
        np.save(
            "k_lengths_{}_{}_prune_{}_{}_{}_{}".format(
                k_vals,
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            ),
            k_lengths,
        )
        np.save(
            "k_edges_{}_{}_prune_{}_{}_{}_{}".format(
                k_vals,
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            ),
            k_edges,
        )

    else:
        current_dir = os.getcwd()
        os.chdir(
            "C:\\Users\Tomas\Desktop\Tesis\Programacion\\results\pruning\data\{}_{}".format(
                map_name, "_".join(filter_val.split())
            )
        )

        e_number = np.load(
            "e_number_{}_prune_{}_{}_{}_{}.npy".format(
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            )
        )
        sh_paths = np.load(
            "sh_path_{}_prune_{}_{}_{}_{}.npy".format(
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            )
        )
        clustering = np.load(
            "clustering_{}_prune_{}_{}_{}_{}.npy".format(
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            )
        )
        edge_connectivity = np.load(
            "edge_conn_{}_prune_{}_{}_{}_{}.npy".format(
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            )
        )
        k_paths = np.load(
            "k_paths_{}_{}_prune_{}_{}_{}_{}.npy".format(
                k_vals,
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            )
        )
        k_lengths = np.load(
            "k_lengths_{}_{}_prune_{}_{}_{}_{}.npy".format(
                k_vals,
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            )
        )
        k_edges = np.load(
            "k_edges_{}_{}_prune_{}_{}_{}_{}.npy".format(
                k_vals,
                prune_type,
                np.min(prune_vals) * 100,
                np.max(prune_vals) * 100,
                map_name,
                "_".join(filter_val.split()),
            )
        )

    if map_name == "inv_log":
        map_str = "1/log(1+w)"
    elif map_name == "inv":
        map_str = "1/w"

    if filter_val == "control sano":
        group_str = "Pacientes sanos"
    elif filter_val == "migraña crónica":
        group_str = "Pacientes con migraña crónica"
    elif filter_val == "migraña episódica":
        group_str = "Pacientes con migraña episódica"

    save_dir = "C:\\Users\Tomas\Desktop\Tesis\Programacion\\results\pruning\\figures\{}_{}".format(
        map_name, "_".join(filter_val.split())
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.chdir(save_dir)

    plot(
        prune_vals * 100,
        sh_paths,
        "Aristas eliminadas [%]",
        "Longitud media",
        r"Shortest path - {} - $f\left(w\right)$={}".format(group_str, map_str),
        "sh_path_{}_prune_{}_{}_{}_{}.pdf".format(
            prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100,
            map_name,
            "_".join(filter_val.split()),
        ),
    )

    plot(
        prune_vals * 100,
        clustering,
        "Aristas eliminadas [%]",
        "Coeficiente de clustering medio",
        "Clustering coefficient - {}".format(group_str),
        "clustering_{}_prune_{}_{}_{}_{}.pdf".format(
            prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100,
            map_name,
            "_".join(filter_val.split()),
        ),
    )

    plot(
        prune_vals * 100,
        edge_connectivity,
        "Aristas eliminadas [%]",
        "Edge connectivity ",
        "Edge connectivity - {}".format(group_str),
        "edge_conn_{}_prune_{}_{}_{}_{}.pdf".format(
            prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100,
            map_name,
            "_".join(filter_val.split()),
        ),
    )

    k_legends = [r"$k={}$".format(k) for k in k_vals]

    plot(
        prune_vals * 100,
        k_paths,
        "Aristas eliminadas [%]",
        "Longitud media",
        r"K shortest paths - {} - $f(w)$={}".format(group_str, map_str),
        "k_paths_{}_{}_prune_{}_{}_{}_{}.pdf".format(
            k_vals,
            prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100,
            map_name,
            "_".join(filter_val.split()),
        ),
        legends=k_legends,
    )

    plot(
        prune_vals * 100,
        k_lengths,
        "Aristas eliminadas [%]",
        "Numero de aristas medio",
        r"K shortest paths - {} - $f(w)$={}".format(group_str, map_str),
        "k_lengths_{}_{}_prune_{}_{}_{}_{}.pdf".format(
            k_vals,
            prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100,
            map_name,
            "_".join(filter_val.split()),
        ),
        legends=k_legends,
    )

    plot(
        prune_vals * 100,
        k_edges * 100,
        "Aristas eliminadas [%]",
        "Porcentaje de aristas reales usadas",
        r"K shortest paths - {} - $f(w)$={}".format(group_str, map_str),
        "k_edges_{}_{}_prune_{}_{}_{}_{}.pdf".format(
            k_vals,
            prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100,
            map_name,
            "_".join(filter_val.split()),
        ),
        legends=k_legends,
    )

    plot(
        prune_vals * 100,
        k_edges * e_number,
        "Aristas eliminadas [%]",
        "Numero de aristas reales usadas",
        r"K shortest paths - {} - $f(w)$={}".format(group_str, map_str),
        "k_number_of_edges_{}_{}_prune_{}_{}_{}_{}.pdf".format(
            k_vals,
            prune_type,
            np.min(prune_vals) * 100,
            np.max(prune_vals) * 100,
            map_name,
            "_".join(filter_val.split()),
        ),
        legends=k_legends,
    )

    os.chdir(current_dir)

    return (
        sh_paths,
        clustering,
        edge_connectivity,
        k_paths,
        k_lengths,
        k_edges,
        prune_vals,
    )


if __name__ == "__main__":
    s, c, e_c, k_p, k_l, k_e, p_v = pruned_measures(
        prune_type="percentage",
        map=lambda x: 1 / np.log10(1 + x),
        map_name="inv_log",
        k_vals=[1, 2, 5, 10, 20, 50],
        prune_vals_given=None,
        prune_vals_number=10,
        prune_vals_max=0.85,
        prune_vals_offset=0.01,
        filter_att="GRUPO",
        filter_val="migraña episódica",
        op=op.eq,
        remove_nodes=[34, 83],
        trace=True,
        trace_k_paths=False,
        only_plot=False,
    )

