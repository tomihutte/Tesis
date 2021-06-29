from real_networks import *


def path_probability(G, weight_matrix, path):
    probs = [
        weight_matrix[path[i], path[i + 1]] / G.degree[path[i]]
        for i in range(len(path) - 1)
    ]
    return np.sum(probs)


def k_shortests_paths_length(connectomes, paths, dists, trace=False):
    paths_prob = np.empty(shape=paths.shape)
    nn = connectomes.shape[0]
    for case, connectome in enumerate(connectomes):
        start = time.time()
        if trace:
            print("Caso {}/{}".format(case, nn), end="")
        G = nx.from_numpy_matrix(connectome)
        connectome += connectome.T
        for node_i in range(paths.shape[1]):
            for node_j in range(node_i + 1, paths.shape[2]):
                for k in range(paths.shape[3]):
                    paths_prob[case, node_i, node_j, k] = path_probability(
                        G, connectome, paths[case, node_i, node_j, k]
                    )
        if trace:
            print(" - {:.3f} s".format(time.time() - start))
    paths_prob.reshape(paths.shape)
    paths_prob /= paths_prob.sum(axis=-1)[:, :, :, np.newaxis]
    D_k = (paths_prob * dists).sum(axis=-1)
    return D_k


if __name__ == "__main__":
    connectomes_original = connectomes_filtered("GRUPO", "control sano", op.eq)
    prune_vals = prune_vals_calc(connectomes_original)
    prune_val = prune_vals[-4]
    connectomes = prune_connectomes(connectomes_original, "percentage", prune_val)
    paths, dists = k_shortest_paths_load_npy(
        "percentage", prune_val, "inv_log", "control sano", 50, dists=True
    )
    D_k = k_shortests_paths_length(connectomes, paths, dists, trace=True)

