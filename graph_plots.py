from real_networks import *
from connectivity_measures import k_nodes_edges_centrality

sns.set(style="whitegrid")
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"] = 0

fs = 16

if __name__ == "__main__":
    # connectome = connectomes_filtered("NOMBRE ", "caso010", op.eq)
    # connectome = connectome[:, :20, :20]
    # prune_vals = [0, 0.5, 0.9]
    # g = nx.from_numpy_matrix(connectome[0])
    # pos = nx.random_layout(g)
    # node_edges_color = "black"
    # colormap = plt.cm.coolwarm
    # node_size = 100
    # node_alpha = 1
    # fig, axes = plt.subplots(nrows=1, ncols=len(prune_vals), figsize=(12, 6))
    # vmin = np.log(np.min(connectome[connectome != 0]) + 1)
    # vmax = np.log(np.max(connectome) + 1)
    # for i, prune_val in enumerate(prune_vals):
    #     pruned_connectome = prune_connectomes(
    #         connectome, prune_type="percentage", val=prune_val
    #     )
    #     g = nx.from_numpy_matrix(pruned_connectome[0])
    #     weights = np.log(np.array([g[u][v]["weight"] for u, v in g.edges()]) + 1)
    #     nodes = nx.draw_networkx_nodes(
    #         g,
    #         ax=axes[i],
    #         pos=pos,
    #         node_size=node_size,
    #         edgecolors=node_edges_color,
    #         alpha=node_alpha,
    #     )  # , width=weights)
    #     edges = nx.draw_networkx_edges(
    #         g,
    #         ax=axes[i],
    #         pos=pos,
    #         edge_color=weights,
    #         edge_cmap=colormap,
    #         edge_vmin=vmin,
    #         edge_vmax=vmax,
    #     )
    #     axes[i].set_title("{:.0f}%".format(prune_val * 100), fontsize=fs)
    #     axes[i].axis("off")

    # cbar = fig.colorbar(edges)
    # cbar.set_ticks([])
    # cbar.ax.set_ylabel("Peso de la conexión", fontsize=fs)
    # fig.suptitle("Ejemplo de pruning de redes", fontsize=fs)
    # plt.tight_layout()
    # plt.savefig("charlita\\ejemplo_pruning.png", dpi=400)
    # plt.show()

    # connectomes = connectomes_filtered("NOMBRE ", "0", op.ge)
    # # index = np.triu_indices(82, 1)
    # # connectomes = connectomes[:, index[0], index[1]]
    # std = connectomes.std(axis=0).ravel()
    # std = std[std != 0]
    # mean = connectomes.mean(axis=0).ravel()
    # mean = mean[mean != 0]
    # plt.figure(figsize=(12, 6))
    # plt.scatter(mean[mean != 0], (std / mean), alpha=0.5)
    # plt.gca().set_xscale("log")
    # plt.xlabel("Intensidad media", fontsize=fs)
    # plt.ylabel("Coeficiente de variación", fontsize=fs)
    # plt.tick_params(labelsize=fs - 2)
    # plt.tight_layout()
    # plt.savefig("charlita\\coeficiente_variacion.png", dpi=400)
    # plt.show()

    # connectome = connectomes_filtered("NOMBRE ", "caso010", op.eq)
    # connectome = connectome[:, :15, :15]
    # g = nx.from_numpy_matrix(connectome[0])
    # fs = 16
    # n_paths = 3
    # node_1 = 0
    # node_2 = 1
    # path_colors = ["red", "blue", "green"]
    # g = nx.gnm_random_graph(7, 15)
    # pos = nx.spring_layout(g)
    # paths = nx.shortest_simple_paths(g, node_1, node_2)
    # paths = [path for i, path in enumerate(paths) if i < n_paths]
    # node_edges_color = "black"
    # colormap = plt.cm.coolwarm
    # node_size = 500
    # node_alpha = 1
    # nodes = nx.draw_networkx_nodes(
    #     g, pos=pos, node_size=node_size, edgecolors="black", alpha=node_alpha,
    # )  # , width=weights)
    # labels = nx.draw_networkx_labels(g, pos=pos, font_size=fs)
    # edges = nx.draw_networkx_edges(g, pos=pos, alpha=0.5, width=4)
    # for i in range(n_paths):
    #     path = paths[i]
    #     path_edges = set(zip(path[:-1], path[1:]))
    #     sh_path_edges = nx.draw_networkx_edges(
    #         g,
    #         edgelist=path_edges,
    #         pos=pos,
    #         width=4,
    #         edge_color=path_colors[i],
    #         label="Path {}".format(i + 1),
    #     )

    # plt.grid(False)
    # plt.legend(fontsize=fs)
    # plt.title(
    #     "{}-Shortest paths entre {} y {}".format(n_paths, node_1, node_2), fontsize=fs
    # )
    # plt.tight_layout()
    # plt.savefig("charlita\\ejemplo_k_shortest_paths.png", dpi=400)
    # plt.show()

    # n_nodes = 10
    # connectome = np.random.random(size=(n_nodes, n_nodes))
    # connectome[connectome <= 0.2] = 0
    # node_1 = 0
    # node_2 = 1
    # connectome[node_1, :] = 0
    # connectome[node_1, node_2] = 0.7
    # g = nx.from_numpy_matrix(np.triu(connectome))
    # fs = 16
    # path_colors = ["red"]
    # pos = nx.spring_layout(g)
    # path = nx.dijkstra_path(g, node_1, node_2)
    # node_edges_color = "black"
    # colormap = plt.cm.coolwarm
    # node_size = 500
    # node_alpha = 1
    # nodes = nx.draw_networkx_nodes(
    #     g, pos=pos, node_size=node_size, edgecolors="black", alpha=node_alpha,
    # )  # , width=weights)
    # labels = nx.draw_networkx_labels(g, pos=pos, font_size=fs)
    # edges = nx.draw_networkx_edges(g, pos=pos, alpha=0.5, width=4)
    # path_edges = set(zip(path[:-1], path[1:]))
    # sh_path_edges = nx.draw_networkx_edges(
    #     g, edgelist=path_edges, pos=pos, width=4, edge_color="red",
    # )
    # labels = {path_edge: r"$w_{01}$" for path_edge in path_edges}
    # nx.draw_networkx_edge_labels(g, pos, edge_labels=labels, font_size=fs)
    # plt.grid(False)
    # plt.tight_layout()
    # plt.savefig("charlita\\path_prob_example.png", dpi=400)
    # plt.show()

    # g = nx.Graph()
    # fs = 16
    # edges = [
    #     [0, 1],
    #     [1, 2],
    #     [0, 3],
    #     [3, 1],
    #     [1, 4],
    #     [4, 2],
    #     [1, 5],
    #     [5, 4],
    #     [0, 6],
    #     [6, 2],
    # ]
    # p_edges = [[0, 1], [1, 2]]
    # d_edges = [[0, 3], [3, 1], [1, 4], [4, 2], [0, 6], [6, 2]]
    # g.add_edges_from(edges)
    # labels = [
    #     r"$w_{01}$",
    #     r"$w_{12}$",
    #     r"$w_{03}$",
    #     r"$w_{31}$",
    #     r"$w_{14}$",
    #     r"$w_{42}$",
    #     r"$w_{15}$",
    #     r"$w_{54}$",
    #     r"$w_{06}$",
    #     r"$w_{62}$",
    # ]
    # e_labels = {tuple(edges[i]): labels[i] for i in range(len(edges))}
    # pos = nx.spring_layout(g)
    # nx.draw(g, pos, node_size=500, edgecolors="black", width=5)
    # nx.draw_networkx_labels(g, pos=pos, font_size=fs)
    # nx.draw_networkx_edges(
    #     g, pos, edgelist=d_edges, edge_color="red", width=5, label="Desviación"
    # )
    # nx.draw_networkx_edges(
    #     g, pos, edgelist=p_edges, edge_color="blue", width=5, label="Camino"
    # )
    # nx.draw_networkx_edge_labels(g, pos, edge_labels=e_labels, font_size=fs)
    # plt.legend(fontsize=fs)
    # plt.tight_layout()
    # plt.savefig("charlita\\path_transitivity_example.png", dpi=400)
    # plt.show()

    g = nx.Graph()
    fs = 16
    val = 20
    edges_0 = [[0, i] for i in range(2, val)]
    edges_1 = [[1, i] for i in range(val, 2 * val)]
    edges = edges_0 + edges_1 + [[0, 1]]
    g.add_edges_from(edges)
    pos = nx.spring_layout(g)
    nx.draw(
        g,
        pos,
        node_size=[(v + 0.5) * 200 for v in nx.betweenness_centrality(g).values()],
        edgecolors="black",
        width=[v * 5 for v in nx.edge_betweenness_centrality(g).values()],
    )
    plt.tight_layout()
    plt.savefig("charlita\\centrality.png", dpi=400)
    plt.show()
