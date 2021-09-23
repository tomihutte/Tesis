from matplotlib.pyplot import xticks
from real_networks import *
import seaborn as sns
import scipy as sp
import sklearn.preprocessing, sklearn.decomposition

sns.set(style="whitegrid")
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"] = 1.25


def load_measures(map_name, group, k, k_vals, prune_val):
    current_dir = os.getcwd()
    save_dir = "C:\\Users\Tomas\Desktop\Tesis\Programacion\\results\pruning\data\{}_{}\connectivity".format(
        map_name, "_".join(group.split())
    )
    if not os.path.exists(save_dir):
        print("Bad path")
        return
    os.chdir(save_dir)
    D_k = np.load(
        "k_shortest_paths_length_k_{}_prune_val_{}_{}_{}.npy".format(
            k, prune_val, map_name, "_".join(group.split())
        )
    )
    n_centrality = np.load(
        "nodes_k_centrality_k_{}_prune_val_{}_{}_{}.npy".format(
            k, prune_val, map_name, "_".join(group.split())
        )
    )
    e_centrality = np.load(
        "edges_k_centrality_k_{}_prune_val_{}_{}_{}.npy".format(
            k, prune_val, map_name, "_".join(group.split())
        )
    )
    S = np.load(
        "search_information_k_{}_prune_val_{}_{}_{}.npy".format(
            k_vals, prune_val, map_name, "_".join(group.split())
        )
    )
    P_transitivity = np.load(
        "k_path_transitivity_k_{}_prune_val_{}_{}_{}.npy".format(
            k, prune_val, map_name, "_".join(group.split())
        )
    )
    RE = np.load(
        "routing_efficiency_prune_val_{}_{}.npy".format(
            prune_val, "_".join(group.split())
        )
    )
    DE = np.load(
        "diffusion_efficiency_prune_val_{}_{}.npy".format(
            prune_val, "_".join(group.split())
        )
    )
    C = np.load(
        "communicability_prune_val_{}_{}.npy".format(prune_val, "_".join(group.split()))
    )
    EV = np.load(
        "coupling_matrix_eigenvals_prune_val_{}_{}.npy".format(
            prune_val, "_".join(group.split())
        )
    )
    os.chdir(current_dir)
    return D_k, n_centrality, e_centrality, S, P_transitivity, RE, DE, C, EV


def plot_hist(
    data,
    title,
    savefig,
    rows=False,
    cols=["Control sano", "Migraña crónica", "Migraña episódica"],
    data_val=0,
    fs=16,
    nbins=10,
):

    fig, axes = plt.subplots(
        nrows=len(data),
        ncols=len(cols),
        figsize=(10, 3 * len(data)),
        sharex="row",
        sharey=True,
    )
    for i in range(len(data)):
        bins = np.histogram(
            np.concatenate(
                (data[i, data_val, 0], data[i, data_val, 1], data[i, data_val, 2])
            ),
            bins=nbins,
        )[1]
        for j in range(len(cols)):
            axes[i, j].hist(data[i, data_val, j], bins=bins)

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=fs - 2)

    if rows:
        for ax, row in zip(axes[:, 0], rows):
            ax.set_ylabel(row, rotation=90, fontsize=fs - 2)

    fig.suptitle(title, fontsize=fs)
    fig.tight_layout()
    plt.savefig(savefig)
    plt.show()


def plot_jitter(
    data,
    title,
    savefig,
    rows=False,
    cols=["Control sano", "Migraña crónica", "Migraña episódica"],
    data_val=0,
    fs=16,
    size=7,
    alpha=0.8,
):
    # ipdb.set_trace()
    fig = plt.figure(figsize=(8, 6))
    for i in range(len(data)):
        df = pd.DataFrame()
        df["Grupo"] = cols
        df["vals"] = [data[i, data_val, j] for j in range(len(cols))]
        df = df.explode("vals")
        subplot_number = int("{}{}{}".format(len(data), 1, i + 1))
        ax = fig.add_subplot(subplot_number)
        sns.stripplot(
            x="Grupo", y="vals", data=df, size=size, jitter=True, zorder=1, alpha=alpha
        )
        y_pos = [data[i, data_val, j].mean() for j in range(len(cols))]
        y_err = [
            data[i, data_val, j].std() / np.sqrt(len(data[i, data_val, j]))
            for j in range(len(cols))
        ]
        ax.errorbar(
            x=[n for n in range(len(y_pos))],
            y=y_pos,
            yerr=y_err,
            color="black",
            ecolor="black",
            fmt="o",
        )
        ax.set_xlabel("")
        if rows:
            ax.set_ylabel(rows[i], fontsize=fs)
    ax.set_xlabel("Grupo", fontsize=fs)
    plt.tick_params(labelsize=fs)
    fig.suptitle(title, fontsize=fs)
    fig.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()


def data_frame_create(cols, rows, subcols=None, subrows=None):
    if subcols:
        arrays = [np.repeat(cols, len(subcols)), subcols * len(cols)]
        cols = list(zip(*arrays))
        cols = pd.MultiIndex.from_tuples(cols)
    if subrows:
        arrays = [np.repeat(rows, len(subrows)), subrows * len(rows)]
        rows = list(zip(*arrays))
        rows = pd.MultiIndex.from_tuples(rows)
    df = pd.DataFrame(index=rows, columns=cols)
    return df, cols, rows


def plot_measures(data, title, rows, savefig):
    plot_hist(
        data, title=title, savefig=savefig[0], rows=rows,
    )
    plot_jitter(
        data, title=title, savefig=savefig[1], rows=rows,
    )


def format_scalar_measures(DE, RE, EV):
    # vamos a guardar todos los datos en un mismo formato
    EV2 = np.empty(shape=DE.shape, dtype=object)
    EVN = np.empty(shape=DE.shape, dtype=object)
    R = np.empty(shape=EV.shape, dtype=object)
    for i in range(2):
        for j in range(3):
            EV2[i, j], EVN[i, j], R[i, j] = (
                EV[i, j][:, 0],
                EV[i, j][:, 1],
                EV[i, j][:, 1] / EV[i, j][:, 0],
            )
    data = np.empty(shape=(6, 2, 3), dtype=object)
    data[0] = EV2
    data[1] = EVN
    data[2] = R
    data[3] = RE
    data[4] = RE[[1, 0]]
    data[5] = DE
    return data


def filter_outliers(data, n_sigmas=3):
    filtered_data = np.empty(shape=data.shape, dtype=object)
    for n, variable in enumerate(data):
        for i in range(variable.shape[0]):
            for j in range(variable.shape[1]):
                std = np.atleast_1d(np.std(variable[i, j], axis=0))
                mean = np.mean(variable[i, j], axis=0)
                # ipdb.set_trace()
                upper_bound = variable[i, j] < (mean + std * n_sigmas)[np.newaxis, :]
                lower_bound = variable[i, j] > (mean - std * n_sigmas)[np.newaxis, :]
                filtered_data[n, i, j] = variable[i, j][
                    ...,
                    np.logical_and(
                        np.all(lower_bound, axis=0), np.all(upper_bound, axis=0)
                    ),
                ]
    return filtered_data


def scalar_measures_analysis(
    DE, RE, EV, save_dir, no_outliers, plot_flag, groups, n_sigmas=3
):
    if no_outliers:
        save_token = "no_outliers"
        title_token = r" - No outliers - {}$\sigma$".format(n_sigmas)
    else:
        save_token = "all_samples"
        title_token = " - All samples"

    data = format_scalar_measures(DE, RE, EV)

    if no_outliers:
        data = filter_outliers(data, n_sigmas)

    if plot_flag:
        plot_measures(
            data[3:],
            title="Network Efficiency{}".format(title_token),
            savefig=[
                os.path.join(
                    save_dir, "routing_efficiency_hist_{}.pdf".format(save_token)
                ),
                os.path.join(
                    save_dir, "routing_efficiency_jitter_{}.pdf".format(save_token)
                ),
            ],
            rows=["Routing - Inv map", "Routing - Inv Log map", "Diffusion"],
        )
        plot_measures(
            data[:3],
            title="Autovalores de la matriz de acoplamiento{}".format(title_token),
            savefig=[
                os.path.join(
                    save_dir, "coupling_matrix_eigenval_hist_{}.pdf".format(save_token)
                ),
                os.path.join(
                    save_dir,
                    "coupling_matrix_eigenval_jitter_{}.pdf".format(save_token),
                ),
            ],
            rows=[r"$\lambda_2$", r"$\lambda_N$", r"$\lambda_N$/$\lambda_2$"],
        )

    # analisis estadistico de variables unidimensionales

    ## vamos a crear un dataframe para poder pasar todoa a una planilla automaticamente
    parametros_groups = [
        "estadístico-shapiro",
        "p-value-shapiro",
        "valor medio",
        "mediana",
        "desviación estandar",
    ]
    variables = [
        "Lambda 2",
        "Lambda N",
        "Lambda ratio",
        "Routing Efficiency - Inv",
        "Routing Efficiency - Inv Log",
        "Diffusion Efficiency",
    ]

    parametros_comparison_groups = [
        "estadístico",
        "p-value",
    ]

    comparison_groups = [
        "Control sano vs Migraña crónica",
        "Control sano vs Migraña episódica",
        "Migraña crónica vs Migraña episódica",
    ]

    tests = ["T-TEST", "Wilcoxon"]

    df_groups, df_groups_cols, df_groups_rows = data_frame_create(
        cols=parametros_groups, subcols=None, rows=variables, subrows=groups
    )
    df_comparison, df_comparison_cols, df_comparison_rows = data_frame_create(
        cols=comparison_groups,
        subcols=parametros_comparison_groups,
        rows=tests,
        subrows=variables,
    )

    ## Analisis intra-grupos
    ## Shapiro, media, mediana y desviacion estandar de las variables
    for i, group in enumerate(groups):
        for j, variable in enumerate(data):
            s, p = sp.stats.shapiro(variable[0, i])
            mean = np.mean(variable[0, i])
            median = np.median(variable[0, i])
            std = np.std(variable[0, i])
            df_groups.loc[df_groups_rows[j * len(groups) + i]] = [
                s,
                p,
                mean,
                median,
                std,
            ]

    ## Analisis inter-grupos
    # T-TEST
    comparison_group_indexes = [(0, 1), (0, 2), (1, 2)]
    for i, comparison_group in enumerate(comparison_groups):
        for j, variable in enumerate(data):
            s, p = sp.stats.ttest_ind(
                variable[0, comparison_group_indexes[i][0]],
                variable[0, comparison_group_indexes[i][1]],
                equal_var=False,
            )
            df_comparison.loc[(tests[0], variables[j]), comparison_group] = [s, p]
    # Wilcoxon
    for i, comparison_group in enumerate(comparison_groups):
        for j, variable in enumerate(data):
            s, p = sp.stats.ranksums(
                variable[0, comparison_group_indexes[i][0]],
                variable[0, comparison_group_indexes[i][1]],
            )
            df_comparison.loc[(tests[1], variables[j]), comparison_group] = [s, p]

    df_groups.to_csv(
        os.path.join(save_dir, "shapiro-mean-median-std_{}.csv".format(save_token))
    )
    df_comparison.to_csv(
        os.path.join(save_dir, "ttest-wilcoxon_{}.csv".format(save_token))
    )


def format_vectorial_measures(D_k, n_c, e_c, S, P, C, k_vals=[1, 2, 5, 10, 20, 50]):
    S_format = np.empty(shape=(6, 2, 3), dtype=object)
    for k in range(6):
        for i in range(2):
            for j in range(3):
                S_format[k, i, j] = S[i, j][k]
    data = [D_k, n_c, e_c] + [s for s in S_format] + [P, C]
    data_labels = (
        ["K shortest paths length", "Nodes centrality", "Edges centrality",]
        + ["Search information k = {}".format(i) for i in k_vals]
        + ["K-Paths transitivity", "Communicability"]
    )
    formated_data = np.empty(shape=(len(data), 2, 3), dtype=object)
    for k, measure in enumerate(data):
        for i in range(measure.shape[0]):
            for j in range(measure.shape[1]):
                if k == 1:
                    formated_data[k, i, j] = measure[i, j]
                else:
                    index = np.triu_indices(measure[i, j].shape[-1], 1)
                    formated_data[k, i, j] = measure[i, j][:, index[0], index[1]]
    return formated_data, data_labels


def data_scaling(data, with_mean=True, with_std=True):
    data_scaled = np.empty(shape=(data.shape), dtype=object)
    scaler = sklearn.preprocessing.StandardScaler(
        with_mean=with_mean, with_std=with_std
    )
    for k, measure in enumerate(data):
        for i in range(data.shape[1]):
            scaled = scaler.fit_transform(np.vstack(measure[i]))
            n_cases = 0
            for j in range(data.shape[2]):
                data_scaled[k, i, j] = scaled[
                    n_cases : n_cases + measure[i, j].shape[0]
                ]
                n_cases += measure[i, j].shape[0]
    return data_scaled


def PCA(
    data,
    correlation=False,
    ndim=1,
    groups=[0, 1, 2],
    ret_pca=False,
    data_labels=None,
    maps_name=None,
):
    if ret_pca:
        assert data_labels != None, "With ret_pca must provide data_labels"
        assert maps_name != None, "With ret_pca must provide maps_name"
        pca_df, _, _ = data_frame_create(cols=maps_name, rows=data_labels)
    data = np.copy(data[..., groups])
    data_scaled = data_scaling(data, with_std=correlation)
    e_vals, e_vecs = np.zeros(shape=(2,) + data.shape[:-1], dtype=object)
    data_scores = np.empty(shape=data.shape, dtype=object)
    for k, measure_scaled in enumerate(data_scaled):
        for i in range(data.shape[1]):
            pca = sklearn.decomposition.PCA(ndim)
            pca.fit(np.vstack(measure_scaled[i]))
            e_vals[k, i] = pca.explained_variance_ratio_
            e_vecs[k, i] = pca.components_
            # print(np.vstack(measure_scaled[i]).std(axis=0))
            if ret_pca:
                pca_df.loc[data_labels[k], maps_name[i]] = pca
            for j in range(data.shape[2]):
                data_scores[k, i, j] = pca.transform(measure_scaled[i, j])
    if ret_pca:
        return e_vals, e_vecs, data_scores, pca_df
    return e_vals, e_vecs, data_scores


def plot_pca_results(
    data_scores,
    title,
    savefig,
    labels=["Control sano", "Migraña crónica", "Migraña episódica"],
    fs=16,
    figsize=(12, 6),
    markersize=5,
    marker="o",
    groups=[0, 1, 2],
):
    scores = np.squeeze(np.vstack(data_scores))
    index = np.argsort(scores)
    n_cases = 0
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=fs + 2)
    labels = np.array(labels)[groups]
    for i, group in enumerate(labels):
        # ipdb.set_trace()
        group_index = np.where(
            np.logical_and(index >= n_cases, index < n_cases + data_scores[i].shape[0])
        )
        n_cases += data_scores[i].shape[0]
        plt.plot(
            group_index[0],
            scores[index[group_index]],
            marker,
            markersize=markersize,
            label=group,
        )
    plt.ylabel("PC1", fontsize=fs)
    plt.legend(fontsize=fs)
    plt.tick_params(labelsize=fs - 2)
    plt.tight_layout()
    plt.savefig(savefig)


def plot_scores(
    scores,
    correlation,
    data_labels,
    groups_comparison,
    groups_comparison_names,
    comparison_index,
    maps_name,
    e_vals,
    covariance,
    save_dir,
):
    for k, score in enumerate(scores):
        for i in range(scores.shape[1]):
            title = (
                correlation * "Correlation"
                + "PCA - "
                + data_labels[k]
                + " - "
                + groups_comparison_names[comparison_index]
                + " - "
                + maps_name[i]
                + " - "
                + r"$V_r={:.3f}$".format(e_vals[k, i][0])
            )
            savefig = (
                "PCA"
                + "_correlation" * correlation
                + "_covariance" * covariance
                + "\\"
                + "_".join(groups_comparison_names[comparison_index].lower().split())
                + "_"
                + "_".join(data_labels[k].lower().split())
                + "_"
                + "correlation_" * correlation
                + "_".join(maps_name[i].split())
                + ".png"
            )
            plot_pca_results(
                score[i],
                title,
                os.path.join(save_dir, savefig),
                groups=groups_comparison[comparison_index],
            )


def vectorial_measures_principal_component_analysis(
    D_k,
    n_c,
    e_c,
    S,
    P,
    C,
    maps_name,
    save_dir,
    k_vals=[1, 2, 5, 10, 20, 50],
    correlation=False,
    comparison_index=0,
    plot_scores=False,
):
    covariance = not (correlation)
    groups_comparison = ([0, 1], [0, 2], [1, 2])
    groups_comparison_names = ("CS vs MC", "CS vs ME", "MC vs ME")
    groups = np.array(["control sano", "migraña crónica", "migraña episódica"])
    groups = groups[groups_comparison[comparison_index]]
    data, data_labels = format_vectorial_measures(D_k, n_c, e_c, S, P, C, k_vals=k_vals)
    e_vals, e_vecs, scores = PCA(
        data, correlation=correlation, groups=groups_comparison[comparison_index]
    )

    if plot_scores:
        plot_scores(
            scores,
            correlation,
            data_labels,
            groups_comparison,
            groups_comparison_names,
            comparison_index,
            maps_name,
            e_vals,
            covariance,
            save_dir,
        )

    parametros_comparison_groups = [
        "estadístico",
        "p-value",
    ]

    parametros_groups = [
        "estadístico-shapiro",
        "p-value-shapiro",
        "valor medio",
        "mediana",
        "desviación estandar",
    ]

    comparison_groups = [
        "Control sano vs Migraña crónica",
        "Control sano vs Migraña episódica",
        "Migraña crónica vs Migraña episódica",
    ]

    if correlation:
        folder = "PCA_correlation"
        sufix = "correlation"
    else:
        folder = "PCA_covariance"
        sufix = "covariance"

    subfolder = "_".join(comparison_groups[comparison_index].lower().split())

    save_dir = os.path.join(save_dir, folder, subfolder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_groups, df_groups_cols, df_groups_rows = data_frame_create(
        cols=maps_name,
        subcols=parametros_groups,
        rows=data_labels,
        subrows=list(groups),
    )
    df_comparison, df_comparison_cols, df_comparison_rows = data_frame_create(
        cols=maps_name, subcols=parametros_comparison_groups, rows=data_labels,
    )
    ## Analisis intra-grupos
    ## Shapiro, media, mediana y desviacion estandar de las variables
    for i, group in enumerate(groups_comparison[comparison_index]):
        for j, variable in enumerate(scores):
            for k in range(len(maps_name)):
                s, p = sp.stats.shapiro(variable[k, i])
                mean = np.mean(variable[k, i])
                median = np.median(variable[k, i])
                std = np.std(variable[k, i])
                df_groups.loc[df_groups_rows[j * len(groups) + i], maps_name[k]] = [
                    s,
                    p,
                    mean,
                    median,
                    std,
                ]

    ## Analisis inter-grupos
    # T-TEST
    if correlation:
        figure_title = "PCA - Correlation"
        savefig = (
            "_".join(groups_comparison_names[comparison_index].lower().split())
            + "_pca_correlation"
        )
    else:
        figure_title = "PCA - Covariance"
        savefig = (
            "_".join(groups_comparison_names[comparison_index].lower().split())
            + "_pca_covariance"
        )

    if correlation:
        folder = "PCA_correlation"
        sufix = "correlation"
    else:
        folder = "PCA_covariance"
        sufix = "covariance"

    for j, variable in enumerate(scores):
        for k in range(len(maps_name)):
            s, p = sp.stats.ttest_ind(variable[k, 0], variable[k, 1], equal_var=False,)
            if p < 0.05:
                plot_jitter(
                    scores[j : j + 1],
                    figure_title
                    + "- "
                    + data_labels[j]
                    + " - "
                    + maps_name[k]
                    + " - "
                    + "p = {:.2E}".format(p[0]),
                    os.path.join(
                        save_dir,
                        savefig
                        + "_"
                        + "_".join(data_labels[j].lower().split())
                        + "_"
                        + "_".join(maps_name[k].split()),
                    ),
                    rows=["PC1"],
                    cols=groups_comparison_names[comparison_index].split("vs"),
                    data_val=k,
                )
            df_comparison.loc[data_labels[j], maps_name[k]] = np.hstack((s, p))

    df_groups.to_csv(
        os.path.join(save_dir, "shapiro-mean-median-std_{}.csv".format(sufix),)
    )
    df_comparison.to_csv(os.path.join(save_dir, "ttest_{}.csv".format(sufix)))

    return scores, df_groups, df_comparison


def PCA_component_analisys(
    data,
    data_labels,
    correlation,
    maps_name,
    save_dir,
    comparison_index,
    ncols=4,
    nrows=3,
    fig_factor=3,
    fs=16,
    components=20,
    markersize=5,
):
    covariance = not (correlation)
    groups_comparison = ([0, 1], [0, 2], [1, 2])
    groups_comparison_names = ("CS vs MC", "CS vs ME", "MC vs ME")
    comparison_groups = [
        "Control sano vs Migraña crónica",
        "Control sano vs Migraña episódica",
        "Migraña crónica vs Migraña episódica",
    ]
    groups = np.array(["control sano", "migraña crónica", "migraña episódica"])
    groups = groups[groups_comparison[comparison_index]]

    e_vals, e_vecs, scores, pca_df = PCA(
        data,
        correlation=correlation,
        groups=groups_comparison[comparison_index],
        ndim=None,
        ret_pca=True,
        data_labels=data_labels,
        maps_name=maps_name,
    )

    if correlation:
        folder = "PCA_correlation"
        sufix = "correlation"
    else:
        folder = "PCA_covariance"
        sufix = "covariance"

    subfolder = "_".join(comparison_groups[comparison_index].lower().split())

    save_dir = os.path.join(save_dir, folder, subfolder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for map_name in maps_name:
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fig_factor * ncols, fig_factor * nrows),
            sharex=True,
            sharey=True,
        )
        # add a big axes, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )
        plt.grid(False)
        plt.xlabel("Componentes principales", fontsize=fs)
        plt.ylabel("Relacion de varianza total", fontsize=fs)
        for i, data_label in enumerate(data_labels):
            ax = axes[int(i / ncols), i % ncols]
            explained_variances = pca_df.loc[
                data_label, map_name
            ].explained_variance_ratio_
            # cumm_variance = np.cumsum(explained_variances)
            ax.plot(explained_variances[:components], "o", markersize=markersize)
            ax.title.set_text(data_label)

            # plt.plot(cumm_variance)
        plt.suptitle(
            "Correlation" * correlation
            + "Covariance" * covariance
            + " PCA - Importancia de las componentes - Mapeo "
            + map_name,
            fontsize=fs,
        )
        # fig.delaxes(axes[int((i + 1) / ncols), (i + 1) % ncols])
        savefig = "component_ratio_{}_pca_map={}".format(
            "correlation" * correlation + "covariance" * covariance, map_name
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, savefig))
        # plt.show()

    relevant_components = np.empty(shape=e_vecs.shape, dtype=object)

    for j, map_name in enumerate(maps_name):
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fig_factor * ncols, fig_factor * nrows),
            sharex=True,
            sharey=True,
        )

        for i, data_label in enumerate(data_labels):
            ax = axes[int(i / ncols), i % ncols]
            scores_t, scores_p = sp.stats.ttest_ind(
                scores[i, j, 0], scores[i, j, 1], equal_var=False
            )
            relevant_components[i, j] = np.where(scores_p < 1e-5)
            ax.plot(scores_p[:components], "o", markersize=markersize)
            ax.title.set_text(data_label)
            ax.set_yscale("log")

            # plt.plot(cumm_variance)
        plt.suptitle(
            "Correlation" * correlation
            + "Covariance" * covariance
            + " PCA - T-TEST p-value para diferentes componentes - Mapeo "
            + map_name,
            fontsize=fs,
        )
        # fig.delaxes(axes[int((i + 1) / ncols), (i + 1) % ncols])
        savefig = "t-test_p_values_{}_pca_map={}".format(
            "correlation" * correlation + "covariance" * covariance, map_name
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, savefig))
        # plt.show()

    return e_vals, e_vecs, scores, pca_df, relevant_components


def data_std_plot(
    data,
    data_labels,
    maps_name,
    save_dir,
    groups,
    nrows=3,
    ncols=4,
    fig_factor=3,
    fs=16,
    alpha=0.5,
    size=3,
):
    data = np.copy(data[..., groups])
    for k, map_name in enumerate(maps_name):
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fig_factor * ncols, fig_factor * nrows),
            sharex=True,
            sharey=False,
        )
        # add a big axes, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )
        plt.grid(False)
        # plt.xlabel("Varuanza del elemento", fontsize=fs)
        plt.ylabel("Desviación estandar", fontsize=fs)
        for i, measure in enumerate(data):
            ax = axes[int(i / ncols), i % ncols]
            std = np.vstack(measure[k]).std(axis=0)
            # ipdb.set_trace()
            # ax.hist(std)
            sns.stripplot(y=std[std != 0], ax=ax, jitter=True, alpha=alpha, size=size)
            ax.title.set_text(data_labels[i])
            ax.set_yscale("log")
        plt.suptitle(
            " Desviación estandar de los datos - Mapeo " + map_name, fontsize=fs,
        )
        plt.tight_layout()
        savefig = "data_variance_map={}".format(map_name)
        plt.savefig(os.path.join(save_dir, savefig))
        plt.show()


def components_weight_plot(
    data,
    correlation,
    groups_comparison,
    comparison_groups,
    comparison_index,
    data_labels,
    maps_name,
    save_dir,
    n_comps=20,
    plot=True,
):
    e_vals, e_vecs, scores = PCA(
        data,
        correlation=correlation,
        groups=groups_comparison[comparison_index],
        ndim=None,
        ret_pca=False,
        data_labels=data_labels,
        maps_name=maps_name,
    )

    relevant_components = np.empty(shape=e_vecs.shape, dtype=object)

    cota = 1e-5

    for j, map_name in enumerate(maps_name):
        for i, data_label in enumerate(data_labels):
            scores_t, scores_p = sp.stats.ttest_ind(
                scores[i, j, 0], scores[i, j, 1], equal_var=False
            )
            relevant_components[i, j] = np.where(scores_p < cota)[0]

    if plot:
        nodes_relevance = [{}, {}]
        idx = np.triu_indices(82, 1)

        for j, map_name in enumerate(maps_name):
            for i, data_label in enumerate(data_labels):
                if i == 1:
                    for component in relevant_components[i, j]:
                        nodes_relevance[j][
                            data_label + " - Componente " + str(component)
                        ] = np.zeros(shape=(82,))
                        nodes_relevance[j][
                            data_label + " - Componente " + str(component)
                        ] += e_vecs[i, j][component]
                else:
                    for component in relevant_components[i, j]:
                        nodes_relevance[j][
                            data_label + " - Componente " + str(component)
                        ] = np.zeros(shape=(82,))
                        for node in range(82):
                            id_a = np.where(idx[0] == node)
                            id_b = np.where(idx[1] == node)
                            id = np.append(id_a, id_b)
                            nodes_relevance[j][
                                data_label + " - Componente " + str(component)
                            ][node] = np.abs(e_vecs[i, j][component][id]).sum()

        ncols = 3
        fig_factor_x = 4
        fig_factor_y = 3
        fs = 16
        barwidth = 1

        if correlation:
            folder = "PCA_correlation"
            title = "Correlation PCA"
            sufix = "correlation"
        else:
            folder = "PCA_covariance"
            title = "Covariance PCA"
            sufix = "covariance"

        subfolder = "_".join(comparison_groups[comparison_index].lower().split())

        for j, map_name in enumerate(maps_name):
            n_components = len(nodes_relevance[j].keys())
            nrows = int(np.ceil(n_components / ncols))
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(fig_factor_x * ncols, fig_factor_y * nrows),
                sharex=False,
                sharey=True,
            )
            # add a big axes, hide frame
            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axes
            plt.tick_params(
                labelcolor="none", top=False, bottom=False, left=False, right=False
            )
            plt.grid(False)
            plt.xlabel("Nodos", fontsize=fs)
            plt.ylabel("Peso en la dirección elegida", fontsize=fs)
            for i, key in enumerate(nodes_relevance[j].keys()):
                if nrows == 1:
                    ax = axes[i % ncols]
                else:
                    ax = axes[int(i / ncols), i % ncols]
                args = np.argsort(nodes_relevance[j][key])[::-1]
                ax.bar(
                    np.arange(n_comps),
                    nodes_relevance[j][key][args][:n_comps],
                    width=1,
                )
                # ax.set_xticklabels(np.array(atlas)[args][:n_comps], rotation=45, fontsize=8)
                ax.set_xticks(np.arange(n_comps))
                ax.set_xticklabels(
                    np.arange(1, 83)[args][:n_comps], rotation=90, fontsize=10
                )
                ax.title.set_text(key)
                # ax.set_yticks([])

            plt.suptitle(
                title
                + " - Peso de cada nodo en las componentes relevantes - Mapeo "
                + map_name,
                fontsize=fs,
            )
            for n in range(i + 1, ncols * nrows):
                if nrows ==1:
                    fig.delaxes(axes[n % ncols])
                else:
                    fig.delaxes(axes[int(n / ncols), n % ncols])
            
            savefig = "{}_PCA_relevant_components_weights_map_{}_bound_{:.0e}.png".format(
                sufix, map_name, cota
            )
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, folder, subfolder, savefig))
            plt.show()
    else:
        return e_vals, e_vecs, scores, relevant_components


if __name__ == "__main__":
    k = 50
    maps_name = ["inv", "inv_log"]
    groups = ["control sano", "migraña crónica", "migraña episódica"]

    prune_type = "percentage"
    connectomes_control_sano = connectomes_filtered("GRUPO", "control sano", op.eq)
    prune_vals = prune_vals_calc(connectomes_control_sano)
    prune_val = prune_vals[-4]
    k_vals = [1, 2, 5, 10, 20, 50]

    plot_flag = True
    no_outliers = True

    D_k, n_c, e_c, S, P, RE, DE, C, EV = (
        np.empty(shape=(len(maps_name), len(groups)), dtype=object) for i in range(9)
    )

    for i, map_name in enumerate(maps_name):
        for j, group in enumerate(groups):
            (
                D_k[i, j],
                n_c[i, j],
                e_c[i, j],
                S[i, j],
                P[i, j],
                RE[i, j],
                DE[i, j],
                C[i, j],
                EV[i, j],
            ) = load_measures(map_name, group, k, k_vals, prune_val)

    save_dir = "C:\\Users\Tomas\Desktop\Tesis\Programacion\\results\pruning\\figures\connectivity\prune_val={}".format(
        prune_val
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # scalar_measures_analysis(DE, RE, EV, save_dir, no_outliers, plot_flag)

    correlation = False
    comparison_index = 0  # 0 CS vs MC - 1 CS vs ME - 2 MC vs ME
    plot_scores = False

    data, data_labels = format_vectorial_measures(D_k, n_c, e_c, S, P, C, k_vals=k_vals)

    connectomes = np.empty(shape=(1, 2, 3), dtype=object)
    for i, group in enumerate(groups):
        connectome = connectomes_filtered("GRUPO", group, op.eq)
        index = np.triu_indices_from(connectome[0], 1)
        connectome = connectome[:, index[0], index[1]]
        connectomes[0, 0, i] = connectome
        connectomes[0, 1, i] = connectome

    data = np.concatenate((data, connectomes))
    data_labels += ["Conectomas sin tratar"]

    covariance = not (correlation)
    groups_comparison = ([0, 1], [0, 2], [1, 2])
    groups_comparison_names = ("CS vs MC", "CS vs ME", "MC vs ME")
    comparison_groups = [
        "Control sano vs Migraña crónica",
        "Control sano vs Migraña episódica",
        "Migraña crónica vs Migraña episódica",
    ]
    groups = np.array(["control sano", "migraña crónica", "migraña episódica"])
    groups = groups[groups_comparison[comparison_index]]

    # atlas = load_atlas()

    e_vals, e_vecs, scores, relevant_components = components_weight_plot(
        data,
        correlation,
        groups_comparison,
        comparison_groups,
        comparison_index,
        data_labels,
        maps_name,
        save_dir,
        n_comps=20,
        plot=True,
    )

    # list_path = r"C:\Users\Tomas\Desktop\Tesis\datos\conjunto_datos_conectomica_migranya\lista_de_casos.xlsx"
    # cases = pd.read_excel(list_path)
    # cases = cases[cases["GRUPO"] != "Migraña episódica"]
    # sex = cases["SEXO"].values

    # sex_scores = np.empty(shape=scores.shape, dtype=object)
    # sex_p = np.empty(shape=scores.shape[:-1], dtype=object)
    # for j in range(len(maps_name)):
    #     for i in range(len(data_labels)):
    #         all_scores = np.vstack(scores[i, j])[:, 0]
    #         sex_scores[i, j, 0] = all_scores[sex == "V"]
    #         sex_scores[i, j, 1] = all_scores[sex == "M"]
    #         _, sex_p[i, j] = sp.stats.ttest_ind(
    #             sex_scores[i, j, 0], sex_scores[i, j, 1], equal_var=False
    #         )

    # scores, df_groups, df_comparison = vectorial_measures_principal_component_analysis(
    #     D_k,
    #     n_c,
    #     e_c,
    #     S,
    #     P,
    #     C,
    #     maps_name,
    #     save_dir=save_dir,
    #     k_vals=k_vals,
    #     correlation=correlation,
    #     comparison_index=comparison_index,
    #     plot_scores=plot_scores,
    # )

    # PCA_component_analisys(
    #     data=data,
    #     data_labels=data_labels,
    #     correlation=correlation,
    #     maps_name=maps_name,
    #     save_dir=save_dir,
    #     comparison_index=comparison_index,
    #     ncols=4,
    #     nrows=3,
    #     fig_factor=3,
    #     fs=16,
    #     components=20,
    #     markersize=5,
    # )

    # data_std_plot(
    #     data=data,
    #     data_labels=data_labels,
    #     maps_name=maps_name,
    #     save_dir=save_dir,
    #     groups=groups_comparison[comparison_index],
    #     nrows=3,
    #     ncols=4,
    #     fig_factor=3,
    #     fs=16,
    #     alpha=0.5,
    #     size=3,
    # )

