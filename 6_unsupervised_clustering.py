#%%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

#%%
pca_df = pd.read_csv("data/processed/pca_df.csv")

# %%
def get_base_x_y(pca_df):
    filter_col = [col for col in pca_df if col.startswith("PC")]
    cat_col = [col for col in pca_df if not col.startswith("PC")]
    y_df = pca_df[cat_col]
    x_df = pca_df[filter_col]
    return x_df, y_df


def get_optimal_n_cluster(x, maxClusters, modelStr):
    scores = []
    print("starting: " + modelStr)

    for k in range(2, maxClusters + 1):
        if modelStr == "KMeans":
            model = KMeans(k, random_state=0)
        elif modelStr == "GaussianMixture":
            model = GaussianMixture(k, random_state=0)
        elif modelStr == "KMeans_pipe":
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("minMax", MinMaxScaler()),
                    ("pca", PCA()),
                    ("kmeans", KMeans(k, random_state=0)),
                ]
            )
        elif modelStr == "Gaussian_pipe":
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("minMax", MinMaxScaler()),
                    ("pca", PCA()),
                    ("kmeans", GaussianMixture(k, random_state=0)),
                ]
            )

        predValues = model.fit_predict(x)
        dbScore = davies_bouldin_score(x, predValues)
        chScore = calinski_harabasz_score(x, predValues)
        scores.append([modelStr, k, dbScore, chScore])

    print("finished: " + modelStr)
    result = pd.DataFrame(scores, columns=["model", "clusters", "db_score", "ch_score"])

    return result


def get_n_cluster_score_per_model(x, modelList, maxClusters, modelStr2):
    final_df = pd.DataFrame()
    for modelStr in modelList:
        model_df = get_optimal_n_cluster(x, maxClusters, modelStr)
        if final_df.empty:
            final_df = model_df
        else:
            final_df = pd.concat([final_df, model_df])

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(14, 5))
    sns.lineplot(
        data=final_df, x="clusters", y="db_score", hue="model", marker="o", ax=ax1
    )
    sns.lineplot(
        data=final_df, x="clusters", y="ch_score", hue="model", marker="o", ax=ax2
    )
    if modelStr2 in ["KMeans", "GaussianMixture"]:
        fig.suptitle(
            "Inital Clustering by {} - N Clusters vs. Davies Bouldin Calinski Harabasz Score by KMeans and Gaussian Models".format(
                modelStr2
            )
        )
        fig.savefig("data/results/images/best_cluster_{}.jpg".format(modelStr2))
    elif modelStr2 in ["Agg Gaussian"]:
        fig.suptitle(
            "Inital Clustering by {} - N Clusters vs. Davies Bouldin Calinski Harabasz Score by KMeans and Gaussian Models".format(
                modelStr2
            )
        )
        fig.savefig("data/results/images/best_cluster_{}.jpg".format(modelStr2))
    else:
        fig.suptitle(
            "N Clusters vs. Davies Bouldin Calinski Harabasz Score by KMeans and Gaussian Models"
        )
        fig.savefig("data/results/images/best_cluster_1.jpg")

    return final_df


#%%
modelList = ["KMeans", "GaussianMixture"]
results_per_model = get_n_cluster_score_per_model(
    get_base_x_y(pca_df)[0], modelList, 20, ""
)


#%%
def get_cluster_for_champ_build_from_agg(pca_df, n_clusters, modelStr):
    x, y = get_base_x_y(pca_df)

    if modelStr == "KMeans":
        gm = KMeans(n_clusters, random_state=0)
    elif modelStr == "GaussianMixture":
        gm = GaussianMixture(n_clusters, random_state=0)
    gm.fit(x)
    y_copy = y.copy()
    y_copy["predClasses"] = gm.predict(x)
    y_copy["predClasses"] = y_copy.predClasses.astype("category")
    aggChampBuild = y_copy.groupby(["champion", "build", "predClasses"]).size()
    aggChampBuild = aggChampBuild.reset_index().rename(columns={0: "count"})
    pivotChampBuild = aggChampBuild.pivot(
        index=["champion", "build"], columns="predClasses", values="count"
    )
    flattenChampBuild = pd.DataFrame(pivotChampBuild.to_records())
    flattenChampBuild["total"] = flattenChampBuild.sum(axis=1)
    flattenChampBuild = flattenChampBuild[flattenChampBuild["total"] > 0].reset_index(
        drop=True
    )
    return flattenChampBuild


def get_n_cluster_score_per_model_from_agg(
    pca_df, n_clusters, maxClusters, modelList, modelList_pipe
):
    for modelStr in modelList:
        pred_df = get_cluster_for_champ_build_from_agg(pca_df, n_clusters, modelStr)
        result = get_n_cluster_score_per_model(
            pred_df.drop(columns=["champion", "build", "total"]),
            modelList_pipe,
            maxClusters,
            modelStr,
        )
    return result


#%%
n_clusters = 9
maxClusters = 20
modelList = ["KMeans", "GaussianMixture"]
modelList_pipe = ["KMeans_pipe", "Gaussian_pipe"]
agg_results = get_n_cluster_score_per_model_from_agg(
    pca_df, n_clusters, maxClusters, modelList, modelList_pipe
)

#%%
def get_champBuild_cluster_from_agg_clusters(
    pca_df, n_cluster_1, n_cluster_2, modelStr
):

    cluster_per_champ_build = get_cluster_for_champ_build_from_agg(
        pca_df, n_cluster_1, "KMeans"
    )

    clusterBase = cluster_per_champ_build.copy()
    for n in range(n_cluster_1):
        clusterBase["C%s" % (n)] = clusterBase[str(n)] / clusterBase["total"]
        clusterBase = clusterBase.drop(columns=[str(n)])
    clusterBase = clusterBase.drop(columns=["total"])

    pcaPipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("minMax", MinMaxScaler()),
            ("pca", PCA(n_components=2)),
        ]
    )

    x_cluster = cluster_per_champ_build.drop(columns=["champion", "build", "total"])
    y_cluster = cluster_per_champ_build.copy()[["champion", "build"]]
    pcaResults = pcaPipe.fit_transform(x_cluster)

    if modelStr == "KMeans_pipe":
        model = KMeans(n_cluster_2, random_state=0)
    elif modelStr == "Gaussian_pipe":
        model = GaussianMixture(n_cluster_2, random_state=0)

    y_cluster["predCluster"] = model.fit_predict(pcaResults)
    y_cluster["predCluster"] = y_cluster.predCluster.astype("category")
    results = y_cluster.join(pd.DataFrame(pcaResults, columns=["PC0", "PC1"]))

    return results, clusterBase


#%%
results, clusterBase = get_champBuild_cluster_from_agg_clusters(
    pca_df, 9, 8, "KMeans_pipe"
)
results.to_csv("data/results/cluster_per_champ_build.csv")
clusterBase.to_csv("data/results/cluster_percents_per_champ_build.csv")

#%%
ax = sns.scatterplot(data=results, x="PC0", y="PC1", hue="predCluster")
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)
fig = ax.get_figure()
fig.suptitle("Cluster by Champion and Builds")
fig.savefig("data/results/images/championCluster_pct_kmeans.jpg")


#%%
def predict_gaussian_for_n_clusters(df, n_clusters):
    x, y = get_base_x_y(df)
    gm = GaussianMixture(n_components=n_clusters, random_state=0)
    gm.fit(x)

    y_copy = y.copy()
    y_copy["predClasses"] = gm.predict(x)
    y_copy["predClasses"] = y_copy.predClasses.astype("category")

    predProb_df = gm.predict_proba(x)
    cluster_cols = ["c{}".format(n) for n in list(range(n_clusters))]
    cluster_df = pd.DataFrame(predProb_df, columns=cluster_cols)

    result = y_copy.join(cluster_df)
    return result


def lowerBound(mean, std):
    lowerBound = mean - std
    if lowerBound < 0:
        return 0
    else:
        return lowerBound


def agg_cluster_proba_per_champ_build(df, n_clusters):
    aggMeanStd_dict = {}
    colMeanStd_list = []
    for x in range(n_clusters):
        clusterStr = "c{}".format(x)
        aggMeanStd_dict[clusterStr] = ["mean", "std"]
        colMeanStd_list.append([clusterStr, clusterStr + "_mean", clusterStr + "_std"])

    results = df.groupby(["champion", "build"]).agg(aggMeanStd_dict).reset_index()
    results.columns = ["_".join(col).strip() for col in results.columns.values]
    results = results.rename(columns={"champion_": "champion"})
    results = results.rename(columns={"build_": "build"})
    for clusterList in colMeanStd_list:
        cStr, cMean, cStd = clusterList
        results[cStr + "_upper"] = round(results[cMean] + results[cStd], 5)
        results[cStr + "_avg"] = round(results[cMean], 5)
        results[cStr + "_lower"] = results.apply(
            lambda x: lowerBound(x[cMean], x[cStd]), axis=1
        )
        results = results.drop(columns=[cMean, cStd])
    return results


def apply_pca_pipeline(df, n_cluster, modelStr):
    x = df.copy().drop(columns=["champion", "build"])
    y = df.copy()[["champion", "build"]]

    pcaPipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("minMax", MinMaxScaler()),
            ("pca", PCA(n_components=2)),
        ]
    )

    pca_x = pcaPipe.fit_transform(x)
    if modelStr == "KMeans":
        model = KMeans(n_cluster, random_state=0)
    elif modelStr == "Gaussian":
        model = GaussianMixture(n_cluster, random_state=0)

    y["predCluster"] = model.fit_predict(pca_x)
    y["predCluster"] = y.predCluster.astype("category")
    pca_results = y.join(pd.DataFrame(pca_x, columns=["PC0", "PC1"]))
    return pca_results


#%%
modelStrList = ["KMeans_pipe", "Gaussian_pipe"]
n_clusters = 9
maxClusters = 20
clustersAvg_df = predict_gaussian_for_n_clusters(pca_df, n_clusters)
clustersGrouped = agg_cluster_proba_per_champ_build(clustersAvg_df, n_clusters)
optimal_cluster_results = get_n_cluster_score_per_model(
    clustersGrouped.copy().drop(columns=["champion", "build"]),
    modelStrList,
    maxClusters,
    "Agg Gaussian",
)


#%%
clustersGrouped = pd.read_csv("data/results/cluster_prob_per_champ_build.csv")

x = clustersGrouped.copy().drop(columns=["champion", "build"])
y = clustersGrouped.copy()[["champion", "build"]]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(x)

#%%
pca = PCA()
pca.fit(data_scaled)

# Calculate the explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Plot the explained variance
plt.plot(np.cumsum(explained_variance))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Elbow Method for PCA')
plt.show()

#%%
pca_comp= 7
pca = PCA(n_components=pca_comp)
pca_fitted = pca.fit_transform(data_scaled)
pca_results = y.join(pd.DataFrame(pca_fitted, columns=["PC{}".format(x) for x in range(pca_comp)]))
pca_results.to_csv(
    "data/results/pca_per_champ_build.csv", index=False
)

#%%
## GRAPH TSNE AND KMEANS
tsne = TSNE(n_components=2, random_state=42)
tsne_fitted = tsne.fit_transform(pca_fitted)
kmeans = KMeans(n_clusters=10, random_state=42)
y["predCluster"] = kmeans.fit_predict(tsne_fitted)
y["predCluster"] = y.predCluster.astype("category")
tsne_results = y.join(pd.DataFrame(tsne_fitted, columns=["TSNE_0", "TSNE_1"]))

ax = sns.scatterplot(data=tsne_results, x="TSNE_0", y="TSNE_1", hue="predCluster")
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)
fig = ax.get_figure()
fig.suptitle("Cluster by Champion and Builds")
fig.savefig("data/results/images/championCluster_tsne.jpg")

#%%
# pcaGaussianClusters = apply_pca_pipeline(clustersGrouped, 9, "KMeans")
# pcaGaussianClusters.to_csv(
#     "data/results/gaussian_cluster_per_champ_build.csv", index=False
# )
# clustersGrouped.to_csv("data/results/cluster_prob_per_champ_build.csv", index=False)

# %%
# GET LIST OF CHAMP + BUILDS AND FORMAT FOR README FILE
tsne_results["champBuild"] = (
    tsne_results["champion"] + " (" + tsne_results["build"].str[5:] + ")"
)
clusterChampsListed = (
    tsne_results.groupby("predCluster")["champBuild"]
    .apply(lambda x: ", ".join(x))
    .reset_index()
)
clusterChampsListed["predCluster"] = clusterChampsListed.predCluster.astype("string")
clusterChampsListed['|Cluster|Champion and Build|'] = "|"  + clusterChampsListed['predCluster'] + "|" + clusterChampsListed['champBuild'] + "|"
clusterChampsListed = clusterChampsListed[['|Cluster|Champion and Build|']]
clusterChampsListed
# %%
clusterChampsListed.to_csv("data/results/listed_champBuild_clusters_tsne.csv", index=False)
# %%
