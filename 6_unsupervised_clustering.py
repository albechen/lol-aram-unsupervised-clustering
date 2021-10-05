#%%
import pandas as pd
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
# kmeans_df = get_optimal_n_cluster(x_df, 2, "AgglomerativeClustering")
# gaussian_df = get_optimal_n_cluster(x_df, 10, "Birch")
# result = pd.concat([kmeans_df, gaussian_df])


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
n_clusters = 13
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

    return results


#%%
result_cluster_per_champ = get_champBuild_cluster_from_agg_clusters(
    pca_df, 12, 9, "KMeans_pipe"
)
result_cluster_per_champ.to_csv("data/results/cluster_per_champ_build.csv")

#%%
ax = sns.scatterplot(data=result_cluster_per_champ, x="PC0", y="PC1", hue="predCluster")
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)
fig = ax.get_figure()
fig.suptitle("Cluster by Champion and Builds")
fig.savefig("data/results/images/championCluster.jpg")