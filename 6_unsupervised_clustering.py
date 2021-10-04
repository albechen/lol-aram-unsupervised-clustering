#%%
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import (
    KMeans,
    SpectralClustering,
    AgglomerativeClustering,
    Birch,
    AffinityPropagation,
)

#%%
pca_df = pd.read_csv("data/processed/pca_df.csv")
test_df = pd.read_csv("data/processed/baseFeatures_test.csv")

# %%
filter_col = [col for col in pca_df if col.startswith("PC")]
cat_col = [col for col in pca_df if not col.startswith("PC")]
y_df = pca_df[cat_col]
x_df = pca_df[filter_col]

#%%
def get_optimal_n_cluster(x, maxClusters, modelStr):
    scores = []
    print("starting: " + modelStr)

    if modelStr == "AffinityPropagation":
        model = AffinityPropagation(random_state=0)
        predValues = model.fit_predict(x)
        k = len(set(predValues))
        dbScore = davies_bouldin_score(x, predValues)
        chScore = calinski_harabasz_score(x, predValues)
        scores.append([modelStr, k, dbScore, chScore])
    elif modelStr == "Birch":
        model = Birch()
        predValues = model.fit_predict(x)
        k = len(set(predValues))
        dbScore = davies_bouldin_score(x, predValues)
        chScore = calinski_harabasz_score(x, predValues)
        scores.append([modelStr, k, dbScore, chScore])

    for k in range(2, maxClusters + 1):
        print(k)
        if modelStr == "KMeans":
            model = KMeans(k, random_state=0)
        elif modelStr == "GaussianMixture":
            model = GaussianMixture(k, random_state=0)
        elif modelStr == "SpectralClustering":
            model = SpectralClustering(k, random_state=0)
        elif modelStr == "AgglomerativeClustering":
            model = AgglomerativeClustering(k)

        elif modelStr == "AffinityPropagation":
            model = AffinityPropagation(k, random_state=0)

        predValues = model.fit_predict(x)
        dbScore = davies_bouldin_score(x, predValues)
        chScore = calinski_harabasz_score(x, predValues)
        scores.append([modelStr, k, dbScore, chScore])

    print("finished: " + modelStr)
    result = pd.DataFrame(scores, columns=["model", "clusters", "db_score", "ch_score"])
    return result


def get_n_cluster_score_per_model(x_df, modelList, maxClusters):
    final_df = pd.DataFrame()
    for modelStr in modelList:
        model_df = get_optimal_n_cluster(x_df, maxClusters, modelStr)
        if final_df.empty:
            final_df = model_df
        else:
            final_df = pd.concat([final_df, model_df])
    return final_df


#%%
modelList = ["KMeans", "GaussianMixture"]
result = get_n_cluster_score_per_model(x_df, modelList, 20)

#%%
# kmeans_df = get_optimal_n_cluster(x_df, 2, "AgglomerativeClustering")
# gaussian_df = get_optimal_n_cluster(x_df, 10, "Birch")
# result = pd.concat([kmeans_df, gaussian_df])

#%%
sns.lineplot(data=result, x="clusters", y="db_score", hue="model", marker="o")

#%%
sns.lineplot(data=result, x="clusters", y="ch_score", hue="model", marker="o")

#%%
def predict_gaussian_for_n_clusters(x, y, n_clusters):
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


def cluster_by_champ_build_dataset(df, n_clusters):
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


n_clusters = 7
clustersAvg_df = predict_gaussian_for_n_clusters(x_df, y_df, n_clusters)
clustersGrouped = cluster_by_champ_build_dataset(clustersAvg_df, n_clusters)

#%%
clustersGrouped.to_csv("data/processed/clustered_champ_build.csv", index=False)
#%%
# df_copy = x_df.copy()
# df_copy = result.join(df_copy)
# sns.pairplot(
#     df_copy[["predClasses", "PC0", "PC1", "PC2", "PC3", "PC4"]], hue="predClasses"
# )