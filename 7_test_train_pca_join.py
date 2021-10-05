#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

pca_df = pd.read_csv("data/processed/pca_df.csv")


def get_base_x_y(pca_df):
    filter_col = [col for col in pca_df if col.startswith("PC")]
    cat_col = [col for col in pca_df if not col.startswith("PC")]
    y_df = pca_df[cat_col]
    x_df = pca_df[filter_col]
    return x_df, y_df


#%%
def predict_kmeans_for_n_clusters(df, n_clusters):
    x, y = get_base_x_y(df)
    gm = KMeans(n_components=n_clusters, random_state=0)
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
clustersAvg_df = predict_kmeans_for_n_clusters(pca_df, n_clusters)
clustersGrouped = cluster_by_champ_build_dataset(clustersAvg_df, n_clusters)

#%%
clustersGrouped.to_csv("data/processed/clustered_champ_build.csv", index=False)
#%%
# df_copy = x_df.copy()
# df_copy = result.join(df_copy)
# sns.pairplot(
#     df_copy[["predClasses", "PC0", "PC1", "PC2", "PC3", "PC4"]], hue="predClasses"
# )
#%%
clustersGrouped_df = pd.read_csv("data/processed/clustered_champ_build.csv")
train = pd.read_csv("data/processed/pca_df.csv")
test = pd.read_csv("data/processed/baseFeatures_test.csv")

#%%
def join_games_with_clustered_champs(df, cluster_df):
    catCols = ["win", "teamId", "gameId", "champion", "build"]
    df = df[catCols]
    joined_df = df.merge(cluster_df, on=["champion", "build"], how="left")
    agg_df = joined_df.groupby(["gameId", "teamId", "win"]).sum().reset_index()
    y = agg_df[["win"]]
    x = agg_df.drop(columns=["win", "teamId", "gameId"])
    onlyXY = y.join(x)
    return agg_df, x, y, onlyXY


# %%
agg_df, x, y, onlyXY = join_games_with_clustered_champs(train, clustersGrouped_df)
agg_df_test, x_test, y_test, onlyXY_test = join_games_with_clustered_champs(
    test, clustersGrouped_df
)

# create distplots
for column in x.columns:
    plt.figure()
    sns.displot(data=onlyXY, x=column, hue="win")
# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("minMax", MinMaxScaler()),
        ("pca", PCA()),
        ("rfc", RandomForestClassifier()),
    ]
)
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe.fit(x, y)

pipe.score(x_test, y_test)
# %%
