#%%
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#%%
pca_df = pd.read_csv("data/processed/pca_df.csv")

# %%
filter_col = [col for col in pca_df if col.startswith("PC")]
cat_col = [col for col in pca_df if not col.startswith("PC")]

#%%
y_df = pca_df[cat_col]
x_df = pca_df[filter_col]

#%%
def get_optimal_n_cluster(x, maxClusters, modelStr):
    scores = []
    for k in range(2, maxClusters + 1):
        if modelStr == "KMeans":
            model = KMeans(k, random_state=0)
        if modelStr == "GaussianMixture":
            model = GaussianMixture(k, random_state=0)
        predValues = model.fit_predict(x)
        dbScore = davies_bouldin_score(x, predValues)
        chScore = calinski_harabasz_score(x, predValues)
        scores.append([modelStr, k, dbScore, chScore])
    result = pd.DataFrame(scores, columns=["model", "clusters", "db_score", "ch_score"])
    return result


kmeans_df = get_optimal_n_cluster(x_df, 30, "KMeans")
gaussian_df = get_optimal_n_cluster(x_df, 30, "GaussianMixture")
result = pd.concat([kmeans_df, gaussian_df])

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
    cluster_cols = ["cluster_{}".format(n) for n in list(range(n_clusters))]
    cluster_df = pd.DataFrame(predProb_df, columns=cluster_cols)

    result = y_copy.join(cluster_df)

    return result


n_clusters = 12
result = predict_gaussian_for_n_clusters(x_df, y_df, n_clusters)

#%%
df_copy = x_df.copy()
df_copy = result.join(df_copy)
sns.pairplot(
    df_copy[["predClasses", "PC0", "PC1", "PC2", "PC3", "PC4"]], hue="predClasses"
)

# %%
groupedGames = result.groupby(["gameId", "teamId", "win", "patch"]).sum().reset_index()
y = groupedGames["win"]
X = groupedGames.drop(columns=["gameId", "teamId", "win", "patch"])
# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
# %%
confusion_matrix(y_test, y_pred)
# %%
confusion_matrix(y_train, clf.predict(X_train))
# %%
