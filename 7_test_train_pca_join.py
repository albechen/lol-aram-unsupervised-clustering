#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
