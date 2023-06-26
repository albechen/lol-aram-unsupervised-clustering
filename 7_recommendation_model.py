#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#%%
train = pd.read_csv("data/processed/pca_df.csv")
test = pd.read_csv("data/processed/baseFeatures_test.csv")
pca_per_champ_build = pd.read_csv("data/results/pca_per_champ_build.csv")

#%%
def join_games_with_clustered_champs(df, cluster_df):
    catCols = ["win", "teamId", "gameId", "champion", "build"]
    df = df[catCols]
    joined_df = df.merge(cluster_df, on=["champion", "build"], how="left")
    agg_df = joined_df.groupby(["gameId", "teamId", "win"]).mean().reset_index()

    return agg_df


def get_x_y_values(df):
    filter_col = [col for col in df if col.startswith("PC")]
    x = df[filter_col]
    y = df[["win"]]
    return x, y


def scale_normailze_adjust_skew(df):
    x_df, y_df = get_x_y_values(df)
    x_copy = x_df.copy()

    skewDict = x_copy.skew().to_dict()
    for x in skewDict:
        if skewDict[x] > 0.5:
            x_copy[x] = np.log(x_copy[x] + 1)

    scaler = StandardScaler()
    minMax = MinMaxScaler()
    x_scaled = minMax.fit_transform(scaler.fit_transform(x_copy))

    x_scaled_df = pd.DataFrame(x_scaled, columns=x_copy.columns)
    result = y_df.join(x_scaled_df)
    return result


def get_pca_values_per_row(df, pca_comp):
    x_df, y_df = get_x_y_values(df)

    pca = PCA(n_components=pca_comp)
    pca_fitted = pca.fit_transform(x_df)
    print(sum(pca.explained_variance_ratio_))
    minMax = MinMaxScaler()
    pca_scaled = minMax.fit_transform(pca_fitted)
    pca_df = pd.DataFrame(pca_scaled, columns=["PC" + str(x) for x in range(pca_comp)])

    result = y_df.join(pca_df)
    return result


#%%
pca_comp = 10
pca_df = (
    train.pipe(join_games_with_clustered_champs, pca_per_champ_build)
    .pipe(scale_normailze_adjust_skew)
    .pipe(get_pca_values_per_row, pca_comp)
)

# %%
aggDF = join_games_with_clustered_champs(train, pca_per_champ_build)
x, y = get_x_y_values(aggDF)

aggDF_test = join_games_with_clustered_champs(test, pca_per_champ_build)
x_test, y_test = get_x_y_values(aggDF_test)

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
x_test_scaled = scaler.transform(x_test)

clf = LogisticRegression(random_state=0)
clf.fit(x_scaled, y)

#%%
modeldf = aggDF_test[["gameId", "teamId", "win"]].copy()
modeldf = modeldf.join(
    pd.DataFrame(clf.predict_proba(x_test_scaled), columns=["loss", "winPercent"])
).drop(columns="loss")
pivotModelDF = modeldf.pivot(index=["gameId"], columns="win", values="winPercent")
pivotModelDF = pd.DataFrame(pivotModelDF.to_records())
pivotModelDF["winHigherPercent"] = pivotModelDF["True"] > pivotModelDF["False"]

len(pivotModelDF[pivotModelDF["winHigherPercent"] == True]) / len(pivotModelDF)

# %%
#### NAEIVE MODEL ####
winLoseCount = train.groupby(["champion", "build", "win"]).size().reset_index()
winPercent = winLoseCount.pivot(index=["champion", "build"], columns="win", values=0)
winPercent = pd.DataFrame(winPercent.to_records())
winPercent["winPercent"] = winPercent["True"] / (
    winPercent["False"] + winPercent["True"]
)
winPercent
# %%
mergedWinRates = test.merge(winPercent, on=["champion", "build"], how="left")
groupedAvgWinRates = (
    mergedWinRates.groupby(["gameId", "teamId", "win"])
    .agg({"winPercent": "mean"})
    .reset_index()
)
pivotAvgWinRates = groupedAvgWinRates.pivot(
    index=["gameId"], columns="win", values="winPercent"
)
pivotAvgWinRates = pd.DataFrame(pivotAvgWinRates.to_records())
pivotAvgWinRates["winHigherPercent"] = (
    pivotAvgWinRates["True"] > pivotAvgWinRates["False"]
)
# %%
len(pivotAvgWinRates[pivotAvgWinRates["winHigherPercent"] == True]) / len(
    pivotAvgWinRates
)
# %%
