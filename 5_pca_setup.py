#%%
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

#%%
raw_df = pd.read_csv("data/processed/baseFeature_df.csv")
# %%
def test_train_pca_split(df, testPercent):
    listGames = list(set(df["gameId"]))
    numTestGames = round(testPercent * len(listGames))
    randomTestGames = random.sample(listGames, k=numTestGames)

    testDF = df[df["gameId"].isin(randomTestGames)].reset_index(drop=True)
    testDF.to_csv("data/processed/baseFeatures_test.csv", index=False)
    pca_df = df[~df["gameId"].isin(randomTestGames)].reset_index(drop=True)

    return pca_df


def get_x_y_values(df):
    catCols = ["gameId", "teamId", "win", "patch", "champion", "build"]
    y_raw = df[catCols]

    filter_col = [col for col in df if col.startswith("pct") or col.startswith("per")]
    x_raw = df[filter_col]
    return x_raw, y_raw


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
pca_comp = 15
testPercent = 0.1
pca_df = (
    raw_df.pipe(test_train_pca_split, testPercent)
    .pipe(scale_normailze_adjust_skew)
    .pipe(get_pca_values_per_row, pca_comp)
)

#%%
pca_df.to_csv("data/processed/pca_df.csv", index=False)

# %%
