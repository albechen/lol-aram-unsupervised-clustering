#%%
import pandas as pd

rawFlatStats = pd.read_csv("data/interim/rawFlatStats.csv")

#%%
def get_and_filter_patch(df, patch):
    df["patch"] = df["gameVersion"].apply(lambda x: ".".join(x.split(".", 2)[:2]))
    filtered_df = df[df["patch"] >= patch]
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df


def filter_games_without_all_player_info(df):
    aggPlayer = df.groupby("gameId").agg({"accountId": "count"})
    aggPlayer = aggPlayer.rename(columns={"accountId": "playerCount"}).reset_index()
    playerCount_df = df.merge(aggPlayer, on="gameId", how="left")
    filtered_df = playerCount_df[playerCount_df["playerCount"] == 10].reset_index(
        drop=True
    )
    return filtered_df


minPatch = "11.1"
filtered_df = rawFlatStats.pipe(get_and_filter_patch, minPatch).pipe(
    filter_games_without_all_player_info
)
filtered_df
#%%
filtered_df.to_csv("data/interim/filtered_rawFlatStats.csv", index=False)
# %%
