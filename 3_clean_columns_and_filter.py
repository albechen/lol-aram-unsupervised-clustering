#%%
import pandas as pd
import pickle

rawFlatStats = pd.read_csv("data/interim/rawFlatStats.csv")

#%%
def open_pickle(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


def get_and_filter_patch(df, patch):
    df["patch"] = df["gameVersion"].apply(lambda x: ".".join(x.split(".", 2)[:2]))
    filtered_df = df[df["patch"] >= patch]
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df


def get_champion_names(df):
    champion_dict = open_pickle("data/interim/champion_dict.pkl")
    df["champion"] = df["championId"].apply(lambda x: champion_dict[str(x)])
    return df


def get_build_breakdown(df):
    df["itemList"] = df[
        [
            "item0",
            "item1",
            "item2",
            "item3",
            "item4",
            "item5",
            "item6",
        ]
    ].values.tolist()

    itemDF = pd.read_csv("data/interim/itemsType.csv")
    itemDict = itemDF.set_index("ID").to_dict()

    def get_item_build_by_player(itemList):
        playerItemType = {
            "ad": 0,
            "bruiser": 0,
            "ap": 0,
            "tank": 0,
            "support": 0,
        }
        for item in itemList:
            try:
                playerItemType[itemDict["Type"][item]] += int(itemDict["Gold"][item])
            except:
                pass
        itemList = list(playerItemType.values())
        if sum(itemList) > 0:
            itemList = [x / sum(itemList) for x in itemList]
            itemList.append(0)
        else:
            itemList.append(1)
        return itemList

    df["itemList"] = df["itemList"].apply(lambda x: get_item_build_by_player(x))
    buildList = [
        "item_ad",
        "item_bruiser",
        "item_ap",
        "item_tank",
        "item_support",
        "item_none",
    ]
    df[buildList] = pd.DataFrame(df["itemList"].tolist(), index=df.index)
    df = df.drop(["itemList"], axis=1)
    df["build"] = df[buildList].idxmax(axis=1)
    return df


def filter_percent_champ_build(df, minPctBuild):
    print(len(df))
    dfGrouped = df.groupby(["champion", "build"]).agg({"win": "count"})
    dfPct = dfGrouped.groupby(level=0).apply(lambda x: x / float(x.sum()))
    dfPct = dfPct.rename(columns={"win": "build_percent"}).reset_index()
    dfMerge = df.merge(dfPct, on=["champion", "build"], how="left")
    result = dfMerge[dfMerge["build_percent"] >= minPctBuild].reset_index(drop=True)
    print(len(result))
    return result


def filter_games_without_all_player_info(df):
    aggPlayer = df.groupby(["gameId", "teamId"]).agg({"accountId": "count"})
    aggPlayer = aggPlayer.rename(columns={"accountId": "playerCount"}).reset_index()
    playerCount_df = df.merge(aggPlayer, on=["gameId", "teamId"], how="left")
    filtered_df = playerCount_df[playerCount_df["playerCount"] == 5]
    result = filtered_df.drop(columns=["playerCount"]).reset_index(drop=True)
    print(len(result))
    return result


minPatch = "11.1"
minPctBuild = 0.1
filtered_df = (
    rawFlatStats.pipe(get_and_filter_patch, minPatch)
    .pipe(get_champion_names)
    .pipe(get_build_breakdown)
    .pipe(filter_percent_champ_build, minPctBuild)
    .pipe(filter_games_without_all_player_info)
)
filtered_df

#%%
filtered_df.to_csv("data/interim/filtered_rawFlatStats.csv", index=False)
# %%
