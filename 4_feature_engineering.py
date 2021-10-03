#%%
import pandas as pd
import pickle

#%%
flatStats = pd.read_csv("data/interim/filtered_rawFlatStats.csv")

# %%
def open_pickle(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


statList = {
    "kills": "kill",
    "deaths": "death",
    "assists": "assist",
    "goldEarned": "gold",
    "totalMinionsKilled": "cs",
    "totalDamageTaken": "ttlTank",
    "magicalDamageTaken": "apTank",
    "physicalDamageTaken": "adTank",
    "trueDamageTaken": "trueTank",
    "totalDamageDealtToChampions": "ttlDmg",
    "magicDamageDealtToChampions": "apDmg",
    "physicalDamageDealtToChampions": "adDmg",
    "trueDamageDealtToChampions": "trueDmg",
    "timeCCingOthers": "CC",
    "totalHeal": "heal",
}


def rename_stats(df):
    for stat in statList:
        df["base_{}".format(statList[stat])] = df[stat]
    return df


def get_stats_per_min(df, statList):
    for stat in statList:
        df["perMin_{}".format(statList[stat])] = df[stat] / (df["gameDuration"] / 60)
    return df


def get_stats_percent_per_team_and_match(df, statList):
    for stat in statList:
        df["pctTeam_{}".format(statList[stat])] = df[stat] / df.groupby(
            ["gameId", "teamId"]
        )[stat].transform("sum")
    for stat in statList:
        df["pctMatch_{}".format(statList[stat])] = df[stat] / df.groupby("gameId")[
            stat
        ].transform("sum")
    df = df.fillna(0)
    return df


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


def get_percent_champ_build(df):
    dfGrouped = df.groupby(["champion", "build"]).agg({"win": "count"})
    dfPct = dfGrouped.groupby(level=0).apply(lambda x: x / float(x.sum()))
    dfPct = dfPct.rename(columns={"win": "build_percent"}).reset_index()
    dfMerge = df.merge(dfPct, on=["champion", "build"], how="left")
    return dfMerge


# %%
finalDF = (
    flatStats.pipe(get_champion_names)
    .pipe(get_build_breakdown)
    .pipe(rename_stats)
    .pipe(get_stats_per_min, statList)
    .pipe(get_stats_percent_per_team_and_match, statList)
    .pipe(get_percent_champ_build)
)

# %%
dropCol = [
    "playerCount",
    "gameDuration",
    "gameCreation",
    "gameVersion",
    "accountId",
    "summonerName",
    "championId",
    "kills",
    "deaths",
    "assists",
    "goldEarned",
    "champLevel",
    "item0",
    "item1",
    "item2",
    "item3",
    "item4",
    "item5",
    "item6",
    "totalMinionsKilled",
    "totalDamageTaken",
    "magicalDamageTaken",
    "physicalDamageTaken",
    "trueDamageTaken",
    "totalDamageDealtToChampions",
    "magicDamageDealtToChampions",
    "physicalDamageDealtToChampions",
    "trueDamageDealtToChampions",
    "timeCCingOthers",
    "totalHeal",
]

prunedDF = finalDF.drop(dropCol, axis=1)
prunedDF.to_csv("data/processed/baseFeature_df.csv", index=False)

# %%
