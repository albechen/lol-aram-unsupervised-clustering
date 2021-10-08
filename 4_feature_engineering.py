#%%
import pandas as pd

flatStats = pd.read_csv("data/interim/filtered_rawFlatStats.csv")

# %%
statList = {
    "kills": "kill",
    "deaths": "death",
    "assists": "assist",
    "goldEarned": "gold",
    "totalMinionsKilled": "cs",
    "totalDamageTaken": "ttlTank",
    "magicDamageTaken": "apTank",
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


# %%
minPctBuild = 0.1
finalDF = (
    flatStats.pipe(rename_stats)
    .pipe(get_stats_per_min, statList)
    .pipe(get_stats_percent_per_team_and_match, statList)
)

# %%
dropCol = [
    "gameDuration",
    "gameCreation",
    "gameVersion",
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
    "magicDamageTaken",
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
