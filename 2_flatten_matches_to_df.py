#%%
import pickle
import os
import numpy as np
import pandas as pd

#%%
def open_pickle(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


# match = open_pickle(
#     "data/raw/_kc9cIxy8t3P_7whBECwdIiUxkeesNufJcSrGXaQwNPdkhk_1620180835169_3894004753.pkl"
# )

#%%

matchInfoList = [
    "gameId",
    "gameCreation",
    "gameDuration",
    "gameVersion",
]

playerInfoList = [
    "accountId",
    "summonerName",
]

playerMatchList = [
    "teamId",
    "championId",
]

playerStatList = [
    "win",
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
    # TANK STATS
    "totalDamageTaken",
    "magicalDamageTaken",
    "physicalDamageTaken",
    "trueDamageTaken",
    # DMG STATS
    "totalDamageDealtToChampions",
    "magicDamageDealtToChampions",
    "physicalDamageDealtToChampions",
    "trueDamageDealtToChampions",
    # OTHER COMBAT STATS
    "timeCCingOthers",
    "totalHeal",
]


def pull_raw_values_per_match(match):

    matchStats = None

    for playerStat, playerIdentity in zip(
        match["participants"], match["participantIdentities"]
    ):
        playerRow = []

        for matchInfo in matchInfoList:
            playerRow.append(match[matchInfo])

        for playerInfo in playerInfoList:
            playerRow.append(playerIdentity["player"][playerInfo])

        for playerMatchInfo in playerMatchList:
            playerRow.append(playerStat[playerMatchInfo])

        for playerStatInfo in playerStatList:
            playerRow.append(playerStat["stats"][playerStatInfo])

        playerArray = np.array(playerRow)

        if matchStats is not None:
            matchStats = np.vstack([matchStats, playerArray])
        else:
            matchStats = playerArray

    return matchStats


def iterate_matches_pull_raw_data():
    fullStats = None
    for subdir, dirs, files in os.walk("data/raw"):
        for file in files:
            match = open_pickle(os.path.join(subdir, file))
            matchStats = pull_raw_values_per_match(match)

            if fullStats is not None:
                fullStats = np.vstack([fullStats, matchStats])
            else:
                fullStats = matchStats
    fullColumns = matchInfoList + playerInfoList + playerMatchList + playerStatList
    fullStats_df = pd.DataFrame(fullStats, columns=fullColumns)
    fullStats_dfdd = fullStats_df.drop_duplicates()

    fullStats_dfdd.to_csv("data/interim/rawFlatStats.csv", index=False)

    return fullStats_df


# %%
fullStats_df = iterate_matches_pull_raw_data()
# %%
