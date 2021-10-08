#%%
import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime

#%%
def open_pickle(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


matchInfoList = [
    "gameId",
    "gameCreation",
    "gameDuration",
    "gameVersion",
]

matchPUUID = ["puuid"]

playerStatList = [
    "summonerName",
    "championId",
    "teamId",
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
    "magicDamageTaken",
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
    matchStats = []
    for playerStat, puuid in zip(
        match["info"]["participants"], match["metadata"]["participants"]
    ):
        playerRow = []

        for matchInfo in matchInfoList:
            playerRow.append(match["info"][matchInfo])

        playerRow.append(puuid)

        for playerStatInfo in playerStatList:
            playerRow.append(playerStat[playerStatInfo])

        matchStats.append(playerRow)

    return matchStats


# match = open_pickle("data/raw/NA1_3982842863.pkl")
# test = pull_raw_values_per_match(match)
# test

#%%
def iterate_matches_pull_raw_data():
    fullStats = []
    count = 0
    allMatchStats = []
    for subdir, dirs, files in os.walk("data/raw"):
        for file in files:
            match = open_pickle(os.path.join(subdir, file))
            matchStats = pull_raw_values_per_match(match)
            allMatchStats.extend(matchStats)

            count += 1
            if count % 200 == 0:
                print(count, ": ", datetime.now())

    fullColumns = matchInfoList + matchPUUID + playerStatList
    fullStats_df = pd.DataFrame(allMatchStats, columns=fullColumns)
    fullStats_dfdd = fullStats_df.drop_duplicates()

    fullStats_dfdd.to_csv("data/interim/rawFlatStats.csv", index=False)

    return fullStats_df


# %%
fullStats_df = iterate_matches_pull_raw_data()
# %%
