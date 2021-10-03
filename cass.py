#%%
import random
import numpy as np
from cassiopeia.core.match import Participant
from dateutil.tz.tz import gettz
from numpy.lib.function_base import append
from sortedcontainers import SortedList
import arrow

import cassiopeia as cass
from cassiopeia.core import Summoner, MatchHistory, Match
from cassiopeia import Queue, Patch, Items, Item

from cassiopeia.data import Season, Queue
from collections import Counter

import datetime

cass.set_riot_api_key("RGAPI-cb27e808-3a7f-4f3c-93ad-7df1bc4e89e7")
import pandas as pd

#%%
def filter_match_history(summoner):
    now = datetime.datetime.now()
    oneYearPast = now - datetime.timedelta(days=365)

    match_history = MatchHistory(
        summoner=summoner,
        queues={Queue.aram},
        begin_time=arrow.get(oneYearPast),
        end_time=arrow.get(now),
    )
    return match_history


#%%
region = "NA"
patch = Patch.latest(region=region)
summoner = Summoner(name="rectumhoward", region=region)
matches = filter_match_history(summoner)
match = Match(id=matches[0].id, region=region)
p = match.participants[0]


#%%
def pull_items_stats_per_player(p):
    itemDF = pd.read_csv("itemsType.csv")
    itemDict = itemDF.set_index("Name").to_dict()
    playerItemType = {"ad": 0, "bruiser": 0, "ap": 0, "tank": 0, "support": 0}

    for item in p.stats.items:
        try:
            playerItemType[itemDict["Type"][item.name]] += itemDict["Gold"][item.name]
        except:
            pass

    itemList = list(playerItemType.values())

    return itemList


itemColumns = ["item_ad", "item_bruiser", "item_ap", "item_tank", "item_support"]

#%%
endAttrList = [
    "win",
    "kills",
    "deaths",
    "assists",
    "gold_earned",
    "level",
    "total_minions_killed",
    "combat_player_score",
    "objective_player_score",
    "total_player_score",
    # TANK STATS
    "total_damage_taken",
    "magical_damage_taken",
    "physical_damage_taken",
    "true_damage_taken",
    # DMG STATS
    "total_damage_dealt_to_champions",
    "magic_damage_dealt_to_champions",
    "physical_damage_dealt_to_champions",
    "true_damage_dealt_to_champions",
    # OTHER COMBAT STATS
    "time_CCing_others",
    "total_time_crowd_control_dealt",
    "total_heal",
]

endColumns = [
    "id",
    "patch",
    "duration",
    "side",
    "name",
    "id",
    "level",
    "champion",
    "soloDiv",
    "soloTier",
    "flexDiv",
    "flexTier",
]


def pull_end_game_player_stats(endAttrList, match, p):
    statList = [
        match.id,
        match.patch,
        match.duration.seconds,
        p.side,
        p.summoner.name,
        p.summoner.id,
        p.summoner.level,
        p.champion.name,
    ]

    try:
        statList.append(p.summoner.ranks[Queue.ranked_solo_fives].division)
    except:
        statList.append(None)

    try:
        statList.append(p.summoner.ranks[Queue.ranked_solo_fives].tier)
    except:
        statList.append(None)

    try:
        statList.append(p.summoner.ranks[Queue.ranked_flex_fives].division)
    except:
        statList.append(None)

    try:
        statList.append(p.summoner.ranks[Queue.ranked_flex_fives].tier)
    except:
        statList.append(None)

    for statAttr in endAttrList:
        statList.append(getattr(p.stats, statAttr))
    return statList


#%%
def pull_delta_player_stats(p, intervalSec, durationSec, deltaAttrList):

    maxInterval = int(durationSec - (durationSec % intervalSec))
    pastList = [0, 0, 0, 0, 0]
    interval = intervalSec
    valueList = []

    while interval <= min(maxInterval, intervalSec * 5):
        pState = p.cumulative_timeline[datetime.timedelta(seconds=interval)]

        currentList = []
        for attr in deltaAttrList:
            currentList.append(getattr(pState, attr))

        deltaList = [x1 - x2 for (x1, x2) in zip(currentList, pastList)]
        valueList.extend(deltaList)

        pastList = currentList
        interval += intervalSec

    valueList = (valueList + [None] * (5 * 5))[: (5 * 5)]

    return valueList


deltaAttrList = ["kills", "deaths", "assists", "gold_earned", "experience"]
durationSec = match.duration.seconds
intervalSec = 3 * 60
deltaListEx = pull_delta_player_stats(p, intervalSec, durationSec, deltaAttrList)

#%%
flatStats = None
for p in match.participants:
    intervalSec = 3 * 60
    durationSec = match.duration.seconds
    endStatsList = pull_end_game_player_stats(endAttrList, match, p)
    itemList = pull_items_stats_per_player(p)
    deltaStatsList = pull_delta_player_stats(p, intervalSec, durationSec, deltaAttrList)
    statList = np.array(endStatsList + itemList + deltaStatsList)
    if flatStats is not None:
        flatStats = np.vstack([flatStats, statList])
    else:
        flatStats = statList
#%%
deltaColumn = []
deltaLen = int(len(deltaStatsList) / len(deltaAttrList))
colInterval = 0

for x in range(deltaLen):
    colInterval += intervalSec
    for attr in deltaAttrList:
        deltaColumn.append(str(int(colInterval / 60)) + "_" + attr)

columnNames = endColumns + endAttrList + itemColumns + deltaColumn
df = pd.DataFrame(flatStats, columns=columnNames)

#%%


def collect_matches():
    initial_summoner_name = "TheWaterFell"
    region = "NA"

    summoner = Summoner(name=initial_summoner_name, region=region)

    unpulled_summoner_ids = SortedList([summoner.id])
    pulled_summoner_ids = SortedList()

    unpulled_match_ids = SortedList()
    pulled_match_ids = SortedList()
    count = 0

    while unpulled_summoner_ids:
        # Get a random summoner from our list of unpulled summoners and pull their match history
        new_summoner_id = random.choice(unpulled_summoner_ids)
        new_summoner = Summoner(id=new_summoner_id, region=region)
        matches = filter_match_history(new_summoner)
        unpulled_match_ids.update([match.id for match in matches])
        unpulled_summoner_ids.remove(new_summoner_id)
        pulled_summoner_ids.add(new_summoner_id)

        while unpulled_match_ids or count <= 100:
            # Get a random match from our list of matches
            new_match_id = random.choice(unpulled_match_ids)
            match = Match(id=new_match_id, region=region)
            for participant in match.participants:
                if (
                    participant.summoner.id not in pulled_summoner_ids
                    and participant.summoner.id not in unpulled_summoner_ids
                ):
                    unpulled_summoner_ids.add(participant.summoner.id)

                flatStats = None
                for p in match.participants:
                    intervalSec = 3 * 60
                    durationSec = match.duration.seconds
                    endStatsList = pull_end_game_player_stats(endAttrList, match, p)
                    itemList = pull_items_stats_per_player(p)
                    # deltaStatsList = pull_delta_player_stats(
                    #     p, intervalSec, durationSec, deltaAttrList
                    # )
                    statList = np.array(endStatsList + itemList)
                    if flatStats is not None:
                        flatStats = np.vstack([flatStats, statList])
                    else:
                        flatStats = statList
            # The above lines will trigger the match to load its data by iterating over all the participants.
            # If you have a database in your datapipeline, the match will automatically be stored in it.
            unpulled_match_ids.remove(new_match_id)
            pulled_match_ids.add(new_match_id)
            count += 1

        # deltaColumn = []
        # deltaLen = int(len(deltaStatsList) / len(deltaAttrList))
        # colInterval = 0

        # for x in range(deltaLen):
        #     colInterval += intervalSec
        #     for attr in deltaAttrList:
        #         deltaColumn.append(str(int(colInterval / 60)) + "_" + attr)

        columnNames = endColumns + endAttrList + itemColumns  # + deltaColumn
        df = pd.DataFrame(flatStats, columns=columnNames)

    return (
        df,
        unpulled_summoner_ids,
        pulled_summoner_ids,
        unpulled_match_ids,
        pulled_match_ids,
    )


(
    df,
    unpulled_summoner_ids,
    pulled_summoner_ids,
    unpulled_match_ids,
    pulled_match_ids,
) = collect_matches()


# %%
