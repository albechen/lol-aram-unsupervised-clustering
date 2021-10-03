#%%
from numpy.lib.function_base import append
from riotwatcher import LolWatcher
from sortedcontainers import SortedList
import random
import pandas as pd
import pickle
from datetime import datetime

# https://developer.riotgames.com/#
api_key = "RGAPI-c959d8fd-5547-4247-a0cc-0514e34bb7db"
watcher = LolWatcher(api_key)
my_region = "na1"
# %%
def filter_by_gameMode(int_gamemode, my_region, accountId):
    my_matches = watcher.match.matchlist_by_account(my_region, accountId)
    matchList = []
    for match in my_matches["matches"]:
        if match["queue"] == int_gamemode:
            matchList.append(match["gameId"])
    return matchList


def open_pickle(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


def replace_pickle(path, newFile):
    with open(path, "wb") as f:
        pickle.dump(newFile, f)


# %%
def collect_matches(totalMatches, initial_summoner_name=None):
    my_region = "na1"
    if initial_summoner_name == None:
        unpulled_summoner_ids = open_pickle("data/interim/unpulled_summoner_ids.pkl")
        pulled_summoner_ids = open_pickle("data/interim/pulled_summoner_ids.pkl")
        unpulled_match_ids = open_pickle("data/interim/unpulled_match_ids.pkl")
        pulled_match_ids = open_pickle("data/interim/pulled_match_ids.pkl")
    else:
        summoner = watcher.summoner.by_name(my_region, initial_summoner_name)
        unpulled_summoner_ids = SortedList([summoner["accountId"]])
        pulled_summoner_ids = SortedList()
        unpulled_match_ids = SortedList()
        pulled_match_ids = SortedList()

    count = 0
    print(count, ": ", datetime.now())
    while unpulled_summoner_ids:
        if count >= totalMatches:
            break
        try:
            new_summoner_id = random.choice(unpulled_summoner_ids)
            matches = filter_by_gameMode(450, my_region, new_summoner_id)
            unpulled_match_ids.update(matches)
            unpulled_summoner_ids.remove(new_summoner_id)
            pulled_summoner_ids.add(new_summoner_id)

            while unpulled_match_ids:
                new_match_id = random.choice(unpulled_match_ids)
                match = watcher.match.by_id(my_region, new_match_id)
                for player in match["participantIdentities"]:
                    if (
                        player["player"]["accountId"] not in pulled_summoner_ids
                        and player["player"]["accountId"] not in unpulled_summoner_ids
                    ):
                        unpulled_summoner_ids.add(player["player"]["accountId"])

                createdDate = match["gameCreation"]
                replace_pickle(
                    "data/raw/{}_{}_{}.pkl".format(
                        new_summoner_id, createdDate, new_match_id
                    ),
                    match,
                )
                unpulled_match_ids.remove(new_match_id)
                pulled_match_ids.add(new_match_id)
                count += 1
                if count >= totalMatches:
                    break
                if count % 200 == 0:
                    print(count, ": ", datetime.now())
        except Exception as e:
            print(e)

    replace_pickle("data/interim/unpulled_summoner_ids.pkl", unpulled_summoner_ids)
    replace_pickle("data/interim/pulled_summoner_ids.pkl", pulled_summoner_ids)
    replace_pickle("data/interim/unpulled_match_ids.pkl", unpulled_match_ids)
    replace_pickle("data/interim/pulled_match_ids.pkl", pulled_match_ids)


#%%
collect_matches(totalMatches=5000, initial_summoner_name="TheWaterFell")
# %%
import time

time.sleep(60 * 3)
collect_matches(totalMatches=5000, initial_summoner_name=None)

# %%
