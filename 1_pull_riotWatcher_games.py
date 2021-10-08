#%%
from riotwatcher import LolWatcher
from sortedcontainers import SortedList
import random
import pandas as pd
import pickle
from datetime import datetime

# https://developer.riotgames.com/#
api_key = "RGAPI-4834d45b-49b0-4d36-af1a-81a65e71bde6"
watcher = LolWatcher(api_key)
my_region = "americas"

# puuid = watcher.summoner.by_name("na1", 'TheWaterFell')['puuid']
# matches = watcher.match_v5.matchlist_by_puuid(
#                 my_region, puuid, queue=450, type='normal'
#             )
# matches
# match = watcher.match_v5.by_id(my_region, matches[0])
# %%
def open_pickle(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


def replace_pickle(path, newFile):
    with open(path, "wb") as f:
        pickle.dump(newFile, f)


# %%
def collect_matches(totalMatches, initial_summoner_name=None):
    my_region = "americas"
    if initial_summoner_name == None:
        unpulled_summoner_ids = open_pickle("data/interim/unpulled_summoner_ids.pkl")
        pulled_summoner_ids = open_pickle("data/interim/pulled_summoner_ids.pkl")
        unpulled_match_ids = open_pickle("data/interim/unpulled_match_ids.pkl")
        pulled_match_ids = open_pickle("data/interim/pulled_match_ids.pkl")
    else:
        summoner = watcher.summoner.by_name("na1", initial_summoner_name)
        unpulled_summoner_ids = SortedList([summoner["puuid"]])
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
            matches = watcher.match_v5.matchlist_by_puuid(
                my_region, new_summoner_id, queue=450, type="normal"
            )
            unpulled_match_ids.update(matches)
            unpulled_summoner_ids.remove(new_summoner_id)
            pulled_summoner_ids.add(new_summoner_id)

            while unpulled_match_ids:
                new_match_id = random.choice(unpulled_match_ids)
                match = watcher.match_v5.by_id(my_region, new_match_id)
                for puuid in match["metadata"]["participants"]:
                    if (
                        puuid not in pulled_summoner_ids
                        and puuid not in unpulled_summoner_ids
                    ):
                        unpulled_summoner_ids.add(puuid)

                replace_pickle("data/raw/{}.pkl".format(new_match_id), match)
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
collect_matches(totalMatches=8000, initial_summoner_name="TheWaterFell")
# %%
import time

collect_matches(totalMatches=4, initial_summoner_name=None)

# %%
