#%%
import pandas as pd
from riotwatcher import LolWatcher
import pickle

api_key = "RGAPI-c959d8fd-5547-4247-a0cc-0514e34bb7db"
watcher = LolWatcher(api_key)
my_region = "na1"


def replace_pickle(path, newFile):
    with open(path, "wb") as f:
        pickle.dump(newFile, f)


#%%
def pull_champion_dict():
    latest = watcher.data_dragon.versions_for_region(my_region)["n"]["champion"]
    static_champ_list = watcher.data_dragon.champions(latest, False, "en_US")

    champ_dict = {}
    for key in static_champ_list["data"]:
        row = static_champ_list["data"][key]
        champ_dict[row["key"]] = row["id"]

    replace_pickle("data/interim/champion_dict.pkl", champ_dict)


def pull_item_dict():
    latest = watcher.data_dragon.versions_for_region(my_region)["n"]["item"]
    static_champ_list = watcher.data_dragon.items(latest, "en_US")

    nameList = []
    goldList = []
    idList = []
    for key in static_champ_list["data"]:
        row = static_champ_list["data"][key]
        idList.append(key)
        nameList.append(row["name"])
        goldList.append(row["gold"]["total"])

    itemDF = pd.DataFrame()
    itemDF["Name"] = nameList
    itemDF["ID"] = idList
    itemDF["Gold"] = goldList

    itemDF.to_csv("data/interim/item_df.csv", index=False)


# %%
pull_champion_dict()
pull_item_dict()
# %%
