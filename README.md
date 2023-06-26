# League of Legends ARAM Unsupervised Clustering to Recommend Champion

## Overview

In Riot's flagship game, League of Legends, there is a game mode called ARAM; an acronym for *"All Random - All Middle"*. This game mode features a single lane map in which two teams of five random champions duke it out in a simplified team fight focused game of League. Prior to the start of the game, the five random champions can be re-rolled to provide some level of control over champions used. Therefore, this research aims to:

  1. Cluster champions and build given match stattics across 20k matches pulled using Riot's API
  2. Analyze the best way to choose what champion to play by either:
     1. **Win Rate** -- picking champion based on overall win rate
     2. **Team Composition**** -- picking champions based on what niche your team is missing

**Language:** Python (riot-watcher, pandas, sklearn, seaborn, numpy)
**Skills:** Unsupervised Clustering, Cluster Visualization (PCA, TSNE), API Usage

### Notebooks

- [0: static_data_dragon_keys](https://github.com/albechen/lol-aram-recommendation-model/blob/main/0_static_data_dragon_keys.py)
Data pulls using Data Dragon API of champion and items in league of legends - constant across all games

- [1: pull_riotWatcher_games](https://github.com/albechen/lol-aram-recommendation-model/blob/main/1_pull_riotWatcher_games.py)
Created workflow to continuously pull new match histories of players by incrementing through each player in the newly pulled match. Rate limiting supported by riot-watcher but pulls about 2,800 games an hour. Matches stored in raw data as pickle files but can be changed to JSON

- [2: flatten_matches_to_df.](https://github.com/albechen/lol-aram-recommendation-model/blob/main/2_flatten_matches_to_df..py)
All matches in raw data folder aggregated with stats to be used. Flattened into single dataframe to be saved as csv file with each player's stat as a single row - such that each game produces 10 rows.

- [3: clean_columns_and_filter](https://github.com/albechen/lol-aram-recommendation-model/blob/main/3_clean_columns_and_filter.py)
Adding some simple stats and cleaning to filter out data. Games prior to season 11 are filtered out along with games that have a player that build items not regularly built on their champion.

- [4: feature_engineering](https://github.com/albechen/lol-aram-recommendation-model/blob/main/4_feature_engineering.py)
Main features engineered for clustering and modeling. All numerical stats are transformed to
  1. stat per min of game duration
  2. % of stat per team's stat
  3. % of stat per game's stat

- [5: pca_setup](https://github.com/albechen/lol-aram-recommendation-model/blob/main/5_pca_setup.py)
Engineered features are cleaned using
  1. log transformation on skewed columns
  2. normalizing and min max scaling
  3. PCA transformation with elbow method to determine number of components (10)

- [6: unsupervised_clustering](https://github.com/albechen/lol-aram-recommendation-modelblob/main/6_unsupervised_clustering.py)
Clustering of PCA transformed data was preformed to determine champion/build grouping and to create champ/build cluster probability dataset.
  1. Checked optimal clusters using Kmeans and Gaussian for data (each row being player's stats in single game)
  2. Preform clustering given optimal clusters on pca data
  3. Aggregated clustering data for each champion and build (for Kmeans, gather number of rows for each cluster and for Gaussian, gather average probability and +/- one std)
     - use this dataset to build recommendation system by joining and aggregate by teams
  4. Applied scaling and pca for aggregated champion and build data
  5. Checked optimal cluster using new aggregated data
  6. Preform clustering on aggregated data given optimal clusters using TSNE
  7. Graph TSNE0 vs. TSNE1 with coloring of different clusters to visually observe separation

- [7: recommendation_model](https://github.com/albechen/lol-aram-recommendation-model/blob/main/7_recommendation_model.py)
Aim is to aggregate data on a team level to represent each team's cluster composition to create dataset to model team's victories.
  1. Join prior dataset to every game and sum probability of each cluster across each team in each game
  2. Adjust for skew, scale, and preform PCA to min 95% variance accounted
  3. Compare some models and do light tuning on hyper parameters
  4. Test using holdout data from 4_feature_engineering which wasn't used when building champion clustering for building any of the later datasets
  5. Built naïve model with multiplying team's champion/build win probability given this dataset and compare against model

## Results

### Modeling Results

| Model | Test Accuracy |
| - | - |
| Naïve (Highest Win Rate) | 60.3% |
| Log Regression (Team Composition) | 51.5% |

- CONCLUSION: recommendation system using champion cluster identity to aggregate team composition success is less successful than picking champions with generally higher win rates

### Champions and Build Clustered

The optimal clustering was completed with a gaussian mixture using davies_bouldin_score and calinski_harabasz_score to determine the optimal clusters. After preforming clustering on the individual games per each champion and aggregating the results, further clustering was preformed per champion and build to result in the following clusters:

![alt text](/data/results/images/championCluster_tsne.jpg " championCluster")

Each cluster below is makes sense for those that play league, but also now analytically shows the similarity between each champion and their corresponding build's similarity and category.

| Cluster | Champion (Build) |
| - | - |
|0|Ivern (support), Janna (ap), Janna (support), Karma (support), Lulu (ap), Lulu (support), Nami (ap), Nami (support), Seraphine (support), Sona (ap), Sona (support), Soraka (ap), Soraka (tank), Yuumi (ap), Yuumi (support)|
|1|Aatrox (bruiser), Camille (bruiser), Darius (bruiser), Darius (tank), Fiora (bruiser), Garen (bruiser), Garen (tank), Hecarim (bruiser), Hecarim (tank), Illaoi (bruiser), Illaoi (tank), Irelia (ad), Irelia (bruiser), Kayle (ad), Kayn (bruiser), Kled (bruiser), Kled (tank), KogMaw (ad), Nasus (bruiser), Nasus (tank), Olaf (bruiser), Olaf (tank), RekSai (bruiser), Renekton (bruiser), Renekton (tank), Rengar (bruiser), Sett (bruiser), Sett (tank), Sion (tank), Trundle (bruiser), Trundle (tank), Twitch (ap), Udyr (bruiser), Viego (bruiser), Volibear (bruiser), XinZhao (bruiser), Yorick (bruiser), Yorick (tank)|
|2|Aphelios (ad), Ashe (ad), Draven (ad), Ezreal (ad), Gnar (ad), Gnar (bruiser), Gnar (tank), Hecarim (ad), JarvanIV (ad), Jayce (ad), Jhin (ad), Kalista (ad), Nocturne (ad), Nocturne (bruiser), Sion (ad), Sivir (ad), Varus (ad), Yasuo (ad)|
|3|Ahri (ap), Akali (ap), Amumu (ap), Bard (ap), Blitzcrank (ap), Chogath (ap), Ekko (ap), Evelynn (ap), Fiddlesticks (ap), Galio (ap), Gragas (ap), Gwen (ap), Katarina (ap), KogMaw (ap), Lillia (ap), Maokai (ap), Mordekaiser (ap), Mordekaiser (tank), Nidalee (ap), Nunu (ap), Singed (ap), Swain (ap), Sylas (ap), Velkoz (ap), Vladimir (ap), Zac (ap), Zoe (ap)|
|4|Caitlyn (ad), Graves (ad), JarvanIV (bruiser), JarvanIV (tank), Jax (ad), Jax (bruiser), Kayn (ad), Kindred (ad), LeeSin (ad), LeeSin (bruiser), MonkeyKing (ad), MonkeyKing (bruiser), Pantheon (bruiser), Riven (bruiser), Samira (ad), Talon (bruiser), Tryndamere (ad), Udyr (ad), Urgot (bruiser), Urgot (tank), Vi (ad), Vi (bruiser), Viego (ad), Xayah (ad), XinZhao (ad), Yorick (ad)|
|5|Alistar (tank), Amumu (tank), Blitzcrank (tank), Braum (tank), Chogath (tank), Galio (tank), Gragas (tank), Leona (tank), Malphite (tank), Maokai (tank), Nautilus (tank), Nunu (tank), Ornn (tank), Rakan (ap), Rakan (support), Rakan (tank), Rammus (tank), Rell (tank), Singed (tank), Soraka (support), TahmKench (tank), Taric (tank), Thresh (tank), Zac (tank)|
|6|Annie (ap), Diana (ap), Elise (ap), Fizz (ap), Karthus (ap), Kassadin (ap), Kennen (ap), Leblanc (ap), Lissandra (ap), Lux (ap), Malphite (ap), Malzahar (ap), Neeko (ap), Rumble (ap), Shaco (ap), Syndra (ap), Taliyah (ap), Teemo (ap), Veigar (ap), Vex (ap), Zyra (ap)|
|7|Bard (ad), DrMundo (tank), Katarina (bruiser), Kayle (ap), Poppy (tank), Sejuani (tank), Shaco (ad), Shen (tank), Skarner (tank), Thresh (ad), TwistedFate (ad), Udyr (tank), Volibear (tank), Warwick (ad), Warwick (bruiser), Warwick (tank)|
|8|Anivia (ap), AurelionSol (ap), Azir (ap), Brand (ap), Cassiopeia (ap), Corki (ad), Heimerdinger (ap), Ivern (ap), Kaisa (ap), Karma (ap), MissFortune (ap), Morgana (ap), Nasus (ap), Orianna (ap), Ryze (ap), Seraphine (ap), Shyvana (ap), TwistedFate (ap), Viktor (ap), Xerath (ap), Ziggs (ap), Zilean (ap)|
|9|Akshan (ad), Fiora (ad), Gangplank (ad), Garen (ad), Jinx (ad), Kaisa (ad), Khazix (ad), Lucian (ad), MasterYi (ad), MissFortune (ad), Pantheon (ad), Pyke (ad), Qiyana (ad), Quinn (ad), RekSai (ad), Rengar (ad), Senna (ad), Talon (ad), Tristana (ad), Twitch (ad), Vayne (ad), Yone (ad), Zed (ad)|
