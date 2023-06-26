# League of Legends ARAM Champion Suggestion

Unsupervised clustering of champion profile in ARAM to build recommendation model given re-roll options in for highest win rate

**Language:** Python (riot-watcher, pandas, sklearn, seaborn, numpy)

## Overview

In Riot's flagship game, League of Legends, there is a game mode called ARAM; an acronym for *"All Random - All Middle"*. This game mode features a single lane map in which two teams of five random champions duke it out in a simplified team fight focused game of League. Prior to the start of the game, the five random champions can be re-rolled to provide some level of control over champions used. Therefore, this research aims to

  1. cluster champions and build given match statistics of each player
  2. create a recommendation model given re-rolled champion options and current team composition.

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
  6. Preform clustering on aggregated data given optimal clusters
  7. Graph PC0 vs. PC1 with coloring of different clusters to visually observe separation

- [7: recommendation_model](https://github.com/albechen/lol-aram-recommendation-model/blob/main/7_recommendation_model.py)
Aim is to aggregate data on a team level to represent each team's cluster composition to create dataset to model team's victories.
  1. Join prior dataset to every game and sum probability of each cluster across each team in each game
  2. Adjust for skew, scale, and preform PCA to min 95% variance accounted
  3. Compare some models and do light tuning on hyper parameters
  4. Test using holdout data from 4_feature_engineering which wasn't used when building champion clustering for building any of the later datasets
  5. Built naïve model with multiplying team's champion/build win probability given this dataset and compare against model

## Results

### Modeling Results

- Prediction system was 51.7% accuracy using simple log regression model
- Naïve model predicted 56.9% accuracy
- CONCLUSION: recommendation system using champion cluster identity to aggregate team composition success is less successful than picking champions with generally higher win rates

### Champions and Build Clustered

The optimal clustering was completed with a gaussian mixture using davies_bouldin_score and calinski_harabasz_score to determine the optimal clusters. After preforming clustering on the individual games per each champion and aggregating the results, further clustering was preformed per champion and build to result in the following clusters:

![alt text](/data/results/images/championCluster_gaussian.jpg " championCluster")

Each cluster below is makes sense for those that play league, but also now analytically shows the similarity between each champion and their corresponding build's similarity and category.

| Cluster | Champion (Build) |
| - | - |
| 0 | DrMundo (tank), Katarina (bruiser), Sejuani (tank), Shen (tank), Skarner (tank), Trundle (tank), Udyr (tank), Volibear (tank), Warwick (ad), Warwick (bruiser), Warwick (tank) |
| 1 | Ezreal (ad), Fiora (ad), Garen (bruiser), Gnar (bruiser), Gnar (tank), Hecarim (ad), JarvanIV (bruiser), Jax (ad), Jax (bruiser), Kindred (ad), LeeSin (bruiser), MonkeyKing (bruiser), Nocturne (bruiser), Pantheon (bruiser), RekSai (bruiser), Riven (bruiser), Sion (ad), Talon (bruiser), Tryndamere (ad), Udyr (ad), Urgot (bruiser), Urgot (tank), Vi (bruiser), Viego (ad), XinZhao (ad), Yone (ad), Yorick (ad) |
| 2 | Anivia (ap), Annie (ap), AurelionSol (ap), Azir (ap), Brand (ap), Diana (ap), Elise (ap), Evelynn (ap), Fizz (ap), Heimerdinger (ap), Kaisa (ap), Karma (ap), Karthus (ap), Kassadin (ap), Kennen (ap), Leblanc (ap), Lissandra (ap), Lux (ap), Malphite (ap), Malzahar (ap), Neeko (ap), Orianna (ap), Rumble (ap), Ryze (ap), Shaco (ap), Shyvana (ap), Syndra (ap), Taliyah (ap), Teemo (ap), TwistedFate (ap), Veigar (ap), Vex (ap), Viktor (ap), Xerath (ap), Ziggs (ap), Zyra (ap) |
| 3 | Bard (ap), Chogath (ap), Gragas (ap), Ivern (support), Janna (ap), Janna (support), Karma (support), Lulu (support), Malphite (tank), Maokai (ap), Mordekaiser (tank), Nami (ap), Nami (support), Nunu (ap), Rakan (ap), Seraphine (support), Singed (ap), Singed (tank), Sona (ap), Sona (support), Soraka (ap), Soraka (tank), Swain (ap), Sylas (ap), Yuumi (ap) |
|4 |Bard (ad), Gwen (ap), Kayle (ap), Shaco (ad), Thresh (ad), TwistedFate (ad) |
|5 |Aatrox (bruiser), Camille (bruiser), Darius (bruiser), Darius (tank), Fiora (bruiser), Garen (tank), Hecarim (bruiser), Hecarim (tank), Illaoi (bruiser), Illaoi (tank), Irelia (ad), Irelia (bruiser), JarvanIV (tank), Kayle (ad), Kayn (bruiser), Kled (bruiser), Kled (tank), KogMaw (ad), Nasus (bruiser), Nasus (tank), Olaf (bruiser), Olaf (tank), Poppy (tank), Renekton (bruiser), Renekton (tank), Rengar (bruiser), Sett (bruiser), Sett (tank), Sion (tank), Trundle (bruiser), Twitch (ap), Udyr (bruiser), Viego (bruiser), Volibear (bruiser), XinZhao (bruiser), Yorick (bruiser), Yorick (tank) |
|6 |Ahri (ap), Akali (ap), Amumu (ap), Blitzcrank (ap), Cassiopeia (ap), Corki (ad), Ekko (ap), Fiddlesticks (ap), Galio (ap), Ivern (ap), Katarina (ap), KogMaw (ap), Lillia (ap), Lulu (ap), MissFortune (ap), Mordekaiser (ap), Morgana (ap), Nasus (ap), Nidalee (ap), Seraphine (ap), Velkoz (ap), Vladimir (ap), Zilean (ap), Zoe (ap) |
|7 |Akshan (ad), Aphelios (ad), Ashe (ad), Caitlyn (ad), Draven (ad), Gangplank (ad), Garen (ad), Gnar (ad), Graves (ad), JarvanIV (ad), Jayce (ad), Jhin (ad), Jinx (ad), Kaisa (ad), Kalista (ad), Kayn (ad), Khazix (ad), LeeSin (ad), Lucian (ad), MasterYi (ad), MissFortune (ad), MonkeyKing (ad), Nocturne (ad), Pantheon (ad), Pyke (ad), Qiyana (ad), Quinn (ad), RekSai (ad), Rengar (ad), Samira (ad), Senna (ad), Sivir (ad), Talon (ad), Tristana (ad), Twitch (ad), Varus (ad), Vayne (ad), Vi (ad), Xayah (ad), Yasuo (ad), Zed (ad) |
|8 |Alistar (tank), Amumu (tank), Blitzcrank (tank), Braum (tank), Chogath (tank), Galio (tank), Gragas (tank), Leona (tank), Maokai (tank), Nautilus (tank), Nunu (tank), Ornn (tank), Rakan (support), Rakan (tank), Rammus (tank), Rell (tank), Soraka (support), TahmKench (tank), Taric (tank), Thresh (tank), Yuumi (support), Zac (ap), Zac (tank) |
