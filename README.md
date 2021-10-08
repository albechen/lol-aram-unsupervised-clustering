# League of Legends ARAM Champion Suggestion

**Unsupervised clustering of champion profile in ARAM to build recommendation model given re-roll options in for highest win rate**

**Language:** Python (riot-watcher, pandas, sklearn, seaborn, numpy)

In Riot's flagship game, League of Legends, there is a game mode called ARAM; an acronym for *"All Random - All Middle"*. This game mode features a single lane map in which two teams of five random champions duke it out in a simplified team fight focused game of League. Prior to the start of the game, the five random champions can be re-rolled to provide some level of control over champions used. Therefore, this research aims to 1) cluster champions and build given match statistics of each player and 2) create a recommendation model given re-rolled champion options and current team composition.

- [0: static_data_dragon_keys](https://github.com/albechen/lol_aram_character_suggestion/blob/main/0_static_data_dragon_keys.py) <br/>
Data pulls using Data Dragon API of champion and items in league of legends - constant across all games

- [1: pull_riotWatcher_games](https://github.com/albechen/lol_aram_character_suggestion/blob/main/1_pull_riotWatcher_games.py) <br/>
Created workflow to continuously pull new match histories of players by incrementing through each player in the newly pulled match. Rate limiting supported by riot-watcher but pulls about 2,800 games an hour. Matches stored in raw data as pickle files but can be changed to JSON

- [2: flatten_matches_to_df.](https://github.com/albechen/lol_aram_character_suggestion/blob/main/2_flatten_matches_to_df..py) <br/>
All matches in raw data folder aggregated with stats to be used. Flattened into single dataframe to be saved as csv file with each player's stat as a single row - such that each game produces 10 rows.

- [3: clean_columns_and_filter](https://github.com/albechen/lol_aram_character_suggestion/blob/main/3_clean_columns_and_filter.py) <br/>
Adding some simple stats and cleaning to filter out data. Games prior to season 11 are filtered out along with games that have a player that build items not regularly built on their champion.

- [4: feature_engineering](https://github.com/albechen/lol_aram_character_suggestion/blob/main/4_feature_engineering.py) <br/>
Main features engineered for clustering and modeling. All numerical stats are transformed to 1) stat per min of game duration, 2) % of stat per team's stat, and 3) % of stat per game's stat.

- [5: pca_setup](https://github.com/albechen/lol_aram_character_suggestion/blob/main/5_pca_setup.py) <br/>
Engineered features are cleaned using 1) log transformation on skewed columns, 2) normalizing and min max scaling, 3) PCA transformation with min 95% variance accounted for

- [6: unsupervised_clustering](https://github.com/albechen/lol_aram_character_suggestion/blob/main/6_unsupervised_clustering.py) <br/>
Clustering of PCA transformed data was preformed to determine champion/build grouping and to create champ/build cluster probability dataset.
  1. Checked optimal clusters using Kmeans and Gaussian for data (each row being player's stats in single game)
  2. Preform clustering given optimal clusters on pca data
  3. Aggregated clustering data for each champion and build (for Kmeans, gather number of rows for each cluster and for Gaussian, gather average probability and +/- one std)
     - use this dataset to build recommendation system by joining and aggregate by teams
  4. Applied scaling and pca for aggregated champion and build data
  5. Checked optimal cluster using new aggregated data
  6. Preform clustering on aggregated data given optimal clusters
  7. Graph PC0 vs. PC1 with coloring of different clusters to visually observe separation

- [7: recommendation_model](https://github.com/albechen/lol_aram_character_suggestion/blob/main/7_recommendation_model.py) <br/>
Aim is to aggregate data on a team level to represent each team's cluster composition to create dataset to model team's victories.
  1. Join prior dataset to every game and sum probability of each cluster across each team in each game
  2. Adjust for skew, scale, and preform PCA to min 95% variance accounted
  3. Compare some models and do light tuning on hyper parameters
  4. Test using holdout data from 4_feature_engineering which wasn't used when building champion clustering for building any of the later datasets
  5. Built naïve model with multiplying team's champion/build win probability given this dataset and compare against model

## Results
### Modeling Results:
- Prediction system was 51.7% accuracy using simple log regression model <br/>
- Naïve model predicted 56.9% accuracy
- CONCLUSION: recommendation system using champion cluster identity to aggregate team composition success is less successful than picking champions with generally higher win rates

### Champions and Build Clustered:
![alt text](/data/results/images/championCluster_gaussian.jpg " championCluster")

| Cluster | Champion (Build)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0       | DrMundo (tank), Gwen (ap), Katarina (bruiser), Kayle (ap), MissFortune (ap), Ornn (tank), Sejuani (tank), Shen (tank), Skarner (tank), Thresh (ad), TwistedFate (ad), Volibear (tank), Warwick (ad), Warwick (bruiser), Warwick (tank)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 1       | Akali (ap), Amumu (ap), Ekko (ap), Evelynn (ap), Fiddlesticks (ap), Fizz (ap), Galio (ap), Karma (ap), Katarina (ap), KogMaw (ap), Lulu (ap), Malphite (ap), Maokai (ap), Morgana (ap), Nasus (ap), Nidalee (ap), Seraphine (ap), Shyvana (ap), Singed (ap), Velkoz (ap), Zilean (ap), Zoe (ap)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2       | Aatrox (bruiser), Ashe (ad), Camille (bruiser), Darius (bruiser), Darius (tank), Fiora (ad), Fiora (bruiser), Gangplank (ad), Garen (ad), Garen (bruiser), Garen (tank), Gnar (ad), Gnar (bruiser), Gnar (tank), Hecarim (ad), Illaoi (bruiser), Illaoi (tank), JarvanIV (bruiser), Jax (ad), Jax (bruiser), Kaisa (ad), Kayn (bruiser), Kindred (ad), LeeSin (ad), LeeSin (bruiser), MasterYi (ad), MonkeyKing (ad), MonkeyKing (bruiser), Nocturne (ad), Olaf (bruiser), Olaf (tank), Pantheon (bruiser), RekSai (bruiser), Renekton (bruiser), Rengar (bruiser), Riven (bruiser), Senna (ad), Sett (bruiser), Sett (tank), Sion (ad), Talon (bruiser), Twitch (ad), Twitch (ap), Udyr (bruiser), Urgot (bruiser), Urgot (tank), Vayne (ad), Vi (bruiser), Viego (ad), Viego (bruiser), XinZhao (ad), XinZhao (bruiser), Yone (ad) |
| 3       | Bard (ad), Ezreal (ad), Hecarim (bruiser), Hecarim (tank), Irelia (ad), Irelia (bruiser), JarvanIV (tank), Kayle (ad), Kled (bruiser), Kled (tank), KogMaw (ad), Nasus (bruiser), Nasus (tank), Nocturne (bruiser), Poppy (tank), Renekton (tank), Shaco (ad), Sion (tank), Trundle (bruiser), Trundle (tank), Udyr (tank), Volibear (bruiser), Yorick (ad), Yorick (bruiser), Yorick (tank)                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 4       | Ahri (ap), Alistar (tank), Amumu (tank), Bard (ap), Blitzcrank (ap), Blitzcrank (tank), Braum (tank), Chogath (ap), Chogath (tank), Corki (ad), Galio (tank), Gragas (ap), Gragas (tank), Ivern (ap), Ivern (support), Janna (ap), Janna (support), Kaisa (ap), Karma (support), Leona (tank), Lillia (ap), Lulu (support), Malphite (tank), Maokai (tank), Mordekaiser (ap), Mordekaiser (tank), Nami (ap), Nami (support), Nautilus (tank), Nunu (ap), Nunu (tank), Rakan (ap), Rakan (support), Rakan (tank), Rammus (tank), Rell (tank), Seraphine (support), Singed (tank), Sona (ap), Sona (support), Soraka (ap), Soraka (support), Soraka (tank), Swain (ap), Sylas (ap), TahmKench (tank), Taric (tank), Thresh (tank), Vladimir (ap), Yuumi (ap), Yuumi (support), Zac (ap), Zac (tank)                                    |
| 5       | Anivia (ap), Annie (ap), AurelionSol (ap), Azir (ap), Brand (ap), Cassiopeia (ap), Diana (ap), Elise (ap), Heimerdinger (ap), Karthus (ap), Kassadin (ap), Kennen (ap), Leblanc (ap), Lissandra (ap), Lux (ap), Malzahar (ap), Neeko (ap), Orianna (ap), Rumble (ap), Ryze (ap), Shaco (ap), Syndra (ap), Taliyah (ap), Teemo (ap), TwistedFate (ap), Veigar (ap), Vex (ap), Viktor (ap), Xerath (ap), Ziggs (ap), Zyra (ap)                                                                                                                                                                                                                                                                                                                                                                                                         |
| 6       | Akshan (ad), Aphelios (ad), Caitlyn (ad), Draven (ad), Graves (ad), JarvanIV (ad), Jayce (ad), Jhin (ad), Jinx (ad), Kalista (ad), Kayn (ad), Khazix (ad), Lucian (ad), MissFortune (ad), Pantheon (ad), Pyke (ad), Qiyana (ad), Quinn (ad), RekSai (ad), Rengar (ad), Samira (ad), Sivir (ad), Talon (ad), Tristana (ad), Tryndamere (ad), Udyr (ad), Varus (ad), Vi (ad), Xayah (ad), Yasuo (ad), Zed (ad)                                                                                                                                                                                                                                                                                                                                                                                                                         |
