# Paper title and data availability
This repository contains codes for the paper entitled " Machine learning-based water quality prediction using octennial in-situ Daphnia magna biological early warning system data" authored by Heewon Jeong, Sanghyun Park, Byeongwook Choi, Chung Seok Yu, Ji Young Hong, Tae-Yong Jeong, Kyung Hwa Cho, Korea. The file name "BEWS data", "Flow rate data", "Water quality data", and "Weather data" contain raw data sets used in this study to build this model. 
Besides, The published article can be found on

# Introduction
_Daphnia magna_ biological early warning system (BEWS) has successfully detected sudden spills of contaminants into surface waters. However, it has exhibited limitations, including low alarm reproducibility and difficulties in biological interpretation. This study applied machine learning (ML) models to predict contamination alarms from Daphnia behavioral parameters. Six ML models were adopted for eight years of in-situ BEWS data for _Daphnia magna_. The dataset includes _Daphnia Magna_ behavior, chemical water quality, and hydrometeorological data

# Files in this Repository
∙ *_model.py: Files named “_model.py” were used to build a prediction model utilizing six machine learning algorithms. Applied machine learning algorithms were artificial neural network, convolution neural network, TabNet, random forest, light gradient boosting machine, and extreme gradient boosting. They can be identified by abbreviations (i.e., ANN, CNN, TabNet, RF, LGBM, and XGB) at the beginning of the file name. All the required detail processes, including optimizing hyperparameters, and training models, are included in the code files.
∙ *_data.xlsx: Files named "_data.xlsx" were the raw data sets applied in this study. The data were measured by Korea Ministry of Environment and Korea Meteorological Administration from 2012 to 2020 (8 years long) and are also available on the websites of these institutions. 

# Correspondance
If you feel any difficulties in executing these codes, please contact us through email at gua01114@gmail.com. 
Thank you
