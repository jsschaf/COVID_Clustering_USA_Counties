# COVID_Clustering_USA_Counties
Analysis of USA Counties by confirmed COVID-19 cases. Performs k-means clustering before and after dimensionally reduction by PCA.

Original data from: https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv

time_series_converter.py converts data into County-level time-series csv.

time_series_PCA.py performs k-means clustering before and after PCA and analyzed how dimensionality reduction has changed the clustering of counties.
