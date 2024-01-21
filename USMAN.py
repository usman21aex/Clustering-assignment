# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 23:17:27 2024

@author: 786
"""

# Importing Modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setting environment to ignore future warnings
import warnings
warnings.simplefilter('ignore')

# Loading dataset
data = pd.read_csv("/content/data.csv", skiprows=4).iloc[:, :-1]

# FIlling missing values
data.fillna(0, inplace=True)

fields = ['Population, total',
          'Renewable electricity output (% of total electricity output)',
          'Electricity production from oil sources (% of total)',
          'Electricity production from nuclear sources (% of total)',
          'Electricity production from natural gas sources (% of total)',
          'Electricity production from hydroelectric sources (% of total)',
          'Electricity production from coal sources (% of total)']

df = data[data["Indicator Name"].isin(fields)]

# Dropping records that are not for Countries
not_country = ['Euro area', 'IDA blend', 'Middle East & North Africa (excluding high income)',
               'Africa Western and Central', 'Middle East & North Africa (IDA & IBRD countries)',
               'Central Europe and the Baltics', 'Middle East & North Africa (IDA & IBRD countries)',
               'Middle East & North Africa', "Arab World",
               'Europe & Central Asia (excluding high income)', 'Africa Eastern and Southern', 'Low income',
               'Latin America & Caribbean (excluding high income)',
               'Europe & Central Asia (IDA & IBRD countries)',
               'Heavily indebted poor countries (HIPC)', 'European Union',
               'Latin America & the Caribbean (IDA & IBRD countries)',
               'Latin America & Caribbean', 'Pre-demographic dividend',
               'Fragile and conflict affected situations',
               'Least developed countries: UN classification',
               'Sub-Saharan Africa (excluding high income)', 'Sub-Saharan Africa',
               'Sub-Saharan Africa (IDA & IBRD countries)', 'IDA only',
               'Europe & Central Asia', 'IDA total',
               'Post-demographic dividend', 'High income', 'OECD members',
               'South Asia (IDA & IBRD)', 'South Asia',
               'East Asia & Pacific (IDA & IBRD countries)',
               'East Asia & Pacific (excluding high income)', 'East Asia & Pacific',
               'Late-demographic dividend', 'Upper middle income',
               'Lower middle income', 'Early-demographic dividend', 'IBRD only',
               'Middle income', 'Low & middle income', 'IDA & IBRD total', 'World']

df = df[~df["Country Name"].isin(not_country)]

# Dropping Unnecessary Issues
df.drop(["Country Code", "Indicator Code"], axis=1, inplace=True)

# Analysis
avg_population = df[df["Indicator Name"] == "Population, total"].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)
avg_oil_elec = df[df["Indicator Name"] == 'Electricity production from oil sources (% of total)'].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)
avg_nuc_elec = df[df["Indicator Name"] == 'Electricity production from nuclear sources (% of total)'].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)
avg_gas_elec = df[df["Indicator Name"] == 'Electricity production from natural gas sources (% of total)'].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)
avg_hyd_elec = df[df["Indicator Name"] == 'Electricity production from hydroelectric sources (% of total)'].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)
avg_coal_elec = df[df["Indicator Name"] == 'Electricity production from coal sources (% of total)'].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)
avg_renew_elec = df[df["Indicator Name"] == 'Renewable electricity output (% of total electricity output)'].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)

# Plotting Average Population
ax = avg_population.sort_values(ascending=False)[:15].plot(kind="bar", figsize=(15, 6))

# Adding labels in millions to each bar
for i, v in enumerate(avg_population.sort_values(ascending=False)[:15]):
    ax.text(i, v + 0.5, f"{round(v / 1e6, 2)}M", ha='center', va='bottom')

plt.title("Average Population of Top 15 Countries", fontsize=24)
plt.show()

# Plotting Renewable Electricity Output
ax = avg_renew_elec.sort_values(ascending=False)[:15].plot(kind="bar", figsize=(15, 6))

# Adding labels to each bar
for i, v in enumerate(avg_renew_elec.sort_values(ascending=False)[:15]):
    ax.text(i, v + 0.1, f"{round(v, 2)}%", ha='center', va='bottom')
plt.title("Renewable Electricity Output (% of Total) - Top 15 Countries", fontsize=24)
plt.show()

# Plotting Electricity Production from Coal
ax = avg_coal_elec.sort_values(ascending=False)[:15].plot(kind="bar", figsize=(15, 6))

# Adding labels to each bar
for i, v in enumerate(avg_coal_elec.sort_values(ascending=False)[:15]):
    ax.text(i, v + 0.1, f"{round(v, 2)}%", ha='center', va='bottom')

plt.title("Electricity Production from Coal (% of Total) - Top 15 Countries", fontsize=24)
plt.show()

# Plotting Electricity Production from 
ax = avg_gas_elec.sort_values(ascending=False)[:15].plot(kind="bar", figsize=(15, 6))

# Adding labels to each bar
for i, v in enumerate(avg_gas_elec.sort_values(ascending=False)[:15]):
    ax.text(i, v + 0.1, f"{round(v, 2)}%", ha='center', va='bottom')

plt.title("Electricity Production from Gas (% of Total) - Top 15 Countries", fontsize=24)
plt.show()

# Plotting Electricity Production from Hydroelectric Sources
ax = avg_hyd_elec.sort_values(ascending=False)[:15].plot(kind="bar", figsize=(15, 6))

# Adding labels to each bar
for i, v in enumerate(avg_hyd_elec.sort_values(ascending=False)[:15]):
    ax.text(i, v + 0.1, f"{round(v, 2)}%", ha='center', va='bottom')

plt.title("Electricity Production from Hydroelectric Sources (% of Total) - Top 15 Countries", fontsize=24)
plt.show()

# Plotting Electricity Production from Nuclear Sources
ax = avg_nuc_elec.sort_values(ascending=False)[:15].plot(kind="bar", figsize=(15, 6))

# Adding labels to each bar
for i, v in enumerate(avg_nuc_elec.sort_values(ascending=False)[:15]):
    ax.text(i, v + 0.1, f"{round(v, 2)}%", ha='center', va='bottom')

plt.title("Electricity Production from Nuclear Sources (% of Total) - Top 15 Countries", fontsize=24)
plt.show()

# Plotting Electricity Production from Oil Sources
ax = avg_oil_elec.sort_values(ascending=False)[:15].plot(kind="bar", figsize=(15, 6))

# Adding labels to each bar
for i, v in enumerate(avg_oil_elec.sort_values(ascending=False)[:15]):
    ax.text(i, v + 0.1, f"{round(v, 2)}%", ha='center', va='bottom')

plt.title("Electricity Production from Oil Sources (% of Total) - Top 15 Countries", fontsize=24)
plt.show()

# Creating a DataFrame for Energy Sources
energy_df = pd.DataFrame(avg_coal_elec)
energy_df.columns = ["Coal"]
energy_df["Renewable"] = avg_renew_elec
energy_df["Gas"] = avg_gas_elec
energy_df["Hydro"] = avg_hyd_elec
energy_df["Nuclear"] = avg_nuc_elec
energy_df["Oil"] = avg_oil_elec

# Sorting the DataFrame by each column in descending order
energy_df = energy_df.sort_values(by=["Coal", "Renewable", "Gas", "Hydro", "Nuclear", "Oil"], ascending=False)

# Plotting Top 15 Countries for Each Energy Source on a single graph
energy_df.head(15).plot(kind="bar", figsize=(18, 8))
plt.show()

# Plotting Change in Population of Top 5 Populated Countries Over Years
df[df["Indicator Name"] == "Population, total"].drop("Indicator Name", axis=1).set_index("Country Name").sort_values("2021").tail(5).T.plot(kind="line", figsize=(18, 6))
plt.title("Change in Population of Top 5 Populated Countries", fontsize=24)
plt.show()

# Plotting Change in Population of Top 10 Countries After Top 5 Populated Ones
df[df["Indicator Name"] == "Population, total"].drop("Indicator Name", axis=1).set_index("Country Name").sort_values("2021").tail(15).head(10).T.plot(kind="line", figsize=(18, 6))
plt.title("Change in Population of Top 10 Countries after Top 5 Populated ones", fontsize=24)
plt.show()

# Clustering
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

df_new = df.copy()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_new["Indicator Name"] = encoder.fit_transform(df_new["Indicator Name"])

X = df[df["Indicator Name"] == "Renewable electricity output (% of total electricity output)"].set_index("Country Name").drop("Indicator Name", axis=1)

data, labels = make_blobs(n_samples=50, centers=3, random_state=0)

kmeans = AgglomerativeClustering (n_clusters=5)
X['cluster'] = kmeans.fit_predict(X)

# Create linkage matrix
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(26, 7))
dendrogram(Z, labels=X.index)
plt.show()

# Plotting Cluster Distribution

X['cluster'].value_counts().plot(kind="bar", figsize=(10, 6))
plt.title("Cluster Distribution", fontsize=18)
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# Displaying Top Countries in Each Cluster
ind_1 = X[X.cluster == 1].index[:10]
ind_2 = X[X.cluster == 2].index[:10]
ind_3 = X[X.cluster == 3].index[:10]
ind_4 = X[X.cluster == 4].index[:10]

result_df = pd.DataFrame({"Cluster 1": ind_1, "Cluster 2": ind_2, "Cluster 3": ind_3, "Cluster 4": ind_4})
result_df.head(10)
