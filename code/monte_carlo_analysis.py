#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import urllib
import xlrd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
import io
import matplotlib.colors


# In[2]:


# Load data as dataframes

# RCP8.5 Baseline
RCP85_baseline = pd.read_excel('RCP85_baseline.xls',0)

# RCP4.5 Baseline
RCP45_baseline = pd.read_excel('RCP45_baseline.xls',0)

# Gas breakdown
gases = pd.read_excel('gas_breakdown.xls',0)

# Food group breakdown
foods = pd.read_excel('foodgroup_breakdown.xls',0)

# Figure S1 Data
S1 = pd.read_excel('figure_s1.xls',0)

# Figure 3
mitigation = pd.read_excel('mitigation_toplot.xls',0)
mitigation_updated = pd.read_excel('mitigation_updated.xls',0)

# Figure S2 data
per_capita = pd.read_excel('regional_emissions_figure.xls',0)
total_annual = pd.read_excel('regional_emissions_figure.xls',1)

# Figure S3 data
emiss_diff = pd.read_excel('dietary_change_emissions_diff.xls',0)


# In[3]:


# Load in data
grains = pd.read_excel('bootstrap_analysis.xls',0)
rice = pd.read_excel('bootstrap_analysis.xls',1)
fruit = pd.read_excel('bootstrap_analysis.xls',2)
veg = pd.read_excel('bootstrap_analysis.xls',3)
rm = pd.read_excel('bootstrap_analysis.xls',4)
nrm = pd.read_excel('bootstrap_analysis.xls',5)
seafood = pd.read_excel('bootstrap_analysis.xls',6)
dairy = pd.read_excel('bootstrap_analysis.xls',7)
eggs = pd.read_excel('bootstrap_analysis.xls',8)
oils = pd.read_excel('bootstrap_analysis.xls',9)
bev = pd.read_excel('bootstrap_analysis.xls',10)
other = pd.read_excel('bootstrap_analysis.xls',11)
calc = pd.read_excel('bootstrap_analysis.xls',12)

food_groups = [grains, rice, fruit, veg, rm, nrm, seafood, dairy, eggs, oils, bev, other]


# In[9]:


import random

group_names = ['Grains','Rice','Fruit','Vegetables','Ruminant Meat',
               'Non-Ruminant Meat','Seafood','Dairy','Eggs','Oils','Beverages','Other']

grouped_counts = calc.groupby(['Food Group']).count()

N = 1000

CO2_total = np.zeros(N)
CH4_total = np.zeros(N)
N2O_total = np.zeros(N)

for j in range(N):

    gas_breakdown = []

    for i in range(len(food_groups)):

        group = food_groups[i]

        # Select random row from food group LCAs
        ind = random.randrange(len(group))
        gas_prct = np.array(group.iloc[ind])
        num_tile = int(grouped_counts[grouped_counts.index == group_names[i]]['Food Item'])
        prct_input = np.tile(gas_prct,(num_tile,1))

        # Stack all percent breakdown data on top of each other
        if i == 0:
            gas_breakdown = prct_input

        if i > 0:
            gas_breakdown = np.vstack((gas_breakdown, prct_input))

    ### Calculate across to get total emissions
    food_item_GHGs = calc['Greenhouse Gas Emissions (kg CO2e100/kg) IPCC 2013 GWP100s with cc feedbacks (CH4 34 and N2O 298)']

    # Emissions in CO2e/kg food
    CO2_CO2e = food_item_GHGs*gas_breakdown[:,0]/100
    CH4_CO2e = food_item_GHGs*gas_breakdown[:,1]/100
    N20_CO2e = food_item_GHGs*gas_breakdown[:,2]/100

    # Emissions in kg gas/kg food
    CO2_kg = CO2_CO2e
    CH4_kg = CH4_CO2e/34
    N20_kg = N20_CO2e/298

    # Emissions in MMt/yr
    food_supply = calc['Global food supply 2010 (kg/capita/yr) - old methodology and population from FAO database**']

    CO2_mmt = CO2_kg*food_supply*6770979000/1000000000
    CH4_mmt = CH4_kg*food_supply*6770979000/1000000000
    N2O_mmt = N20_kg*food_supply*6770979000/1000000000

    # Calculate total annual emissions
    CO2_total[j] = sum(CO2_mmt)
    CH4_total[j] = sum(CH4_mmt)
    N2O_total[j] = sum(N2O_mmt)


# In[10]:


# Calculate 5th and 95th percentiles in each gas
CO2_5th = round(np.percentile(CO2_total,5),0)
CO2_95th = round(np.percentile(CO2_total,95),0)

CH4_5th = round(np.percentile(CH4_total,5),0)
CH4_95th = round(np.percentile(CH4_total,95),0)

N2O_5th = round(np.percentile(N2O_total,5),0)
N2O_95th = round(np.percentile(N2O_total,95),0)

print('CO2: ' + str(CO2_5th) + ' to ' + str(CO2_95th) + ' MMt.')
print('CH4: ' + str(CH4_5th) + ' to ' + str(CH4_95th) + ' MMt.')
print('N2O: ' + str(N2O_5th) + ' to ' + str(N2O_95th) + ' MMt.')


# In[ ]:




