#!/usr/bin/env python
# coding: utf-8

# In[15]:


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


# In[3]:


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


# # Figure 1

# In[5]:


RCP85_baseline = RCP85_baseline[RCP85_baseline.Title != 'Default']
titles = ['Constant_Population', 'SSP1', 'SSP5', 'SSP2', 'SSP4', 'SSP3']
titles.reverse()


# In[8]:


plt.rcParams.update({'font.size': 7})

cm = 1/2.54

fig1, ax = plt.subplots(dpi = 300, facecolor='w', edgecolor='k')
fig1.set_size_inches(8.8*cm,8.8*cm)
titles = ['Constant_Population', 'SSP1', 'SSP5', 'SSP2', 'SSP4', 'SSP3']
titles.reverse()
colors = ['#2a02ad','#3c8fe8','#0aab2d','#fac507','#f54f02','#940c0c']
colors.reverse()
labels = ['Constant 2020 Global Population', 'Sustainability (SSP1)', 'Fossil-Fueled Development (SSP5)', 'Middle of the Road (SSP2)', 'Inequality (SSP4)', 'Regional Rivalry (SSP3)']
labels.reverse()

for key, grp in RCP85_baseline.groupby(['Title'], sort=False):
    
    ind = titles.index(grp['Title'].iloc[0])
    
    if key == 'SSP4':
        line = grp.plot(ax = ax, kind = 'line', x = 'Year', y = 'mean', label = labels[ind], c = colors[ind], linewidth = 1.5, linestyle = 'dotted', zorder = 10)
    else:
        line = grp.plot(ax = ax, kind = 'line', x = 'Year', y = 'mean', label = labels[ind], c = colors[ind], linewidth = 1.5)
    
    plt.fill_between(grp['Year'], grp['ci95_lo'], grp['ci95_hi'], color = colors[ind], alpha = 0.15)
    
# Manual legend
line1 = Line2D([0], [0], color=colors[0], alpha = 0.8, linestyle = 'solid', linewidth = 1.5)
line2 = Line2D([0], [0], color=colors[1], alpha = 0.8, linestyle = 'dotted', linewidth = 1.5)
line3 = Line2D([0], [0], color=colors[2], alpha = 0.8, linestyle = 'solid', linewidth = 1.5)
line4 = Line2D([0], [0], color=colors[3], alpha = 0.8, linestyle = 'solid', linewidth = 1.5)
line5 = Line2D([0], [0], color=colors[4], alpha = 0.8, linestyle = 'solid', linewidth = 1.5)
line6 = Line2D([0], [0], color=colors[5], alpha = 0.8, linestyle = 'solid', linewidth = 1.5)

legend = ax.legend([line1, line2,line3,line4,line5,line6],labels, loc = 'upper left', framealpha=1, title = 'Population Projections', fontsize = 5)

plt.setp(legend.get_title(),fontsize= 5, fontweight = 'bold')

plt.xlim(2020,2100)
plt.ylim(0,1)
plt.xlabel('Year')
plt.ylabel('Global Mean Surface Air Temperature (C) Response to \n Greenhouse Gas Emissions from Future Food Consumption')
plt.show()


# In[9]:


RCP45_baseline = RCP45_baseline[RCP45_baseline.Title != 'Default']

titles = ['Constant_Population', 'SSP1', 'SSP5', 'SSP2', 'SSP4', 'SSP3']
titles.reverse()
colors = ['#2a02ad','#3c8fe8','#0aab2d','#fac507','#f54f02','#940c0c']
colors.reverse()
labels = ['Constant 2020 Global Population', 'Sustainability (SSP1)', 'Fossil-Fueled Development (SSP5)', 'Middle of the Road (SSP2)', 'Inequality (SSP4)', 'Regional Rivalry (SSP3)']
labels.reverse()

plt.rcParams.update({'font.size': 7})

cm = 1/2.54

fig2, ax = plt.subplots(dpi = 300, facecolor='w', edgecolor='k')
fig2.set_size_inches(8.8*cm,8.8*cm)

for key, grp in RCP45_baseline.groupby(['Title']):
    
    ind = titles.index(grp['Title'].iloc[0])
    
    if key == 'SSP4':
        line = grp.plot(ax = ax, kind = 'line', x = 'Year', y = 'mean', label = labels[ind], c = colors[ind], linewidth = 1.5, linestyle = 'dotted')
    else:
        line = grp.plot(ax = ax, kind = 'line', x = 'Year', y = 'mean', label = labels[ind], c = colors[ind], linewidth = 1.5)
    
    plt.fill_between(grp['Year'], grp['ci95_lo'], grp['ci95_hi'], color = colors[ind], alpha = 0.25)
    

# Manual legend
line1 = Line2D([0], [0], color=colors[0], alpha = 0.8, linestyle = 'solid', linewidth = 1.5)
line2 = Line2D([0], [0], color=colors[1], alpha = 0.8, linestyle = 'dotted', linewidth = 1.5)
line3 = Line2D([0], [0], color=colors[2], alpha = 0.8, linestyle = 'solid', linewidth = 1.5)
line4 = Line2D([0], [0], color=colors[3], alpha = 0.8, linestyle = 'solid', linewidth = 1.5)
line5 = Line2D([0], [0], color=colors[4], alpha = 0.8, linestyle = 'solid', linewidth = 1.5)
line6 = Line2D([0], [0], color=colors[5], alpha = 0.8, linestyle = 'solid', linewidth = 1.5)

legend = ax.legend([line1, line2,line3,line4,line5,line6],labels, loc = 'upper left', framealpha=1, title = 'Population Projections', fontsize = 5)

plt.setp(legend.get_title(),fontsize= 5, fontweight = 'bold')

plt.xlim(2020,2100)
plt.ylim(0,1.2)
plt.xlabel('Year')
plt.ylabel('Global Mean Surface Air Temperature (C) Response to \n Greenhouse Gas Emissions from Future Food Consumption')

plt.show()


# # Figure 2

# In[10]:


import string
plt.rcParams.update({'font.size': 7})

grouped_gases = gases.groupby(['Title'])
CO2 = grouped_gases.get_group('CO2')
CH4 = grouped_gases.get_group('CH4')
N2O = grouped_gases.get_group('N2O')

colors = ['#372674','#810a85','#c91e7f']

labels = ['N2O','CO2','CH4']

fig3, ax = plt.subplots(dpi = 300, facecolor='w', edgecolor='k')
fig3.set_size_inches(8.8*cm,8.8*cm)

ax.stackplot(CO2['Year'], CH4['mean'],CO2['mean'],N2O['mean'], colors = colors, labels = labels)
ax.set_xlim(2020,2100)
ax.set_ylim(0,1)
ax.set_xlabel('Year')
plt.ylabel('Global Mean Surface Air Temperature (C) Response to \n Greenhouse Gas Emissions from Future Food Consumption')


# Manual legend
line1 = Line2D([0], [0], color=colors[2], linestyle = 'solid', linewidth = 4)
line2 = Line2D([0], [0], color=colors[1], linestyle = 'solid', linewidth = 4)
line3 = Line2D([0], [0], color=colors[0],  linestyle = 'solid', linewidth = 4)

legend = ax.legend([line1, line2,line3],labels, loc = 'upper left', framealpha=1, fontsize = 5, borderpad=1)

plt.show()


# # Figure 3

# In[12]:


labels = ['Ruminant Meat','Rice','Dairy','Non-Ruminant Meat','Vegetables','Grains','Seafood','Oils','Other','Beverages','Eggs','Fruit']
grouped_foods = foods.groupby(['Year'])
foods_2030 = grouped_foods.get_group(2030)
foods_2050 = grouped_foods.get_group(2050)
foods_2100 = grouped_foods.get_group(2100)

years = [2030,2050,2100]

foods_2030['Percent'] = 100*foods_2030['mean']/sum(foods_2030['mean'])
foods_2050['Percent'] = 100*foods_2050['mean']/sum(foods_2050['mean'])
foods_2100['Percent'] = 100*foods_2100['mean']/sum(foods_2100['mean'])

# Get them all in descending order
foods_2030 = foods_2030.sort_values(by=['mean'],ascending=False)

foods_2050 = foods_2050.set_index('Title')
foods_2050 = foods_2050.reindex(index=foods_2030['Title'])
foods_2050 = foods_2050.reset_index()

foods_2100 = foods_2100.set_index('Title')
foods_2100 = foods_2100.reindex(index=foods_2030['Title'])
foods_2100 = foods_2100.reset_index()


# In[14]:


# Bar chart
colors = ['#212529','#636b73','#ADB5BD']

N = 12
ind = np.arange(N) 
width = 0.25

fig, axs = plt.subplots(1,1, figsize=(7,4.5), facecolor='w', edgecolor='k', dpi = 300)
plt.rcParams.update({'font.size': 7})
  
prctl_2030 = foods_2030['Percent']
bar1 = plt.bar(ind, prctl_2030, width, color = colors[0])
  
prctl_2050 = foods_2050['Percent']
bar2 = plt.bar(ind+width, prctl_2050, width, color=colors[1])
  
prctl_2100 = foods_2100['Percent']
bar3 = plt.bar(ind+width*2, prctl_2100, width, color = colors[2])
  
plt.xlabel("Food Group")
plt.ylabel('Relative Contribution to Global Mean Surface Air Temperature (C) \n Response to Future Food Consumption Greenhouse Gas Emissions')
  
plt.xticks(ind+width,foods_2030.Title,rotation=45, ha='right')

# Put a legend to the right of the current axis
axs.legend((bar1, bar2, bar3), ('2030', '2050', '2100'),loc='center left', bbox_to_anchor=(0.13, 0.84))

# Add pie chart to upper righthand corner
left, bottom, width, height = [0.40, 0.35, 0.55, 0.55]
ax2 = fig.add_axes([left, bottom, width, height])

year = 2030

labels = ['Ruminant Meat','Rice','Dairy','Non-Ruminant Meat','Vegetables','Grains','Seafood','Oils','Other','Beverages','Eggs','Fruit']
colors2 = ['#044E14','#06741E','#089B28','#09AE2D','#54DF72','#65E280','#76E58E','#87E89C','#98EBAA','#AAEEB9','#BBF2C7','#CCF5D5']

warmings = np.squeeze(np.array([RM.loc[RM.Year == year]['mean'],Rice.loc[Rice.Year == year]['mean'],Dairy.loc[Dairy.Year == year]['mean'],NRM.loc[NRM.Year == year]['mean'],
    Vegetables.loc[Vegetables.Year == year]['mean'],Grains.loc[Grains.Year == year]['mean'],Seafood.loc[Seafood.Year == year]['mean'],Oils.loc[Oils.Year == year]['mean'],Other.loc[Other.Year == year]['mean'],
    Beverages.loc[Beverages.Year == year]['mean'],Eggs.loc[Eggs.Year == year]['mean'],Fruit.loc[Fruit.Year == year]['mean']]))

total_warming = sum(warmings)
    
sizes = warmings/total_warming
    
label_pct = []
    
for j in range(len(labels)):
    label_pct.append(labels[j]+ ', ' + str(int(np.round(sizes[j]*100))) + '%')

# Plot
ax2.pie(sizes, labels=label_pct, autopct=' ', pctdistance=0.64, startangle=180, counterclock = False, colors = colors2, wedgeprops = {'linewidth': 1, 'edgecolor':'w'}, textprops={'fontsize': 7})
    
#draw circle
circle = plt.Circle((0,0), 0.50, fc='white')
ax2.add_patch(circle)
    
# Add in middle of circle
ax2.text(-0.15, -0.05, '2030', fontsize = 7, fontweight = 'bold')


# ## Figure 4

# In[17]:


mitigation_no_old = mitigation[(mitigation['Title'] != 'All Three') & (mitigation['Title'] != 'Food Loss')]
mitigation_full = mitigation_no_old.append(mitigation_updated)
mitigation_full.Title.unique()


# In[20]:


fig4, ax = plt.subplots(dpi = 300, facecolor='w', edgecolor='k')
fig4.set_size_inches(8.8*cm,8.8*cm)

titles = ['Baseline','Retail','Consumer', 'Decarbonization', 'Diet', 'Production','All_Mitigation']
colors = ['#2a02ad','#3c8fe8','#0aab2d','#fac507','#f54f02','#940c0c','#5e5e5e']
colors.reverse()

labels = ['No Mitigation', 'Reduce Retail Food Waste by 50%','Reduce Consumer Food Waste by 50%','Decarbonization of Food Production', 'Global Conversion to Healthy Diet', 'Maximum Production Improvements',  'All Mitigation Methods']

for key, grp in mitigation_full.groupby(['Title'], sort = False):
    ind = titles.index(grp['Title'].iloc[0])
    line = grp.plot(ax = ax, kind = 'line', x = 'Year', y = 'mean', label = labels[ind], c = colors[ind], linewidth = 1.5)
    plt.fill_between(grp['Year'], grp['ci95_lo'], grp['ci95_hi'], color = colors[ind], alpha = 0.25)

width = 1.5
line0 = Line2D([0], [0], color=colors[0], alpha = 1, linestyle = 'solid', linewidth = width)
line1 = Line2D([0], [0], color=colors[1], alpha = 1, linestyle = 'solid', linewidth = width)
line2 = Line2D([0], [0], color=colors[2], alpha = 1, linestyle = 'solid', linewidth = width)
line3 = Line2D([0], [0], color=colors[3], alpha = 1, linestyle = 'solid', linewidth = width)
line4 = Line2D([0], [0], color=colors[4], alpha = 1, linestyle = 'solid', linewidth = width)
line5 = Line2D([0], [0], color=colors[5], alpha = 1, linestyle = 'solid', linewidth = width)
line6 = Line2D([0], [0], color=colors[6], alpha = 1, linestyle = 'solid', linewidth = width)

legend = ax.legend([line0,line1, line2,line3,line4,line5,line6],labels, loc = 'upper left', framealpha=1, title = 'Mitigation Strategies', fontsize = 4.8)

plt.setp(legend.get_title(),fontsize= 5, fontweight = 'bold')

legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((0, 0, 0, 0))

plt.xlim(2020,2100)
plt.ylim(0,1)
plt.xlabel('Year')
plt.ylabel('Global Mean Surface Air Temperature (C) Response to \n Greenhouse Gas Emissions from Future Food Consumption')
plt.show()


# ## ED Figure 2

# In[35]:


# Make this into a bar chart

S1_sorted = S1.sort_values(by=['Percent Emissions (CO2e20)'],ascending = False)
sizes_CO2e100 = S1_sorted['Percent Emissions (CO2e100)']
sizes_CO2e20 = S1_sorted['Percent Emissions (CO2e20)']

labels = S1_sorted['Food Groups']


# In[36]:


# Resubmission bar chart

colors = ['#212529','#ADB5BD']

N = 12
ind = np.arange(N) 
width = 0.25

fig, axs = plt.subplots(1,1, figsize=(7,4.5), facecolor='w', edgecolor='k', dpi = 300)
plt.rcParams.update({'font.size': 7})

bar1 = plt.bar(ind, 100*sizes_CO2e20, width, color = colors[0])

bar2 = plt.bar(ind+width, 100*sizes_CO2e100, width, color=colors[1])
  
plt.xlabel("Food Group")
plt.ylabel('Relative Contribution to Global Mean Surface Air Temperature (C) \n Response to Future Food Consumption Greenhouse Gas Emissions')

plt.xticks(ind+width,labels,rotation=45, ha='right')

# Put a legend to the right of the current axis
axs.legend((bar1, bar2), ('CO2e20', 'CO2e100'),loc='upper right')


# # ED Figures 3, 4, and 5

# In[25]:


# Make Taiwan same value as overall China
china_ind = per_capita.index[per_capita['Country']=='China'].tolist()

taiwan = {'Country':'Taiwan','CO2': float(per_capita.iloc[china_ind].CO2), 'CH4': float(per_capita.iloc[china_ind].CH4), 'N2O': float(per_capita.iloc[china_ind].N2O)}

per_capita = per_capita.append(taiwan, ignore_index = True)

china_ind = total_annual.index[total_annual['Country']=='China'].tolist()

taiwan = {'Country':'Taiwan','CO2': float(total_annual.iloc[china_ind].CO2), 'CH4': float(total_annual.iloc[china_ind].CH4), 'N2O': float(total_annual.iloc[china_ind].N2O)}

total_annual = total_annual.append(taiwan, ignore_index = True)

# Emissions difference
# Make Taiwan same value as overall China
china_ind = emiss_diff.index[emiss_diff['Country Name']=='China'].tolist()

taiwan = {'Country Name':'Taiwan','CO2 (kg/capita/yr)': float(emiss_diff.iloc[china_ind]['CO2 (kg/capita/yr)']), 'CH4 (kg/capita/yr)': float(emiss_diff.iloc[china_ind]['CH4 (kg/capita/yr)']),'N2O (kg/capita/yr)': float(emiss_diff.iloc[china_ind]['N2O (kg/capita/yr)']), 'CO2 (Tg/yr)': float(emiss_diff.iloc[china_ind]['CO2 (Tg/yr)']),'CH4 (Tg/yr)': float(emiss_diff.iloc[china_ind]['CH4 (Tg/yr)']), 'N2O (Tg/yr)': float(emiss_diff.iloc[china_ind]['N2O (Tg/yr)'])}

emiss_diff = emiss_diff.append(taiwan, ignore_index = True)


# In[26]:


import geopandas
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
same = list(set(per_capita.Country) & set(world.name))
not_same_mine = list(set(per_capita.Country).difference(world.name))
not_same_gp = list(set(world.name).difference(per_capita.Country))


# In[27]:


# Replace all names that you can in per_capita and total
to_replace = ['Central African Republic','Democratic Republic of the Congo','Iran (Islamic Republic of)',
                "Lao People's Democratic Republic",'Bolivia (Plurinational State of)','United Republic of Tanzania',
                'Republic of Moldova','Viet Nam','North Macedonia','Russian Federation','United Kingdom of Great Britain and Northern Ireland',
                'Republic of Korea',"Democratic People's Republic of Korea",'Syrian Arab Republic','Dominican Republic',
                'Solomon Islands','Venezuela (Bolivarian Republic of)','Bosnia and Herzegovina','Eswatini']
replace_with = [ 'Central African Rep.','Dem. Rep. Congo','Iran','Laos','Bolivia','Tanzania','Moldova','Vietnam',
                'Macedonia','Russia','United Kingdom','South Korea','North Korea','Syria','Dominican Rep.','Solomon Is.',
                'Venezuela','Bosnia and Herz.','eSwatini',]

per_capita = per_capita.replace(to_replace, replace_with)

total_annual = total_annual.replace(to_replace, replace_with)

emiss_diff = emiss_diff.replace(to_replace, replace_with)

per_capita = per_capita.rename(columns={'Country': 'name'})
total_annual = total_annual.rename(columns={'Country': 'name'})
emiss_diff = emiss_diff.rename(columns={'Country Name': 'name'})


# In[28]:


world_percapita = pd.merge(world,per_capita, how = 'outer', on = 'name')
world_percapita['CO2'] = world_percapita['CO2'].fillna(-5)
world_percapita['CH4'] = world_percapita['CH4'].fillna(-5)
world_percapita['N2O'] = world_percapita['N2O'].fillna(-5)

world_total = pd.merge(world,total_annual, how = 'outer', on = 'name')
world_total['CO2'] = world_total['CO2'].fillna(-5)
world_total['CH4'] = world_total['CH4'].fillna(-5)
world_total['N2O'] = world_total['N2O'].fillna(-5)

world_diff = pd.merge(world,emiss_diff, how = 'outer', on = 'name')
world_diff['CO2 (kg/capita/yr)'] = world_diff['CO2 (kg/capita/yr)'].fillna(-99999)
world_diff['CH4 (kg/capita/yr)'] = world_diff['CH4 (kg/capita/yr)'].fillna(-99999)
world_diff['N2O (kg/capita/yr)'] = world_diff['N2O (kg/capita/yr)'].fillna(-99999)

world_diff['CO2 (Tg/yr)'] = world_diff['CO2 (Tg/yr)'].fillna(-99999)
world_diff['CH4 (Tg/yr)'] = world_diff['CH4 (Tg/yr)'].fillna(-99999)
world_diff['N2O (Tg/yr)'] = world_diff['N2O (Tg/yr)'].fillna(-99999)


# In[29]:


# Create each color map
import matplotlib.colors

colors = ['#810a85','#372674','#c91e7f']

cmap_CO2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",colors[0]])
cmap_CH4 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",colors[1]])
cmap_N2O = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",colors[2]])

cmaps = [cmap_CO2, cmap_CH4, cmap_N2O]


# Per capita

# In[30]:


gases = ['CO2','CH4','N2O']

world_noant = world_percapita[(world_percapita.name!="Antarctica")]
world_data = world_noant[(world_noant.CO2>0)]
world_nodata = world_noant[(world_noant.CO2<0)]


# In[31]:


# Plot each gas per capita
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axs = plt.subplots(3,1, figsize=(6,8.5), facecolor='w', edgecolor='k', dpi = 300)
plt.rcParams.update({'font.size': 7})

for i in range(len(gases)):
    
    axis = axs[i]
    
    # Generate cmap for each gas
    cmap = cmaps[i]
    
    max_gas = np.max(world_data[gases[i]])
    
    if i < 2:
        dec = 0
    if i == 2:
        dec = 1
        
    step = round(max_gas/6,dec)
    bounds = [0, step, 2*step,3*step,4*step,5*step,6*step]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    nodata_norm = plt.Normalize(-8,0)
    
    # Plot countries with data
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    data = world_data.plot(column=gases[i], ax = axis, cmap = cmap, norm = norm, legend = True, cax = cax,
                          legend_kwds={'label': "Per Capita \n" + str(gases[i]) + " Emissions \n (kg/capita/yr)",'fmt': "{:.0f}"})
    
    # Plot no data countries
    world_nodata.plot(column = gases[i], ax = axis, cmap = 'Greys_r', norm = nodata_norm)
    
    axis.set_ylabel('Latitude')
    
    if i == 2:
        axis.set_xlabel('Longitude')
        
    world_noant.boundary.plot(ax = axis,facecolor="none", edgecolor="black", linewidth = 0.5)


# Total Annual

# In[33]:


# Select just EU countries, sum over them
world_noant = world_total[(world_total.name!="Antarctica")]
world_data = world_noant[(world_noant.CO2>0)]
world_nodata = world_noant[(world_noant.CO2<0)]

EUcountries = ['Austria','Belgium','Bulgaria', 'Croatia', 'Cyprus', 
               'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 
               'Germany','Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 
               'Lithuania', 'Luxembourg','Malta', 'Netherlands', 'Poland', 
               'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden']

EU_data = world_data[world_data['name'].isin(EUcountries)]
EU_CO2 = round(EU_data.CO2.sum(),5)
EU_CH4 = round(EU_data.CH4.sum(),5)
EU_N2O = round(EU_data.N2O.sum(),5)

# Set EU countries' values to that sum
world_data.loc[(world_data['name'].isin(EUcountries)), 'CO2'] = EU_CO2
world_data.loc[(world_data['name'].isin(EUcountries)), 'CH4'] = EU_CH4
world_data.loc[(world_data['name'].isin(EUcountries)), 'N2O'] = EU_N2O

EU_toplot = world_data[world_data['name'].isin(EUcountries)]


# In[34]:


from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axs = plt.subplots(3,1, figsize=(6,8.5), facecolor='w', edgecolor='k', dpi = 300)
plt.rcParams.update({'font.size': 7})

for i in range(len(gases)):
    
    axis = axs[i]
    
    # Generate cmap for each gas
    cmap = cmaps[i]
    
    if i < 2:
        dec = 0
    if i == 2:
        dec = 1
    
    max_gas = np.max(world_data[gases[i]])
    step = round(max_gas/15,dec)
    bounds = [0, step, 2*step,3*step,4*step,5*step,6*step,7*step, 8*step, 9*step, 10*step,
              11*step,12*step, 13*step, 14*step, 15*step]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    nodata_norm = plt.Normalize(-8,0)
    
    # Plot countries with data
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    data = world_data.plot(column=gases[i], ax = axis, cmap = cmap, norm = norm, legend = True, cax = cax,
                          legend_kwds={'label': "Total Annual \n" + str(gases[i]) + " Emissions \n (Tg/yr)",'fmt': "{:.0f}"})
    
    # Plot no data countries
    world_nodata.plot(column = gases[i], ax = axis, cmap = 'Greys_r', norm = nodata_norm)
    
    # Plot just EU countries, hatched
    EU_toplot.plot(column=gases[i], ax = axis, cmap = cmap, norm = norm, hatch = '////////') 

    axis.set_ylabel('Latitude')
    
    if i == 2:
        axis.set_xlabel('Longitude')
        
    world_noant.boundary.plot(ax = axis,facecolor="none", edgecolor="black", linewidth = 0.5)


# Emissions Difference per Country

# In[35]:


gases = ['CO2 (kg/capita/yr)','CH4 (kg/capita/yr)','N2O (kg/capita/yr)']

world_noant = world_diff[(world_diff.name!="Antarctica")]
world_data = world_noant[(world_noant['CO2 (kg/capita/yr)'] > -99999)]
world_nodata = world_noant[(world_noant['CO2 (kg/capita/yr)'] == -99999)]


# In[37]:


# Plot each gas per capita
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axs = plt.subplots(3,1, figsize=(6,8.5), facecolor='w', edgecolor='k', dpi = 300)
plt.rcParams.update({'font.size': 7})

for i in range(len(gases)):
    
    axis = axs[i]
    
    if i < 2:
        dec = 0
    if i == 2:
        dec = 1
    
    max_gas = np.max(world_data[gases[i]])
    min_gas = abs(np.min(world_data[gases[i]]))
    bigger = np.max([max_gas,min_gas])
        
    step = round(bigger/8,dec)
    bounds = [-9*step, -8*step, -7*step, -6*step,-5*step, -4*step, -3*step,-2*step, -1*step,
              0, step, 2*step,3*step,4*step,5*step,6*step,7*step, 8*step,9*step]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    nodata_norm = plt.Normalize(-150000,0)
    
    # Plot countries with data
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    data = world_data.plot(column=gases[i], ax = axis, cmap = 'RdBu_r', norm = norm, legend = True, cax = cax,
                          legend_kwds={'label': "Emissions Difference in \n" + str(gases[i]),'fmt': "{:.0f}"})
    
    # Plot no data countries
    world_nodata.plot(column = gases[i], ax = axis, cmap = 'Greys_r', norm = nodata_norm)
    
    axis.set_ylabel('Latitude')
    
    if i == 2:
        axis.set_xlabel('Longitude')
        
    world_noant.boundary.plot(ax = axis,facecolor="none", edgecolor="black", linewidth = 0.5)


# # Monte Carlo analysis

# In[192]:


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


# In[193]:


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

        #print(str(group_names[i]) + ' is done.')

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
    
    #print('Total CO2 = ' + str(CO2_total) + ', Total CH4 = '+ str(CH4_total) + ', Total N2O = '+ str(N2O_total))


# In[194]:


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

