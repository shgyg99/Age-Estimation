import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from functions import dataset_to_csv
### 🟡 Plot histogram for age"""
dataset = dataset_to_csv('E:\\My Drive\\deepcatalist\\utkcropped\\')
fig = sns.displot(data=dataset['age'], kde=True, color='red', facecolor='#3F8F9F')
fig.savefig('E:\\My Drive\\deepcatalist\\plots\\histogram for age.png')

### 🟡 Plot histogram for gender


fig = sns.FacetGrid(dataset, hue='gender', aspect=4)
fig.map(sns.kdeplot, 'age', fill=True)
oldest = dataset['age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()
fig.savefig('E:\\My Drive\\deepcatalist\\plots\\histogram for gender.png')


### 🟡 Plot histogram for ethnicity"""

fig = sns.FacetGrid(dataset, hue='ethnicity', aspect=4)
fig.map(sns.kdeplot, 'age')
oldest = dataset['age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()
fig.savefig('E:\\My Drive\\deepcatalist\\plots\\histogram for ethnicity.png')


# 🟡 Calculate the cross-tabulation of gender and ethnicity using the pandas.crosstab() function."""

tab = pd.crosstab(dataset['ethnicity'], dataset['gender'], rownames=['ethnicity'], colnames=['gender'])
tab

fig = sns.countplot(x=dataset['gender'], hue=dataset['ethnicity'], palette='magma', alpha=0.7).figure
fig.savefig('E:\\My Drive\\deepcatalist\\plots\\cross-tabulation of gender and ethnicity.png')


# 🟡 Create violin plots and box plots for age, separately for men and women."""

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

fig = sns.boxplot(data=dataset, y='age', hue='gender').figure
fig.savefig('E:\\My Drive\\deepcatalist\\plots\\boxplot age gender.png')



fig = sns.violinplot(data=dataset, hue='gender', x='age').figure
fig.savefig('E:\\My Drive\\deepcatalist\\plots\\violinplot age gender.png')


# 🟡 Create violin plots and box plots for age, separately for each ethnicity."""

fig = sns.boxplot(data=dataset, y='age', hue='ethnicity').figure
fig.savefig('E:\\My Drive\\deepcatalist\\plots\\boxplot age ethnicity.png')


fig = sns.violinplot(data=dataset, hue='ethnicity', y='age').figure
fig.savefig('E:\\My Drive\\deepcatalist\\plots\\violinplot age ethnicity.png')


