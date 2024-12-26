import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_excel(r".\tcga_infor\boxplot.xlsx")
data_1 = pd.read_excel(r".\tcga_infor\boxplot.xlsx")

# Set plotting style
sns.set(style="whitegrid")

# Assume the last column is grouping information, we select one of them as the grouping basis
group_column = data.columns[-1]  # Select the last column as the grouping basis
g_c = data_1.columns[-1]

# Create a subplot layout with 2 rows and 4 columns
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 8))
axes = axes.flatten()  # Flatten the axes array for easier iteration
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Use Microsoft YaHei font
plt.rcParams['axes.unicode_minus'] = False  # Correctly display minus signs

# Plot grouped box plots for each feature
for i, feature in enumerate(data.columns[:4]):
    sns.boxplot(x=group_column, y=feature, data=data, ax=axes[i])
    axes[i].set_xlabel('')  # Hide x-axis label
    axes[i].set_ylabel(feature, fontsize=18)  # Set y-axis label and font size
    axes[i].xaxis.set_tick_params(labelsize=16)  # Set x-axis tick label size
    axes[i].yaxis.set_tick_params(labelsize=14)  # Set y-axis tick label size

c = ['lightblue', '#95B37F']
for i, feature in enumerate(data_1.columns[:4]):
    sns.boxplot(x=g_c, y=feature, data=data_1, ax=axes[i+4], palette=c)
    axes[i+4].set_xlabel('')  # Hide x-axis label
    axes[i+4].set_ylabel(feature, fontsize=18)  # Set y-axis label and font size
    axes[i+4].xaxis.set_tick_params(labelsize=16)  # Set x-axis tick label size
    axes[i+4].yaxis.set_tick_params(labelsize=14)  # Set y-axis tick label size

plt.tight_layout()
# plt.show()
plt.savefig(r'.\Figures.png')
