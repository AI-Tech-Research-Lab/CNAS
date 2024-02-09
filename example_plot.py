from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import seaborn as seaborn 
import numpy as np

FONT_SIZE = 8
FIGSIZE = (3.5, 3.0)

# Other colors here
COLORS = [mcolors.TABLEAU_COLORS[k] for k in mcolors.TABLEAU_COLORS.keys()]

seaborn.set_style('whitegrid')

# Only for toy purposes
def generate_data():
    return np.random.normal(size=(5))

fig, ax = plt.subplots(figsize=FIGSIZE, facecolor='w', edgecolor='k')
ax.scatter(np.arange(5), generate_data(), 
           marker='d', label='Example data', color=COLORS[0], s=50, linewidths=1.0)

ax.legend(loc='upper right', prop={'size': FONT_SIZE - 1})
ax.set_xlabel('Time', fontsize=FONT_SIZE)
ax.set_ylabel('Accuracy', fontsize=FONT_SIZE)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)

fig.set_tight_layout(True)
    
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color('black')
    ax.spines[spine].set_linewidth(1)

#fig.savefig('accuracy.pdf')