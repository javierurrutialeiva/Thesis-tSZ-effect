from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

milliquas = fits.open('/data2/javierurrutia/szeffect/data/milliquas.fits')
table = milliquas[1].data
types_unique = np.unique(table["type"])
types = table['type']
type_letters = ["Q","A","B","K","N"]
type_names = ["QSO","AGN","BL Lac", "Narrow Lines", "Seyferts"]
grouped_types = np.full_like(types, 'Other', dtype='U10')
colors = ['purple','pink','darkgreen','blue','cyan']
for i,name in enumerate(type_names):
    grouped_types[types.startswith(type_letters[i])] = name

splitted_data = {}
for i,name in enumerate(type_names):
    subdata =  table["z"][(grouped_types == name)]
    splitted_data[str(name)] = subdata[np.where(np.isnan(subdata) == False)]

fig = plt.figure(figsize = (7,4))
ax = plt.axes()
ax.hist(table['z'], histtype = 'step', edgecolor = 'red', lw = 3, alpha = 0.6, label = 'entire sample',log = True, bins = 30, ls = '--')
for i,name in enumerate(type_names):
    ax.hist(splitted_data[str(name)], histtype = 'step', edgecolor = colors[i], lw = 2, alpha = 0.6, label = name, log = True, bins = 30)

ax.set(xlabel = "redshift $z$", ylabel = "N of samples", title = "redshift distribution in milliquos")
ax.grid(True)
ax.legend()
fig.savefig('milliquos_hist.png')