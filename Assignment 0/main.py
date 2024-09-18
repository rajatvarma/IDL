import csv
import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap
from sklearn.manifold import TSNE

# Open and parse csv files
trainInFile = open("train_in.csv") 
trainOutFile = open("train_out.csv")
train_in_reader = csv.reader(trainInFile)
train_out_reader = csv.reader(trainOutFile)

# Generate array of all corresponding number values of each vector
ins = [[float(i) for i in line] for line in train_in_reader]
outs = [int(i[0]) for i in train_out_reader]

# Initialise dictionary which will contain all the centers (means)
centers = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": []}

# Calculate the center for each digit
indexNumber = 0
for numVector in ins:
    correspondingNumber = outs[indexNumber]
    if len(centers[str(correspondingNumber)]) == 0:
        centers[str(correspondingNumber)] = numVector
    else:
        for j in range(256):
            centers[str(correspondingNumber)][j] = centers[str(correspondingNumber)][j] + numVector[j]
    indexNumber += 1

for key in centers.keys():
    length = len(centers[key])
    centers[key] = [i/length for i in centers[key]]

# Initialise the distance matrix
dist = [[0 for j in range(10)] for i in range(10)]

# Calculate the Euclidean distance between all the centers
for i in range(10):
    for j in range(10):
        squared = [(centers[str(i)][k] - centers[str(j)][k])**2 for k in range(256)]
        distance = round(math.sqrt(sum(squared)), 3)
        dist[i][j] = distance

for i in dist:
    print(i)

# PCA and Plotting
colorsIndex = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#ab44fe", "#0012b1", "#000000", "#ff12a1"]
colors = [colorsIndex[i] for i in outs]

# Normalize data using Min-Max Scaling
scaler = MinMaxScaler()
X = scaler.fit_transform(ins)

# Apply PCA
pca = PCA(n_components=2)
Y = pca.fit_transform(X)

# Apply UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(X)

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Filter PCA results
mask = (Y[:, 0] <= 1) & (Y[:, 1] <= 1)
Y_filtered = Y[mask]
colors_filtered = np.array(colors)[mask]

# Plot PCA results
axs[0].scatter(Y_filtered[:, 0], Y_filtered[:, 1], c=colors_filtered)
unique_labels = list(set(outs))
legend_handles = [mpatches.Patch(color=colorsIndex[label], label=str(label)) for label in unique_labels]
axs[0].legend(handles=legend_handles, title="Digits")
axs[0].set_title('PCA')

# Plot UMAP results
axs[1].scatter(embedding[:, 0], embedding[:, 1], c=colors)
axs[1].legend(handles=legend_handles, title="Digits")
axs[1].set_title('UMAP')

# Plot t-SNE results
axs[2].scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
axs[2].legend(handles=legend_handles, title="Digits")
axs[2].set_title('t-SNE')

# Show the plots
plt.tight_layout()
plt.show()