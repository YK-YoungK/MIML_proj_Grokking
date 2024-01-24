#!/usr/bin/env python

import os
import grok
import matplotlib.pyplot as plt


model = "mlp"
c = "gen"

parser = grok.training.add_args(model)
parser.set_defaults(logdir=os.environ.get("LOGDIR", "."))
hparams = parser.parse_args()
hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)


print(hparams)


if c == "mem":
    ckpts = [(f"./model-mlp/model-mlp-decay-1e-1-{i}/checkpoints/epoch_mem.ckpt", i) for i in range(30, 100, 10)]
elif c == "gen":
    ckpts = [(f"./model-mlp/model-mlp-decay-1e-1-{i}/checkpoints/epoch_gen.ckpt", i) for i in range(30, 100, 10)]


data = []
for ckpt in ckpts:
    data = data + grok.training.draw_scatter_plot(hparams, model, ckpt[0])



# Unpack the tuples into separate lists
x, y, z = zip(*data)

# Define a colormap (you can choose one from Matplotlib or create your own)
colormap = 'viridis_r'

# Create a scatter plot
fig = plt.figure()

# Scatter plot with color determined by z
scatter = plt.scatter(x, y, c=z, cmap=colormap)

# Add colorbar
cbar = plt.colorbar(scatter)

# Set labels and title
plt.xlabel('Parameter Norm')
plt.ylabel('Correct Logit')
cbar.set_label('Training Data Percentage (%)')
plt.title('Scatter Plot for Generalization Models in MLP')


# Save the plot as an image file (e.g., PNG, JPG, etc.)
plt.savefig('dataset_' + model + '_' + c + '.pdf')  # Change the file extension as needed (e.g., 'scatter_plot.jpg')

# Show the plot (optional)
plt.show()