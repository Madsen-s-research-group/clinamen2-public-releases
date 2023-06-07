"""Use CMAPlotHelper to create a trajectory plot."""
import pathlib

import matplotlib.pyplot as plt

from clinamen2.utils.plot_functions import CMAPlotHelper

LABEL = "cigar_trial"
GENERATIONS = -1  # last generation will be read from evolution.json
WORK_DIR = pathlib.Path.cwd() / LABEL

plotter = CMAPlotHelper(
    label=LABEL,
    generation_bounds=(1, GENERATIONS),
    input_dir=WORK_DIR,
    json=True,
)

# the default figure, showing mean loss and step size over generations
fig, _ = plotter.plot_mean_loss_per_generation(
    generation_bounds=plotter.generation_bounds,
    show_sigma=True,
    sigma_e_mult=1.0,
    y_units=r" / \si{\electronvolt\per\angstrom\squared}",
    save_fig=True,
)

plt.show()
plt.close()


# loss, step size and (if applicable) additional information can be accessed
# through the CMAPlotHelper

# diy figure
fig = plt.figure()
ax = fig.add_subplot(111)

# losses have the shape (generations, population_size)
# plot mean loss for the first 50 generations
ax.plot(plotter.losses.mean(axis=1)[:50], color="black")
ax.set_xlabel("generations")
ax.set_ylabel("loss")

# add step size on second y-axis
ax2 = ax.twinx()
ax2.set_ylabel("step size", color="red")
ax2.tick_params(axis="y", colors="red")
ax2.plot(plotter.step_sizes[:50], color="red", linestyle="dashed")
plt.show()
