import pickle
import matplotlib.pyplot as plt

##############################################################################
# 1) Mapping from batch size to line style. GD is treated as bs=None.
##############################################################################
def get_linestyle(bs):
    """
    Return a matplotlib line style string for a given batch size.
    We treat GD (bs=None) the same as bs=64 => solid line '-'.
    """
    if bs is None or bs == 64:
        return "-"
    elif bs == 1:
        return "--"
    elif bs == 8:
        return "-."
    else:
        return "-"

##############################################################################
# 2) Mapping from learning rate to color
##############################################################################
def assign_colors_to_lrs(all_lrs):
    """
    Given a collection (set) of learning rates, return a dict mapping
    each lr to a distinct color, e.g. 'C0', 'C1', ...
    """
    sorted_lrs = sorted(all_lrs)
    color_map = {}
    for i, lr in enumerate(sorted_lrs):
        color_map[lr] = f"C{i}"  # e.g. 'C0', 'C1', ...
    return color_map

##############################################################################
# 3) Our plotting function
##############################################################################

def plot_experiment_results(gd_losses, sgd_losses, title="", filename=None):
    """
    gd_losses:   dict => {lr: {epoch: avg_loss}}
    sgd_losses:  dict => {batch_size: {lr: {epoch: avg_loss}}}

    We color by lr, line style by batch size, and keep line width small (1.0).
    GD markers are smaller (markersize=4), while SGD markers are slightly bigger.
    """
    plt.figure()

    # Collect all learning rates from both dictionaries.
    all_lrs = set(gd_losses.keys())
    for bs, lr_dict in sgd_losses.items():
        all_lrs.update(lr_dict.keys())

    # Build color map from lr -> 'C0', 'C1', ...
    color_map = assign_colors_to_lrs(all_lrs)

    # We'll use a relatively small line width for all curves.
    LINE_WIDTH = 1.0

    # We can use smaller markers for GD, a bit larger for SGD if desired.
    GD_MARKER_SIZE = 4
    SGD_MARKER_SIZE = 6

    # --- Plot GD curves ---
    for lr, epoch_dict in gd_losses.items():
        epochs = sorted(epoch_dict.keys())
        losses = [epoch_dict[e] for e in epochs]
        color = color_map[lr]
        ls = get_linestyle(None)  # treat GD as bs=None => same as bs=64
        label_str = f"GD (lr={lr})"

        plt.plot(
            epochs,
            losses,
            color=color,
            linestyle=ls,
            linewidth=LINE_WIDTH,
            marker='o',
            markersize=GD_MARKER_SIZE,
            label=label_str
        )

    # --- Plot SGD curves ---
    for bs, lr_dict in sgd_losses.items():
        for lr, epoch_dict in lr_dict.items():
            epochs = sorted(epoch_dict.keys())
            losses = [epoch_dict[e] for e in epochs]
            color = color_map[lr]
            ls = get_linestyle(bs)
            label_str = f"SGD (lr={lr}, bs={bs})"

            plt.plot(
                epochs,
                losses,
                color=color,
                linestyle=ls,
                linewidth=LINE_WIDTH,
                marker='x',
                markersize=SGD_MARKER_SIZE,
                label=label_str
            )

    plt.xlabel("Epoch")
    plt.xscale('log')
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(loc='upper right')

    # ------------------------------
    #  Force the x-axis ticks here
    # ------------------------------
    plt.xticks([1, 8, 64], [1, 8, 64])

    if filename:
        plt.savefig(filename, dpi=150)
        plt.close()
    else:
        plt.show()


##############################################################################
# 4) Main script: load pickle files, make three figures
##############################################################################

def main():
    # 1) Load the pickled dictionaries
    with open("gd_sin.pkl", "rb") as f:
        gd_sin = pickle.load(f)
    with open("sgd_sin.pkl", "rb") as f:
        sgd_sin = pickle.load(f)

    with open("gd_scalar.pkl", "rb") as f:
        gd_scalar = pickle.load(f)
    with open("sgd_scalar.pkl", "rb") as f:
        sgd_scalar = pickle.load(f)

    with open("gd_vector.pkl", "rb") as f:
        gd_vector = pickle.load(f)
    with open("sgd_vector.pkl", "rb") as f:
        sgd_vector = pickle.load(f)

    # 2) Create and save 3 plots, one per task
    plot_experiment_results(
        gd_sin, sgd_sin, 
        title="Task f1: sin(wx)",
        filename="f1_sin_plot.png"
    )

    plot_experiment_results(
        gd_scalar, sgd_scalar, 
        title="Task f2: sigmoid(MLP)",
        filename="f2_scalar_plot.png"
    )

    plot_experiment_results(
        gd_vector, sgd_vector, 
        title="Task f3: MLP(x)/||MLP(x)||",
        filename="f3_vector_plot.png"
    )


if __name__ == "__main__":
    main()
