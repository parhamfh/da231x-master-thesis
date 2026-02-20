import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logger = logging.getLogger(__name__)
plt.switch_backend(
    "Agg"
)  # https://stackoverflow.com/a/4935945 and https://stackoverflow.com/a/64773637


def save_R_plot(R, plot_name, output_dir, *, invert_y=False, same_scale=False):
    fig, ax = plt.subplots()

    if same_scale:
        im = ax.imshow(R, vmin=-1, vmax=0)
    else:
        im = ax.imshow(R)
    fig.colorbar(im)

    num_beats = R.shape[0]
    logger.info(
        f"Plotting R matrix. num_beats={num_beats}, plot_name={plot_name},"
        f" invert_y={invert_y}, same_scale={same_scale}"
    )

    if invert_y:
        plt.gca().invert_yaxis()
    else:
        ax.xaxis.tick_top()

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.title(f"Recurrence matrix: {plot_name}")
    plt.savefig(os.path.join(output_dir, f"{plot_name}_RecMx.png"))


if __name__ == "__main__":
    m = np.eye(7) + np.random.normal(scale=0.1, size=(7, 7))
    save_R_plot(m, "eye", ".")
