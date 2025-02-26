import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tueplots import bundles
from scipy.ndimage import gaussian_filter1d
from tueplots.constants.color import rgb
from tueplots import cycler
from tueplots.constants import markers
from tueplots.constants.color import palettes
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import re
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

def extract_tensorboard_scalars(logdir):
    """
    Extracts all scalar values from TensorBoard event files in subdirectories.
    Returns a dictionary where keys are subdirectory names and values are DataFrames.
    """
    df_list = []
    subdir_list = []

    for root, _, files in os.walk(logdir):
        event_files = [f for f in files if "events.out.tfevents" in f]
        if not event_files:
            continue  # Skip directories without event files

        subdir = os.path.basename(root)
        event_file = os.path.join(root, event_files[0])  # Assuming one event file per subdir

        event_acc = EventAccumulator(event_file, size_guidance={'scalars': 0})
        event_acc.Reload()

        scalars = {}
        for tag in event_acc.Tags()['scalars']:  # Extract all scalar tags
            scalars[tag] = [(e.step, e.value) for e in event_acc.Scalars(tag)]
        
        # Convert to DataFrame
        df = pd.DataFrame({tag: dict(values) for tag, values in scalars.items()})
        df_list.append(df)
        subdir_list.append(subdir)

    return df_list, subdir_list


def plot_strong_vs_weak_new(df_list_weak, df_list_strong, names_weak, names_strong, wanted_tag="Winrate", sigma=2):
    fig, ax = plt.subplots()


    for idx, (df, name) in enumerate(zip(df_list_strong, names_strong)):
        color = rgb.tue_darkblue
        smoothed_rewards = df[wanted_tag].rolling(window=smoothing_window, min_periods=1).mean()
        ax.plot(smoothed_rewards, ms=0.1, lw=0.5, color=color, zorder=3,label=f"Strong")

    for idx, (df, name) in enumerate(zip(df_list_weak, names_weak)):
        color = rgb.tue_darkgreen
        smoothed_rewards = df[wanted_tag].rolling(window=smoothing_window, min_periods=1).mean()
        ax.plot(smoothed_rewards, ms=0.1, lw=0.5, color=color, zorder=3,label=f"Weak")




    #ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid(axis="y", which="major", color=rgb.tue_gray, alpha=0.3, linewidth=0.3, zorder=1)


    ax.set_xlabel('Updates', fontsize=8)
    if wanted_tag == "Mean Reward":
        ax.set_ylabel("Average Reward", fontsize=8)
    else:
        ax.set_ylabel(wanted_tag, fontsize=8)
    ax.set_title('Training vs Basic Opponent', fontsize=8)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=4.5, loc='lower right')

    plt.show()
    fig.savefig('./PPO/performance_plots/opponent_comparison/' + wanted_tag +'.png')
    fig.savefig('./PPO/performance_plots/opponent_comparison/' + wanted_tag +'.pdf')



def plot_gammma(df_list, names, wanted_tag="Winrate", sigma=2):
    fig, ax = plt.subplots()
    cmap = cm.get_cmap("tab10", len(df_list))


    sorted_pairs = sorted(zip(names, df_list), key=lambda pair: float(re.search(r"[-+]?\d*\.?\d+", pair[0]).group()))
    sorted_names, sorted_df_list = zip(*sorted_pairs)

    names = list(sorted_names)
    df_list = list(sorted_df_list)

    for idx, (df, name) in enumerate(zip(df_list, names)):
        color = cmap(idx)
        gamma_value = re.search(r"[-+]?\d*\.?\d+", name).group()
        smoothed_rewards = df[wanted_tag].rolling(window=smoothing_window, min_periods=1).mean()
        ax.plot(smoothed_rewards, ms=0.1, lw=0.5, color=color, zorder=3,label=f"gamma = {gamma_value}", linestyle='-')





    #ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid(axis="y", which="major", color=rgb.tue_gray, alpha=0.3, linewidth=0.3, zorder=1)


    ax.set_xlabel('Updates', fontsize=8)
    if wanted_tag == "Mean Reward":
        ax.set_ylabel("AverageReward", fontsize=8)
    else:
            ax.set_ylabel(wanted_tag, fontsize=8)
    ax.set_title('Training for different Discount Factors', fontsize=8)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=4, loc='lower right')

    plt.show()
    fig.savefig('./PPO/performance_plots/gamma_comparison/' + wanted_tag +'.png')
    fig.savefig('./PPO/performance_plots/gamma_comparison/' + wanted_tag +'.pdf')




smoothing_window = 10

strong_logdir = "PPO/tensorboard/strong_agent_runs"
weak_logdir = "PPO/tensorboard/weak_agent_runs"
gamma_logdir = "PPO/tensorboard/gamma_runs"

plt.rcParams.update(bundles.icml2022(column="half", nrows=1, ncols=1, usetex=False))
plt.rcParams.update({"figure.dpi": 1000, "font.family": "Nimbus Roman"})

gamma_dataframes, gamma_names = extract_tensorboard_scalars(gamma_logdir)
weak_dataframes, weak_names = extract_tensorboard_scalars(weak_logdir)
strongdataframes, strong_names = extract_tensorboard_scalars(strong_logdir)
plot_strong_vs_weak_new(weak_dataframes, strongdataframes, weak_names, strong_names, wanted_tag="Winrate", sigma=2)
plot_strong_vs_weak_new(weak_dataframes, strongdataframes, weak_names, strong_names, wanted_tag="Mean Reward", sigma=2)
plot_gammma(gamma_dataframes, gamma_names, wanted_tag="Winrate", sigma=2)
plot_gammma(gamma_dataframes, gamma_names, wanted_tag="Mean Reward", sigma=2)






