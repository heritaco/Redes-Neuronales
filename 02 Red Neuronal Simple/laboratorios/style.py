import seaborn as sns

def apply():
    sns.set(
        style="whitegrid",
        palette="muted",
        font="serif",
        font_scale=1.2,
        rc={
            "grid.linestyle": "--",
            "axes.edgecolor": "white",
            "axes.linewidth": 0.8,
            "grid.color": "lightgray",
            "figure.figsize": (8, 8),
            "axes.titlesize": 20,
            "axes.labelsize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
        },
    )