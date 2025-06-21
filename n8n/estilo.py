import matplotlib.pyplot as plt

def aplicar_estilo_minitab():
    plt.style.use("default")  # Base segura
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "font.size": 13,
        "font.family": "sans-serif",
        "grid.linestyle": "--",
        "grid.color": "#CCCCCC",
        "grid.alpha": 0.7,
        "legend.frameon": False
    })
    plt.grid(True)
