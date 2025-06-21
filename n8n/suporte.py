# 🔢 Bibliotecas de análise de dados
import pandas as pd
import numpy as np

# 📊 Visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# 🧪 Testes estatísticos
from scipy import stats
from scipy.stats import chi2_contingency, anderson, shapiro, kstest, norm

# 📦 Modelos estatísticos
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.diagnostic import normal_ad

# 📈 Métricas de modelos
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import PowerTransformer

# 💾 Manipulação de arquivos/imagens
from io import BytesIO
import base64
import os

# ✅ Função para aplicar o estilo
def aplicar_estilo_minitab():
    plt.style.use("default")
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

# 🔹 Você pode colocar outras funções de suporte aqui se quiser no futuro
