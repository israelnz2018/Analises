import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
import os
import io
from io import BytesIO
from estilo import aplicar_estilo_minitab
from suporte import interpretar_coluna


def salvar_grafico():
    caminho = "grafico.png"
    plt.tight_layout()
    plt.savefig(caminho)
    plt.close()
    with open(caminho, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(caminho)
    return img_base64

def gerar_histograma(df: pd.DataFrame, colunas: list, coluna_y=None):
    if not colunas or len(colunas) < 1:
        raise ValueError("⚠ O histograma precisa ter ao menos uma coluna X informada.")

    coluna_x = colunas[0].strip()
    coluna_subgrupo = colunas[1].strip() if len(colunas) > 1 else None

    if coluna_x not in df.columns:
        raise ValueError(f"A coluna '{coluna_x}' não foi encontrada no arquivo.")

    aplicar_estilo_minitab()
    plt.figure(figsize=(10, 6))

    if coluna_subgrupo and coluna_subgrupo in df.columns:
        grupos = df.groupby(coluna_subgrupo)
        cores = sns.color_palette("tab10", n_colors=len(grupos))

        for i, (nome, grupo) in enumerate(grupos):
            sns.histplot(
                grupo[coluna_x],
                bins=10,
                kde=True,
                stat="density",
                alpha=0.4,
                label=str(nome),
                color=cores[i],
                edgecolor="black"
            )
    else:
        sns.histplot(
            df[coluna_x],
            bins=10,
            kde=True,
            stat="density",
            alpha=0.4,
            color="steelblue",
            edgecolor="black"
        )

    plt.xlabel(coluna_x)
    plt.ylabel("Densidade")
    plt.title("Histograma com Curva de Densidade")
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    imagem_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return imagem_base64


def gerar_pareto(df, colunas_usadas):
    if len(colunas_usadas) < 1:
        return "Erro: Coluna X (atributivo) obrigatória não fornecida.", None

    col_x = colunas_usadas[0]
    col_sub = colunas_usadas[1] if len(colunas_usadas) > 1 else None
    col_y = colunas_usadas[2] if len(colunas_usadas) > 2 else None

    resultado_texto = []
    imagens = []

    if col_x and not col_y and not col_sub:
        contagem = df[col_x].value_counts().sort_values(ascending=False)
        if contagem.sum() == 0:
            return "Sem dados para gerar Pareto.", None
        resultado_texto.append(contagem.to_frame().to_html())

        plt.figure(figsize=(8,5))
        ax = contagem.plot(kind="bar")
        cum = contagem.cumsum() / contagem.sum() * 100
        ax2 = cum.plot(secondary_y=True, color='r', marker='o', ax=ax)
        ax.set_ylabel("Frequência")
        ax2.set_ylabel("Acumulado (%)")
        ax.set_title(f"Pareto - {col_x}")
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        imagens.append(base64.b64encode(buf.read()).decode("utf-8"))

    elif col_x and col_y and not col_sub:
        dados = df[[col_x, col_y]].dropna()
        soma = dados.groupby(col_x)[col_y].sum().sort_values(ascending=False)
        if soma.sum() == 0:
            return "Sem dados para gerar Pareto.", None
        resultado_texto.append(soma.to_frame().to_html())

        plt.figure(figsize=(8,5))
        ax = soma.plot(kind="bar")
        cum = soma.cumsum() / soma.sum() * 100
        ax2 = cum.plot(secondary_y=True, color='r', marker='o', ax=ax)
        ax.set_ylabel("Soma de Y")
        ax2.set_ylabel("Acumulado (%)")
        ax.set_title(f"Pareto - {col_x} (Y={col_y})")
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        imagens.append(base64.b64encode(buf.read()).decode("utf-8"))

    elif col_x and col_y and col_sub:
        dados = df[[col_x, col_sub, col_y]].dropna()
        for subgrupo, sub_df in dados.groupby(col_sub):
            soma = sub_df.groupby(col_x)[col_y].sum().sort_values(ascending=False)
            if soma.sum() == 0:
                continue
            resultado_texto.append(f"<strong>Subgrupo {subgrupo}</strong><br>" + soma.to_frame().to_html())

            plt.figure(figsize=(8,5))
            ax = soma.plot(kind="bar")
            cum = soma.cumsum() / soma.sum() * 100
            ax2 = cum.plot(secondary_y=True, color='r', marker='o', ax=ax)
            ax.set_ylabel("Soma de Y")
            ax2.set_ylabel("Acumulado (%)")
            ax.set_title(f"Pareto - {col_x} (Y={col_y}, Subgrupo={subgrupo})")
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            imagens.append(base64.b64encode(buf.read()).decode("utf-8"))

    elif col_x and not col_y and col_sub:
        dados = df[[col_x, col_sub]].dropna()
        for subgrupo, sub_df in dados.groupby(col_sub):
            contagem = sub_df[col_x].value_counts().sort_values(ascending=False)
            if contagem.sum() == 0:
                continue
            resultado_texto.append(f"<strong>Subgrupo {subgrupo}</strong><br>" + contagem.to_frame().to_html())

            plt.figure(figsize=(8,5))
            ax = contagem.plot(kind="bar")
            cum = contagem.cumsum() / contagem.sum() * 100
            ax2 = cum.plot(secondary_y=True, color='r', marker='o', ax=ax)
            ax.set_ylabel("Frequência")
            ax2.set_ylabel("Acumulado (%)")
            ax.set_title(f"Pareto - {col_x} (Subgrupo={subgrupo})")
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            imagens.append(base64.b64encode(buf.read()).decode("utf-8"))

    else:
        return "Configuração de colunas não reconhecida.", None

    return "<br><br>".join(resultado_texto), imagens


def grafico_linha_temporal(df, colunas_usadas, coluna_y=None):
    if len(colunas_usadas) != 2:
        raise ValueError("O gráfico de série temporal requer duas colunas: X (datas/períodos) e Y (valores numéricos).")

    nome_x = colunas_usadas[1]  # X (tempo)
    nome_y = colunas_usadas[0]  # Y (valor)

    # Força conversão da coluna Y para numérico
    df[nome_y] = pd.to_numeric(df[nome_y], errors="coerce")
    if df[nome_y].isnull().all():
        raise ValueError(f"A coluna '{nome_y}' não contém valores numéricos válidos.")

    # Tenta converter X para datetime (se possível)
    try:
        df[nome_x] = pd.to_datetime(df[nome_x], errors='coerce')
    except Exception:
        pass

    # Verifica se ao menos parte das datas foram entendidas
    if df[nome_x].isnull().all():
        raise ValueError(f"A coluna '{nome_x}' não contém datas ou períodos válidos para série temporal.")

    # Ordena pela coluna X (tempo)
    df = df.sort_values(by=nome_x)

    aplicar_estilo_minitab()

    plt.figure(figsize=(10, 5))
    plt.plot(df[nome_x], df[nome_y], color='blue', marker='o', markerfacecolor='red')
    plt.xlabel(nome_x)
    plt.ylabel(nome_y)
    plt.title("Gráfico de Série Temporal")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return imagem_base64



def grafico_ic_media(df, colunas_usadas, coluna_y=None):
    if len(colunas_usadas) != 2:
        raise ValueError("O gráfico de intervalos requer uma coluna Y (numérica) e uma coluna X (categórica).")

    nome_y = colunas_usadas[0]  # ✅ Agora Y vem primeiro
    nome_x = colunas_usadas[1]

    # Força conversão da coluna Y para numérico
    df[nome_y] = pd.to_numeric(df[nome_y], errors="coerce")
    if df[nome_y].isnull().all():
        raise ValueError(f"A coluna '{nome_y}' não contém valores numéricos válidos para calcular a média.")

    aplicar_estilo_minitab()

    grupos = df.groupby(nome_x)[nome_y]
    medias = grupos.mean()
    desvios = grupos.std()
    n = grupos.count()
    erro_padrao = desvios / n**0.5
    intervalo = 1.96 * erro_padrao

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x=medias.index,
        y=medias.values,
        yerr=intervalo.values,
        fmt='o',
        capsize=6,
        elinewidth=2,
        marker='o',
        color='midnightblue'
    )

    plt.title("Intervalos de Confiança (IC 95%) para a Média")
    plt.xlabel(nome_x)
    plt.ylabel(nome_y)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return imagem_base64



def grafico_pizza(df, colunas_usadas, coluna_y=None):
    if len(colunas_usadas) != 1:
        raise ValueError("O Gráfico de Pizza requer exatamente 1 coluna categórica ou discreta.")

    nome_coluna = colunas_usadas[0]
    dados = df[nome_coluna].value_counts()

    aplicar_estilo_minitab()

    plt.figure(figsize=(8, 8))
    plt.pie(
        dados.values,
        labels=dados.index,
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': 12}
    )
    plt.title(f"Distribuição de {nome_coluna}", fontsize=14)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return imagem_base64

def grafico_bolhas(df, colunas_usadas, coluna_y=None):
    if len(colunas_usadas) != 3:
        raise ValueError("O Gráfico de Bolhas requer 3 colunas: X, Y e Tamanho.")

    nome_x = colunas_usadas[0]
    nome_y = colunas_usadas[1]
    nome_tamanho = colunas_usadas[2]

    aplicar_estilo_minitab()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x=nome_x,
        y=nome_y,
        size=nome_tamanho,
        sizes=(50, 800),
        legend=False,
        alpha=0.6
    )

    plt.xlabel(nome_x)
    plt.ylabel(nome_y)
    plt.title("Gráfico de Bolhas")

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return imagem_base64



def grafico_dispersao(df, colunas):
    if len(colunas) < 2:
        raise ValueError("Gráfico de dispersão requer exatamente duas colunas.")
    plt.figure(figsize=(8, 6))
    aplicar_estilo_minitab()
    sns.scatterplot(x=df[colunas[0]], y=df[colunas[1]])
    plt.title("Gráfico de Dispersão")
    return salvar_grafico()

def grafico_boxplot_simples(df, colunas, coluna_y=None):
    if not coluna_y:
        raise ValueError("Para o boxplot simples, a coluna Y (numérica) é obrigatória.")

    y = df[coluna_y].astype(str).str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(y) < 2:
        raise ValueError("Coluna Y deve conter ao menos dois valores numéricos.")

    df_box = pd.DataFrame({coluna_y: y, "grupo": "A"})

    plt.figure(figsize=(6, 6))
    aplicar_estilo_minitab()
    sns.boxplot(data=df_box, x="grupo", y=coluna_y, color="#89CFF0", width=0.3)
    sns.pointplot(data=df_box, x="grupo", y=coluna_y, estimator=np.mean,
                  markers="D", color="red", scale=1.2, errwidth=0)
    plt.xlabel("")
    plt.ylabel(coluna_y)
    plt.title("Boxplot Simples com Média (losango)")
    return salvar_grafico()


def grafico_boxplot_multiplo(df, colunas, coluna_y=None):
    if not coluna_y or not coluna_y.strip():
        raise ValueError("Você deve selecionar uma coluna Y com valores numéricos para o boxplot múltiplo.")

    coluna_y = coluna_y.strip()
    if coluna_y.startswith("Unnamed") or coluna_y not in df.columns:
        raise ValueError(f"A coluna Y '{coluna_y}' não tem título válido ou não foi encontrada.")

    y = df[coluna_y].astype(str).str.replace(",", ".").str.replace(r"[^\d\.\-]", "", regex=True)
    y = pd.to_numeric(y, errors="coerce")
    if y.dropna().shape[0] < 2:
        raise ValueError("A coluna Y deve conter ao menos dois valores numéricos válidos.")

    colunas = [c.strip() for c in colunas if c.strip() and c.strip() != coluna_y and not c.strip().startswith("Unnamed")]
    if not colunas:
        raise ValueError("Nenhuma coluna X válida foi selecionada para o agrupamento.")

    x_col = colunas[0]
    if x_col not in df.columns:
        raise ValueError(f"A coluna X '{x_col}' não foi encontrada no arquivo.")

    grupo = df[x_col].astype(str)
    df_plot = pd.DataFrame({coluna_y: y, x_col: grupo}).dropna()

    if df_plot.empty:
        raise ValueError("Os dados da coluna Y e do grupo X selecionado não têm valores válidos simultaneamente.")

    plt.figure(figsize=(10, 6))
    aplicar_estilo_minitab()

    sns.boxplot(data=df_plot, x=x_col, y=coluna_y, color="#89CFF0")
    sns.pointplot(data=df_plot, x=x_col, y=coluna_y, estimator=np.mean,
                  markers="D", color="red", scale=1.1, errwidth=0)

    plt.title(f"Boxplot Múltiplo por '{x_col}'")
    plt.xlabel(x_col)
    plt.ylabel(coluna_y)
    plt.xticks(rotation=45)

    return salvar_grafico()




def grafico_barras_simples(df, colunas_usadas):
    if len(colunas_usadas) != 1:
        raise ValueError("Selecione exatamente uma coluna para o Gráfico de Barras Simples.")

    nome_coluna = colunas_usadas[0]
    serie = df[nome_coluna].dropna()
    contagem = serie.value_counts().sort_index()

    aplicar_estilo_minitab()
    fig, ax = plt.subplots(figsize=(6, 4))
    contagem.plot(kind='bar', color='skyblue', ax=ax)

    ax.set_title(f"Gráfico de Barras - {nome_coluna}")
    ax.set_xlabel(nome_coluna)
    ax.set_ylabel("Frequência")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return imagem_base64


def grafico_barras_agrupado(df, colunas_x, coluna_y=None):
    if not colunas_x or len(colunas_x) < 2:
        raise ValueError("Selecione duas colunas no campo X — a primeira será o eixo X, a segunda o agrupador (cores).")

    coluna_x = colunas_x[0]            # Eixo X = primeiro item do Drop X
    coluna_agrupador = colunas_x[1]    # Agrupador (cor) = segundo item do Drop X

    if coluna_x not in df.columns:
        raise ValueError(f"A coluna '{coluna_x}' não existe no DataFrame.")
    if coluna_agrupador not in df.columns:
        raise ValueError(f"A coluna '{coluna_agrupador}' não existe no DataFrame.")

    aplicar_estilo_minitab()

    # Tabela cruzada
    tabela = df.groupby([coluna_x, coluna_agrupador]).size().unstack(fill_value=0)

    # Gráfico com barras agrupadas
    fig, ax = plt.subplots(figsize=(10, 6))
    largura_barra = 0.8 / len(tabela.columns)
    posicoes = np.arange(len(tabela))

    for i, categoria in enumerate(tabela.columns):
        valores = tabela[categoria].values
        ax.bar(posicoes + i * largura_barra, valores, width=largura_barra, label=str(categoria))

    ax.set_xticks(posicoes + largura_barra * (len(tabela.columns) - 1) / 2)
    ax.set_xticklabels(tabela.index, rotation=45)

    ax.set_title(f'Gráfico de Barras Agrupado: {coluna_x} por {coluna_agrupador}')
    ax.set_xlabel(coluna_x)
    ax.set_ylabel("Frequência")
    ax.legend(title=coluna_agrupador)
    plt.tight_layout()

    # Exporta imagem como base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    imagem_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return imagem_base64


GRAFICOS = {
    "Histograma": gerar_histograma,
    "Pareto": gerar_pareto,
    "Gráfico de tendecias": grafico_linha_temporal,
    "grafico_ic_media": grafico_ic_media,
    "Gráfico de pizza": grafico_pizza,
    "Gráficos de bolhas": grafico_bolhas,
    "Gráfico de disperao": grafico_dispersao,
    "BoxPlot simples": grafico_boxplot_simples,
    "boxplot_multiplo": grafico_boxplot_multiplo,
    "Gráfifo de barras": grafico_barras_simples,
    "Gráfifo de barras com subgrupo": grafico_barras_agrupado,

    
  
    


}
