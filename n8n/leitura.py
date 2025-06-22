import pandas as pd
import io
import os

from fastapi.responses import JSONResponse

async def ler_arquivo(arquivo, aba=None):
    if arquivo and arquivo.filename.endswith(".xlsx"):
        await arquivo.seek(0)
        file_bytes = await arquivo.read()
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", sheet_name=aba)
        df.columns = df.columns.map(str).map(str.strip)
        return df
    else:
        raise ValueError("Envie um arquivo Excel (.xlsx) válido.")
