"""
Definición de los subconjuntos de lesiones y de los cruces benigno / maligno que
se evalúan de forma aislada al terminar el entrenamiento.

Un cruce aísla dos grupos de lesiones y mide el desempeño del modelo solo sobre
ellos, de forma que el comportamiento por subtipo molecular no quede escondido
detrás del resultado global. La pertenencia a cada grupo se deduce de tres
fuentes distintas:

    - Subtipo molecular (luminal_a, luminal_b, her2_enriched, tnbc): columna
      `subtype` del csv de radiómica.
    - Subconjunto TNBC del doctor: data/splits_tnbc_doctor.csv, que trae IDs de
      paciente, no de lesión.
    - Subconjuntos TNBC y benigno métricos: las columnas "Paciente TN" y
      "Pacientes BN" de data/TN_cercanos_vs_BN_cercanos.csv, con IDs de lesión.

El grupo `tnbc_ambos` no vive en ningún archivo: se construye aquí como la unión
de `tnbc_cercano` y `tnbc_doctor`.
"""

import sys

import pandas as pd


# Normalización de la columna `subtype` del CSV de radiómica. Debe coincidir con
# el mapeo de data/csvgeneration.py, que es el script que genera las particiones.
SUBTYPE_MAP = {
    "luminal a":        "luminal_a",
    "luminal b":        "luminal_b",
    "her2-enriched":    "her2_enriched",
    "triple negative":  "tnbc",
}

# Valor de subtipo para las lesiones sin subtipo molecular asignado.
NO_SUBTYPE = "none"

# Grupos de lesiones, en el orden en que se añaden a la tabla de pertenencia.
# Cada nombre es una columna booleana de `build_subset_table`.
GROUPS = [
    "benigno",
    "maligno",
    "luminal_a",
    "luminal_b",
    "her2_enriched",
    "tnbc",
    "tnbc_doctor",
    "tnbc_cercano",
    "tnbc_ambos",
    "benigno_cercano",
]

# Cruces evaluados: nombre -> (grupo negativo, grupo positivo). El nombre se usa
# como prefijo de las métricas en wandb, como valor de la columna `crossing` de
# los csv exportados y como prefijo de los nombres de archivo de las curvas.
CROSSINGS = {
    "Benigno_vs_Luminal-A"              : ("benigno",           "luminal_a"),
    "Benigno_vs_Luminal-B"              : ("benigno",           "luminal_b"),
    "Benigno_vs_HER2"                   : ("benigno",           "her2_enriched"),
    "Benigno_vs_TNBC"                   : ("benigno",           "tnbc"),
    "Benigno_vs_TNBC-cercano"           : ("benigno",           "tnbc_cercano"),
    "Benigno_vs_TNBC-doctor"            : ("benigno",           "tnbc_doctor"),
    "Benigno_vs_TNBC-ambos"             : ("benigno",           "tnbc_ambos"),
    "Benigno-cercano_vs_Maligno"        : ("benigno_cercano",   "maligno"),
    "Benigno-cercano_vs_TNBC-cercano"   : ("benigno_cercano",   "tnbc_cercano"),
}


def load_subtypes(csv_subtype: str) -> pd.DataFrame:
    """
    Read the radiomics CSV and return the molecular subtype of every lesion.

    Args:
        csv_subtype (str): path to the radiomics CSV.

    Returns:
        pd.DataFrame: one row per lesion with lesion_ID, patient_ID and the
            normalized subtype.
    """
    data = pd.read_csv(csv_subtype)

    subtypes = pd.DataFrame()
    subtypes["lesion_ID"] = data["lesion_id"].astype(str).str.strip()

    # El ID de paciente es el prefijo del lesion_ID (ej. D1-0001_R_1 -> D1-0001);
    # nunca contiene guion bajo, por eso basta con partir en el primero.
    subtypes["patient_ID"] = subtypes["lesion_ID"].str.split("_").str[0]

    # Subtipo molecular normalizado; las lesiones sin subtipo quedan como 'none'
    subtypes["subtype"] = (
        data["subtype"].astype(str).str.strip().str.lower().map(SUBTYPE_MAP).fillna(NO_SUBTYPE)
    )

    duplicated = subtypes["lesion_ID"].duplicated()
    if duplicated.any():
        sys.exit(f"[ERROR] lesion_ID duplicados en el csv de subtipos: {subtypes.loc[duplicated, 'lesion_ID'].tolist()[:10]}")

    return subtypes


def load_tnbc_subsets(csv_doctor: str, csv_cercanos: str) -> tuple:
    """
    Read the doctor and metric subset files and return them as sets of IDs.

    Args:
        csv_doctor (str): path to the doctor TNBC subset (patient IDs).
        csv_cercanos (str): path to the metric subset (lesion IDs).

    Returns:
        tuple: (doctor patients, metric TN lesions, metric BN lesions)
    """
    # El archivo del doctor viene con BOM, de ahí utf-8-sig
    doctor_data = pd.read_csv(csv_doctor, encoding="utf-8-sig")
    doctor      = set(doctor_data.iloc[:, 0].dropna().astype(str).str.strip())

    # El archivo métrico empareja cada lesión TNBC con su benigna más cercana:
    # la primera columna son los TN y la segunda los BN, ambas a nivel de lesión.
    cercanos_data   = pd.read_csv(csv_cercanos)
    tnbc_cercano    = set(cercanos_data.iloc[:, 0].dropna().astype(str).str.strip())
    benigno_cercano = set(cercanos_data.iloc[:, 1].dropna().astype(str).str.strip())

    return doctor, tnbc_cercano, benigno_cercano


def build_subset_table(lesions: pd.DataFrame, csv_subtype: str, csv_doctor: str,
                       csv_cercanos: str) -> pd.DataFrame:
    """
    Add one boolean column per group to the evaluated lesions.

    Args:
        lesions (pd.DataFrame): evaluated lesions, with the columns lesion_ID and label.
        csv_subtype (str): path to the radiomics CSV.
        csv_doctor (str): path to the doctor TNBC subset.
        csv_cercanos (str): path to the metric subset.

    Returns:
        pd.DataFrame: copy of `lesions`, in the same row order, with a boolean
            column for every name in `GROUPS`.
    """
    subtypes = load_subtypes(csv_subtype)
    doctor, tnbc_cercano, benigno_cercano = load_tnbc_subsets(csv_doctor, csv_cercanos)

    table = lesions.merge(subtypes, on="lesion_ID", how="left")

    missing = table.loc[table["subtype"].isna(), "lesion_ID"]
    if not missing.empty:
        sys.exit(f"[ERROR] lesiones sin subtipo en el csv de radiómica: {missing.tolist()[:10]}")

    # Grupos por etiqueta binaria: 0 = benigno, 1 = maligno
    table["benigno"] = table["label"] == 0
    table["maligno"] = table["label"] == 1

    # Grupos por subtipo molecular
    table["luminal_a"]      = table["subtype"] == "luminal_a"
    table["luminal_b"]      = table["subtype"] == "luminal_b"
    table["her2_enriched"]  = table["subtype"] == "her2_enriched"
    table["tnbc"]           = table["subtype"] == "tnbc"

    # El archivo del doctor selecciona pacientes completos, así que se restringe
    # a sus lesiones TNBC para que el grupo no arrastre otras lesiones suyas.
    table["tnbc_doctor"]        = table["patient_ID"].isin(doctor) & table["tnbc"]
    table["tnbc_cercano"]       = table["lesion_ID"].isin(tnbc_cercano)
    table["benigno_cercano"]    = table["lesion_ID"].isin(benigno_cercano)

    # Unión de las dos definiciones del subconjunto TNBC difícil
    table["tnbc_ambos"] = table["tnbc_cercano"] | table["tnbc_doctor"]

    # Se devuelven las columnas de entrada más los grupos, y se descartan las
    # auxiliares patient_ID y subtype que solo hacían falta para construirlos.
    return table[list(lesions.columns) + GROUPS]


def get_crossing_mask(table: pd.DataFrame, crossing: str) -> pd.Series:
    """
    Return the mask that isolates the two groups of a crossing.

    Args:
        table (pd.DataFrame): membership table from `build_subset_table`.
        crossing (str): name of the crossing, a key of `CROSSINGS`.

    Returns:
        pd.Series: boolean mask with the lesions of both groups.
    """
    negative, positive = CROSSINGS[crossing]

    return table[negative] | table[positive]
