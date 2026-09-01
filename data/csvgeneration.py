"""
Generación de las particiones train / test / val del dataset de parches
intratumorales.

El split se realiza **a nivel de paciente** para evitar fuga de datos: todas las
lesiones de un mismo paciente caen siempre en el mismo conjunto. Además se
estratifica simultáneamente por:

    - Estado máximo de malignidad del paciente (si alguna lesión o lateralidad
      es maligna, el paciente completo cuenta como maligno).
    - Subtipo molecular (luminal_a, luminal_b, her2_enriched, tnbc, none).
    - Pertenencia al subconjunto TNBC del doctor (data/splits_tnbc_doctor.csv).
    - Pertenencia al subconjunto TNBC métrico, columna "Paciente TN"
      (data/TN_cercanos_vs_BN_cercanos.csv).

Genera cuatro archivos con las columnas exactas ``lesion_ID, label, ruta, split``:
``dataset_split.csv`` (todos los datos), ``train.csv``, ``test.csv`` y ``val.csv``.

Uso:
    python3 data/csvgeneration.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# Orden de los splits: de mayor a menor fracción. El repartidor lo usa como
# prioridad al cubrir estratos con menos pacientes que splits.
SPLITS = ["train", "test", "val"]

# Normalización de la columna `subtype` del CSV de radiómica.
SUBTYPE_MAP = {
    "luminal a":        "luminal_a",
    "luminal b":        "luminal_b",
    "her2-enriched":    "her2_enriched",
    "triple negative":  "tnbc",
}

# Valor de subtipo para las lesiones sin subtipo molecular asignado.
NO_SUBTYPE = "none"

# Columnas exportadas, en este orden. `ImageDataset.__getitem__` accede a las
# dos primeras posicionalmente (iloc[:, 0] = nombre, iloc[:, 1] = etiqueta).
OUTPUT_COLUMNS = ["lesion_ID", "label", "ruta", "split"]


def get_options():
    """
    Function to parse command line arguments and return them as a namespace.

    Returns:
        argparse.Namespace: parsed command line arguments
    """

    # Default directory paths
    root_dir            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_subtype         = os.path.join(root_dir, "data", "radiomics_features_lesion_sin_calc_dist_lipoma.csv")
    csv_doctor          = os.path.join(root_dir, "data", "splits_tnbc_doctor.csv")
    csv_cercanos        = os.path.join(root_dir, "data", "TN_cercanos_vs_BN_cercanos.csv")
    out_dir             = os.path.join(root_dir, "data")

    parser = argparse.ArgumentParser(description="Genera las particiones train/test/val a nivel de paciente")

    parser.add_argument("--csv_subtype",    type=str, default=csv_subtype, help="Ruta del archivo csv con los subtipos y la clasificación")
    parser.add_argument("--csv_doctor",     type=str, default=csv_doctor, help="Ruta del archivo csv con el subconjunto TNBC del doctor")
    parser.add_argument("--csv_cercanos",   type=str, default=csv_cercanos, help="Ruta del archivo csv con el subconjunto TNBC métrico")
    parser.add_argument("--root_dir",       type=str, default=root_dir, help="Raíz del proyecto, usada para comprobar que las rutas existen")
    parser.add_argument("--images_subdir",  type=str, default="intratumoral-patches", help="Subdirectorio de los parches, prefijo de la columna ruta")
    parser.add_argument("--out_dir",        type=str, default=out_dir, help="Directorio donde se escriben los csv generados")

    parser.add_argument("--train_size",     type=float, default=0.70, help="Fracción de datos para entrenamiento")
    parser.add_argument("--test_size",      type=float, default=0.20, help="Fracción de datos para prueba")
    parser.add_argument("--val_size",       type=float, default=0.10, help="Fracción de datos para validación")
    parser.add_argument("--seed",           type=int,   default=42, help="Semilla aleatoria para reproducibilidad")

    options = parser.parse_args()

    # Validar que las fracciones formen una partición completa
    total = options.train_size + options.test_size + options.val_size
    if not np.isclose(total, 1.0):
        parser.error(f"train_size + test_size + val_size debe sumar 1.0, se obtuvo {total}")

    return options


def load_lesions(csv_subtype: str, images_subdir: str) -> pd.DataFrame:
    """
    Read the source CSV and build the lesion level dataframe.

    Args:
        csv_subtype (str): path to the radiomics CSV.
        images_subdir (str): patches subdirectory, used as prefix of `ruta`.

    Returns:
        pd.DataFrame: one row per lesion with lesion_ID, label, ruta, patient_ID
            and the normalized subtype.
    """
    data = pd.read_csv(csv_subtype)

    lesions = pd.DataFrame()
    lesions["lesion_ID"] = data["lesion_id"].astype(str).str.strip()

    # Etiqueta binaria: 0 = benigno, 1 = maligno
    classification = data["classification_TOMPEI"].astype(str).str.strip().str.lower()
    unexpected = set(classification.unique()) - {"benign", "malignant"}
    if unexpected:
        sys.exit(f"[ERROR] Valores inesperados en classification_TOMPEI: {sorted(unexpected)}")
    lesions["label"] = (classification == "malignant").astype(int)

    # Ruta relativa al parche. Se usa '/' literal para que el formato del csv sea
    # idéntico en cualquier plataforma.
    lesions["ruta"] = images_subdir + "/" + lesions["lesion_ID"] + "/" + lesions["lesion_ID"] + ".png"

    # El ID de paciente es el prefijo del lesion_ID (ej. D1-0001_R_1 -> D1-0001);
    # nunca contiene guion bajo, por eso basta con partir en el primero.
    lesions["patient_ID"] = lesions["lesion_ID"].str.split("_").str[0]

    # Subtipo molecular normalizado; las lesiones sin subtipo quedan como 'none'
    lesions["subtype"] = (
        data["subtype"].astype(str).str.strip().str.lower().map(SUBTYPE_MAP).fillna(NO_SUBTYPE)
    )

    duplicated = lesions["lesion_ID"].duplicated()
    if duplicated.any():
        sys.exit(f"[ERROR] lesion_ID duplicados en el csv origen: {lesions.loc[duplicated, 'lesion_ID'].tolist()[:10]}")

    return lesions


def load_tnbc_subsets(csv_doctor: str, csv_cercanos: str) -> tuple:
    """
    Read the two TNBC subset files and return them as sets of patient IDs.

    Args:
        csv_doctor (str): path to the doctor TNBC subset (already patient IDs).
        csv_cercanos (str): path to the metric TNBC subset (lesion IDs).

    Returns:
        tuple: (doctor patients, metric TN patients)
    """
    # El archivo del doctor viene con BOM, de ahí utf-8-sig
    doctor_data = pd.read_csv(csv_doctor, encoding="utf-8-sig")
    doctor      = set(doctor_data.iloc[:, 0].dropna().astype(str).str.strip())

    # El archivo métrico contiene lesion_ID, se recorta al ID de paciente.
    # Solo se utiliza la columna "Paciente TN"; la columna BN queda fuera.
    cercanos_data   = pd.read_csv(csv_cercanos)
    metric_lesions  = cercanos_data.iloc[:, 0].dropna().astype(str).str.strip()
    metric          = set(metric_lesions.str.split("_").str[0])

    return doctor, metric


def build_patient_table(lesions: pd.DataFrame, doctor: set, metric: set) -> pd.DataFrame:
    """
    Aggregate lesions into a patient level table with the stratification key.

    Args:
        lesions (pd.DataFrame): lesion level dataframe from `load_lesions`.
        doctor (set): patients belonging to the doctor TNBC subset.
        metric (set): patients belonging to the metric TNBC subset.

    Returns:
        pd.DataFrame: one row per patient, indexed by patient_ID.
    """
    grouped = lesions.groupby("patient_ID")

    patients = grouped.agg(
        # Si alguna lesión o lateralidad del paciente es maligna, el paciente
        # entero se cataloga como maligno para la estratificación. Las etiquetas
        # individuales de cada lesión no se modifican.
        is_malignant = ("label", "max"),
        n_lesions    = ("label", "size"),
    )

    # Subtipo del paciente: el único valor no nulo entre sus lesiones
    patients["subtype"] = grouped["subtype"].agg(_patient_subtype)

    patients["in_tnbc_doctor"] = patients.index.isin(doctor).astype(int)
    patients["in_tnbc_metric"] = patients.index.isin(metric).astype(int)

    # Clave de estrato: combina las cuatro variables de estratificación
    patients["stratum"] = (
        patients["is_malignant"].astype(str)
        + "|" + patients["subtype"]
        + "|d" + patients["in_tnbc_doctor"].astype(str)
        + "|t" + patients["in_tnbc_metric"].astype(str)
    )

    return patients


def _patient_subtype(subtypes: pd.Series) -> str:
    """
    Return the single molecular subtype of a patient, or 'none' if it has none.

    Args:
        subtypes (pd.Series): normalized subtypes of the patient lesions.

    Returns:
        str: the patient subtype.
    """
    present = sorted(set(subtypes) - {NO_SUBTYPE})
    return present[0] if present else NO_SUBTYPE


def assign_splits(patients: pd.DataFrame, fractions: dict, seed: int) -> dict:
    """
    Assign every patient to a split with a greedy, lesion weighted apportionment.

    Los estratos se recorren de mayor a menor y, dentro de cada uno, los pacientes
    se barajan y luego se ordenan por número de lesiones descendente: colocar
    primero a los pacientes con más lesiones evita que uno grande caiga al final
    y desbalancee las proporciones. Cada paciente va al split con mayor déficit
    relativo de lesiones, usando el déficit de pacientes como desempate. Los
    contadores son globales entre estratos, así que los errores de redondeo se
    cancelan en lugar de acumularse.

    Args:
        patients (pd.DataFrame): patient level table from `build_patient_table`.
        fractions (dict): target fraction per split.
        seed (int): random seed for reproducibility.

    Returns:
        dict: mapping patient_ID -> split name.
    """
    generator = np.random.default_rng(seed)

    lesion_count  = {split: 0 for split in SPLITS}
    patient_count = {split: 0 for split in SPLITS}
    assignment    = {}

    # Estratos de mayor a menor, para que los grandes fijen la línea base global
    for stratum in patients["stratum"].value_counts().index:
        group = patients[patients["stratum"] == stratum]

        # Barajar y después ordenar por nº de lesiones. El sort estable conserva
        # el barajado en los empates, así el resultado es reproducible e insesgado.
        ids     = group.index.to_numpy()[generator.permutation(len(group))]
        group   = group.loc[ids]
        sizes   = group["n_lesions"].to_numpy()
        order   = np.argsort(-sizes, kind="stable")

        # Cobertura: un estrato con menos pacientes que splits se reparte por
        # prioridad (train primero) en vez de por déficit, para que train nunca
        # se quede sin representación del estrato.
        if len(ids) < len(SPLITS):
            for position, index in enumerate(order):
                split = SPLITS[position]
                assignment[ids[index]] = split
                lesion_count[split]   += int(sizes[index])
                patient_count[split]  += 1
            continue

        for index in order:
            weight        = int(sizes[index])
            total_lesions = sum(lesion_count.values()) + weight
            total_patients = sum(patient_count.values()) + 1

            def deficit(split, total_lesions=total_lesions, total_patients=total_patients):
                fraction = fractions[split]
                return (
                    (fraction * total_lesions - lesion_count[split]) / fraction,
                    (fraction * total_patients - patient_count[split]) / fraction,
                )

            split = max(SPLITS, key=deficit)
            assignment[ids[index]] = split
            lesion_count[split]   += weight
            patient_count[split]  += 1

    return assignment


def export(lesions: pd.DataFrame, assignment: dict, out_dir: str) -> pd.DataFrame:
    """
    Write the general split and the three per split CSV files.

    Args:
        lesions (pd.DataFrame): lesion level dataframe.
        assignment (dict): mapping patient_ID -> split name.
        out_dir (str): directory where the CSV files are written.

    Returns:
        pd.DataFrame: the exported dataframe with the four final columns.
    """
    dataframe = lesions.copy()
    dataframe["split"] = dataframe["patient_ID"].map(assignment)
    dataframe = dataframe.sort_values("lesion_ID").reset_index(drop=True)
    dataframe = dataframe[OUTPUT_COLUMNS]

    os.makedirs(out_dir, exist_ok=True)

    general_path = os.path.join(out_dir, "dataset_split.csv")
    dataframe.to_csv(general_path, index=False)
    print(f"  {general_path}  ({len(dataframe)} filas)")

    for split in SPLITS:
        subset = dataframe[dataframe["split"] == split]
        path   = os.path.join(out_dir, f"{split}.csv")
        subset.to_csv(path, index=False)
        print(f"  {path}  ({len(subset)} filas)")

    return dataframe


def report(dataframe: pd.DataFrame, lesions: pd.DataFrame, patients: pd.DataFrame,
           assignment: dict, fractions: dict, root_dir: str) -> None:
    """
    Print the distribution tables and run the sanity checks on the split.

    Args:
        dataframe (pd.DataFrame): exported lesion level dataframe.
        lesions (pd.DataFrame): lesion level dataframe, for the per lesion subtype.
        patients (pd.DataFrame): patient level table.
        assignment (dict): mapping patient_ID -> split name.
        fractions (dict): target fraction per split.
        root_dir (str): project root, used to resolve the `ruta` column.

    Returns:
        None
    """
    patients = patients.copy()
    patients["split"] = patients.index.map(assignment)

    # El subtipo se toma de la propia lesión, no del paciente: en los pacientes
    # con lesiones benignas y malignas a la vez, las benignas no tienen subtipo.
    lesions = dataframe.merge(
        lesions[["lesion_ID", "patient_ID", "subtype"]], on="lesion_ID", how="left"
    )

    patient_counts = patients["split"].value_counts()
    lesion_counts  = lesions["split"].value_counts()

    print("\n--- TAMAÑOS ---")
    for split in SPLITS:
        n_patients = patient_counts.get(split, 0)
        n_lesions  = lesion_counts.get(split, 0)
        print(f"  {split:5s} pacientes {n_patients:4d} ({n_patients / len(patients) * 100:5.2f}%)"
              f" | lesiones {n_lesions:4d} ({n_lesions / len(lesions) * 100:5.2f}%)"
              f"  [meta {fractions[split] * 100:.0f}%]")

    print(f"\n--- BALANCE DE CLASE (maligno) --- global {lesions['label'].mean() * 100:.2f}%")
    for split in SPLITS:
        subset = lesions[lesions["split"] == split]
        print(f"  {split:5s} mal={int(subset['label'].sum()):4d}"
              f" ben={int((1 - subset['label']).sum()):4d}"
              f" -> {subset['label'].mean() * 100:5.2f}%")

    print("\n--- SUBTIPO (% de lesiones) ---")
    overall = lesions["subtype"].value_counts(normalize=True) * 100
    print(f"  {'':8s}", "  ".join(f"{name:>13s}" for name in overall.index))
    print(f"  {'global':8s}", "  ".join(f"{value:13.2f}" for value in overall.values))
    for split in SPLITS:
        distribution = lesions[lesions["split"] == split]["subtype"].value_counts(normalize=True) * 100
        print(f"  {split:8s}", "  ".join(f"{distribution.get(name, 0):13.2f}" for name in overall.index))

    print("\n--- SUBCONJUNTOS TNBC (pacientes) ---")
    for split in SPLITS:
        subset = patients[patients["split"] == split]
        print(f"  {split:5s} doctor={int(subset['in_tnbc_doctor'].sum()):2d}/{int(patients['in_tnbc_doctor'].sum())}"
              f"   metric_TN={int(subset['in_tnbc_metric'].sum()):2d}/{int(patients['in_tnbc_metric'].sum())}")

    print("\n--- PACIENTES POR ESTRATO ---")
    table = pd.crosstab(patients["stratum"], patients["split"]).reindex(columns=SPLITS, fill_value=0)
    print(table.to_string())

    # Verificaciones
    print("\n--- VERIFICACIONES ---")

    leaks = lesions.groupby("patient_ID")["split"].nunique()
    n_leaks = int((leaks > 1).sum())
    print(f"  fuga de pacientes entre splits: {n_leaks}")
    assert n_leaks == 0, f"fuga de datos: {leaks[leaks > 1].index.tolist()}"

    total = int(lesion_counts.sum())
    print(f"  lesiones repartidas: {total} / {len(lesions)}")
    assert total == len(lesions), "la suma de los splits no cubre todas las lesiones"

    missing = [path for path in lesions["ruta"] if not os.path.exists(os.path.join(root_dir, path))]
    print(f"  rutas .png existentes: {len(lesions) - len(missing)} / {len(lesions)}")
    assert not missing, f"rutas inexistentes: {missing[:10]}"

    print("  OK")


def main():
    """
    Build the patient level stratified split and export the four CSV files.
    """
    options = get_options()

    fractions = {
        "train": options.train_size,
        "test":  options.test_size,
        "val":   options.val_size,
    }

    lesions          = load_lesions(options.csv_subtype, options.images_subdir)
    doctor, metric   = load_tnbc_subsets(options.csv_doctor, options.csv_cercanos)
    patients         = build_patient_table(lesions, doctor, metric)

    print(f"Lesiones: {len(lesions)} | Pacientes: {len(patients)} | Estratos: {patients['stratum'].nunique()}")
    print("\n--- ARCHIVOS GENERADOS ---")

    assignment = assign_splits(patients, fractions, options.seed)
    dataframe  = export(lesions, assignment, options.out_dir)

    report(dataframe, lesions, patients, assignment, fractions, options.root_dir)


if __name__ == "__main__":

    main()
