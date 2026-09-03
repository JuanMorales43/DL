"""
Unificación de las métricas del test global y las de los cruces en un solo csv.

`train.py` deja los resultados del test repartidos en dos archivos con formatos
distintos: Results/test_summary.csv, en formato largo y con las métricas
prefijadas por 'Test_', y Results/Subsets/subset_metrics.csv, en formato ancho y
con una fila por cruce. Ambos salen de `Metrics.get_metrics`, así que comparten
el mismo juego de métricas y basta con poner el resumen global como una fila más.

El script recorre los experimentos de result_dir, escribe un combined_metrics.csv
dentro del Results de cada uno y, además, un csv único con todos los
experimentos apilados para poder compararlos.

Uso:
    python combine_metrics.py                     # todos los experimentos
    python combine_metrics.py --exp "<nombre>"    # solo uno
"""

import argparse
import os
import sys

import pandas as pd


# Nombre de la fila que ocupa el resumen global dentro del csv combinado.
GLOBAL_ROW = "Global"

# Columnas de contexto que preceden a las métricas, en orden.
CONTEXT_COLUMNS = ["Experiment", "Crossing", "n", "n_benigno", "n_maligno", "Summary_format"]

# Prefijo con el que `train.py` nombra las métricas del test global.
TEST_PREFIX = "Test_"

# Columna de valor de test_summary.csv. Los experimentos anteriores al pipeline
# actual guardan 'mean' y 'std' en lugar de un único 'Value', así que se acepta
# la primera de estas que aparezca.
VALUE_COLUMNS = ["Value", "mean"]


def load_summary(dir_results: str) -> pd.DataFrame:
    """
    Read test_summary.csv and return it as a single wide row.

    Args:
        dir_results (str): Results directory of one experiment.

    Returns:
        pd.DataFrame: one row with a column per metric, without the 'Test_'
            prefix, or an empty frame if the file does not exist or trae un
            formato que no se reconoce.
    """
    path = os.path.join(dir_results, "test_summary.csv")

    if not os.path.isfile(path):
        return pd.DataFrame()

    summary = pd.read_csv(path)
    column  = next((name for name in VALUE_COLUMNS if name in summary.columns), None)

    if column is None:
        return pd.DataFrame()

    # El archivo es largo (Metric, Value); se gira a una fila y se le quita el
    # prefijo para que las columnas coincidan con las de subset_metrics.csv.
    row = dict(zip(summary["Metric"].astype(str).str.removeprefix(TEST_PREFIX), summary[column]))

    return pd.DataFrame([{"Crossing": GLOBAL_ROW, "Summary_format": column, **row}])


def load_counts(dir_results: str) -> dict:
    """
    Count the test lesions of each class from test_predictions.csv.

    `test_summary.csv` no guarda el tamaño del test, así que se deduce de las
    predicciones para que la fila global traiga las mismas columnas de conteo
    que las de los cruces.

    Args:
        dir_results (str): Results directory of one experiment.

    Returns:
        dict: n, n_benigno and n_maligno, vacío si no hay predicciones.
    """
    path = os.path.join(dir_results, "test_predictions.csv")

    if not os.path.isfile(path):
        return {}

    predictions = pd.read_csv(path)

    return {
        "n"         : len(predictions),
        "n_benigno" : int((predictions["label"] == 0).sum()),
        "n_maligno" : int((predictions["label"] == 1).sum()),
    }


def load_subsets(dir_results: str) -> pd.DataFrame:
    """
    Read the per crossing metrics of one experiment.

    Args:
        dir_results (str): Results directory of one experiment.

    Returns:
        pd.DataFrame: contenido de Subsets/subset_metrics.csv, vacío si el
            experimento es anterior a la evaluación por subconjuntos.
    """
    path = os.path.join(dir_results, "Subsets", "subset_metrics.csv")

    return pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()


def combine_experiment(dir_experiment: str) -> pd.DataFrame:
    """
    Combine the global summary and the crossings of one experiment.

    Args:
        dir_experiment (str): directory of the experiment, el que contiene Results.

    Returns:
        pd.DataFrame: fila global seguida de una fila por cruce, con las columnas
            de contexto delante. Vacío si el experimento no tiene resumen.
    """
    name        = os.path.basename(dir_experiment.rstrip(os.sep))
    dir_results = os.path.join(dir_experiment, "Results")

    summary = load_summary(dir_results)

    # Sin test_summary.csv el experimento no llegó a evaluarse; se descarta.
    if summary.empty:
        return pd.DataFrame()

    for column, value in load_counts(dir_results).items():
        summary[column] = value

    combined = pd.concat([summary, load_subsets(dir_results)], ignore_index=True)
    combined.insert(0, "Experiment", name)

    # Las columnas de contexto van delante y las métricas detrás, en el orden en
    # que las generó `Metrics.get_metrics`.
    context = [column for column in CONTEXT_COLUMNS if column in combined.columns]
    metrics = [column for column in combined.columns if column not in context]

    return combined[context + metrics]


def main():

    parser = argparse.ArgumentParser(description="Combina test_summary.csv y subset_metrics.csv en un único csv")
    parser.add_argument("--result_dir", type=str, default="/mnt/Datos/Master_Camilo/DL/results", help="Directorio con los experimentos")
    parser.add_argument("--exp",        type=str, default=None, help="Nombre del experimento a combinar; por defecto se combinan todos")
    parser.add_argument("--out",        type=str, default=None, help="Ruta del csv con todos los experimentos apilados")
    options = parser.parse_args()

    if not os.path.isdir(options.result_dir):
        sys.exit(f"[ERROR] no existe el directorio de resultados: {options.result_dir}")

    if options.exp:
        names = [options.exp]
    else:
        names = sorted(entry for entry in os.listdir(options.result_dir)
                       if os.path.isdir(os.path.join(options.result_dir, entry)))

    tables = []

    for name in names:

        dir_experiment  = os.path.join(options.result_dir, name)
        combined        = combine_experiment(dir_experiment)

        if combined.empty:
            print(f"⚠ {name}: sin test_summary.csv, se omite")
            continue

        path_combined = os.path.join(dir_experiment, "Results", "combined_metrics.csv")
        combined.to_csv(path_combined, index=False)

        crossings = len(combined) - 1
        legacy    = "  (test_summary en formato antiguo mean/std)" if combined.loc[0, "Summary_format"] == "mean" else ""
        print(f"✅ {name}: 1 fila global + {crossings} cruces -> {path_combined}{legacy}")

        tables.append(combined)

    if not tables:
        sys.exit("[ERROR] ningún experimento evaluado en el directorio de resultados")

    path_all = options.out or os.path.join(options.result_dir, "all_experiments_metrics.csv")
    pd.concat(tables, ignore_index=True).to_csv(path_all, index=False)

    print(f"\n✅ {len(tables)} experimentos apilados en {path_all}")


if __name__ == "__main__":
    main()
