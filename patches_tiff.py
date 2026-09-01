"""
Generación de parches intratumorales 224x224 RGB en TIFF (CMMD-MSC).

Replica la estructura de subdirectorios de `intratumoral-patches/` en
`intratumoral-patches-tiff/`, pero leyendo los datos desde los .tiff originales
de `CMMD_MSC/[PID]/img/` y sus máscaras de `CMMD_MSC/[PID]/seg/mask/[LAT]/[N]/`.

La carpeta de referencia se usa ÚNICAMENTE como plantilla del árbol de directorios:
no se abre ninguna de sus imágenes durante la generación.

Depende de PIL, numpy, OpenCV y la stdlib. El redimensionado de los parches se
hace con `cv2.resize(..., INTER_AREA)`, igual que en el pipeline original.
"""

import argparse
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PATCH_SIZE = 224

NOMBRE_REGEX = re.compile(r'^([A-Z0-9\-]+)_([LR])_(\d+)$')

REF_DEFAULT = '/mnt/Datos/Master_Camilo/DL/intratumoral-patches'
IN_DEFAULT = '/mnt/Datos/BackupCamilo/08-MasterCamilo/05-CMMD_Depurado/CMMD_MSC'
OUT_DEFAULT = '/mnt/Datos/Master_Camilo/DL/intratumoral-patches-tiff'


# --------------------------------------------------------------------------- #
# Estructura de directorios (leída de la carpeta de referencia)
# --------------------------------------------------------------------------- #

def listar_nombres_referencia(raiz_ref):
    """Nombres de las subcarpetas de la referencia. No abre ninguna imagen."""
    raiz_ref = Path(raiz_ref)
    if not raiz_ref.is_dir():
        raise SystemExit(f'No existe la carpeta de referencia: {raiz_ref}')
    return sorted(d.name for d in raiz_ref.iterdir() if d.is_dir())


def parsear_nombre(nombre):
    """'D1-0001_R_1' -> ('D1-0001', 'R', 1). None si no encaja."""
    m = NOMBRE_REGEX.match(nombre)
    if not m:
        return None
    pid, lat, n = m.groups()
    return pid, lat, int(n)


def resolver_rutas(pid, lat, n, raiz_in):
    """Rutas al .tiff de la mamografía MLO y a su máscara numerada."""
    raiz_in = Path(raiz_in)
    img = raiz_in / pid / 'img' / f'{pid}_{lat}_MLO.tiff'
    mask = raiz_in / pid / 'seg' / 'mask' / lat / str(n) / f'{pid}_{lat}_MLO_mask_{n}.tiff'
    return img, mask


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #

def cargar_imagen(path):
    """Abre el .tiff y devuelve np.uint8 2D. `convert('L')` cubre el caso I;16."""
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('L')
    return np.asarray(img, dtype=np.uint8)


def cargar_mascara(path):
    """Abre la máscara y devuelve np.bool_ 2D, con la defensa > 127 del original."""
    m = Image.open(path)
    if m.mode != 'L':
        m = m.convert('L')
    return np.asarray(m) > 127


def bbox_centroide(mascara):
    """(x0, y0, x1, y1, cx, cy, area) o None si la máscara está vacía."""
    ys, xs = np.where(mascara)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cx = int(round(xs.mean()))
    cy = int(round(ys.mean()))
    return x0, y0, x1, y1, cx, cy, int(ys.size)


# --------------------------------------------------------------------------- #
# Extracción del parche
# --------------------------------------------------------------------------- #

def _caso_a(imagen, mascara, cx, cy):
    """bbox <= 224x224: ventana 224x224 centrada en el centroide, con clamping."""
    h, w = imagen.shape
    x0 = max(0, min(cx - PATCH_SIZE // 2, w - PATCH_SIZE))
    y0 = max(0, min(cy - PATCH_SIZE // 2, h - PATCH_SIZE))
    img_crop = imagen[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
    mask_crop = mascara[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
    return (img_crop * mask_crop).astype(np.uint8)


def _caso_b(imagen, mascara, bbox):
    """bbox > 224x224: recorte exacto del bbox, enmascarado y resize a 224x224."""
    x0, y0, x1, y1 = bbox
    enmascarado = (imagen[y0:y1, x0:x1] * mascara[y0:y1, x0:x1]).astype(np.uint8)
    return cv2.resize(enmascarado, (PATCH_SIZE, PATCH_SIZE),
                      interpolation=cv2.INTER_AREA)


def extraer_parche(imagen, mascara, bbox_info):
    """Devuelve (parche_gray_uint8, metodo_str)."""
    x0, y0, x1, y1, cx, cy, _ = bbox_info
    w, h = x1 - x0, y1 - y0
    if w <= PATCH_SIZE and h <= PATCH_SIZE:
        return _caso_a(imagen, mascara, cx, cy), 'A_centrado'
    parche = _caso_b(imagen, mascara, (x0, y0, x1, y1))
    return parche, 'B_resize'


def gray_a_rgb(parche):
    """Stack 3x -> (224, 224, 3) uint8."""
    return np.stack([parche, parche, parche], axis=-1)


def escribir_parche(rgb, dir_salida, nombre, comprimir=True):
    """Crea {nombre}/ y escribe {nombre}.tiff dentro."""
    subdir = Path(dir_salida) / nombre
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / f'{nombre}.tiff'
    im = Image.fromarray(rgb)
    if comprimir:
        im.save(out_path, format='TIFF', compression='tiff_lzw')
    else:
        im.save(out_path, format='TIFF')
    return out_path


# --------------------------------------------------------------------------- #
# Log
# --------------------------------------------------------------------------- #

def escribir_log(ruta_log, header, resumen, entradas_ok, entradas_err):
    with open(ruta_log, 'w') as f:
        f.write(header)
        f.write('\nRESUMEN\n-------\n')
        for k, v in resumen.items():
            f.write(f'{k:<45} {v}\n')
        f.write('\nDETALLE POR LESION\n------------------\n')
        for e in entradas_ok:
            f.write(e + '\n')
        f.write('\nERRORES\n-------\n')
        if not entradas_err:
            f.write('(ninguno)\n')
        for e in entradas_err:
            f.write(e + '\n')
        f.write('=' * 60 + '\n')


# --------------------------------------------------------------------------- #
# Verificación contra los PNG de referencia
# --------------------------------------------------------------------------- #

def verificar(raiz_out, raiz_ref, nombres):
    """Compara cada .tiff generado con su PNG de referencia, píxel a píxel."""
    raiz_out, raiz_ref = Path(raiz_out), Path(raiz_ref)
    exactos, distintos, ausentes = 0, [], []
    for i, nombre in enumerate(nombres, 1):
        tif = raiz_out / nombre / f'{nombre}.tiff'
        png = raiz_ref / nombre / f'{nombre}.png'
        if not tif.exists() or not png.exists():
            ausentes.append(nombre)
            continue
        a = np.array(Image.open(tif))
        b = np.array(Image.open(png))
        if a.shape == b.shape and np.array_equal(a, b):
            exactos += 1
        else:
            d = int(np.abs(a.astype(int) - b.astype(int)).max()) if a.shape == b.shape else -1
            distintos.append((nombre, d))
        if i % 200 == 0:
            print(f'      verificadas {i}/{len(nombres)}')

    print('\n=== VERIFICACION vs PNG de referencia ===')
    print(f'  Exactos (bit a bit):  {exactos}/{len(nombres)}')
    print(f'  Distintos:            {len(distintos)}')
    print(f'  Ausentes:             {len(ausentes)}')
    for nombre, d in distintos[:20]:
        print(f'    - {nombre}  maxdiff={d}')
    if ausentes[:10]:
        print(f'    ausentes: {ausentes[:10]}')
    return not distintos and not ausentes


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description='Genera parches intratumorales 224x224 RGB en TIFF desde los .tiff originales.')
    parser.add_argument('--raiz-entrada', default=IN_DEFAULT)
    parser.add_argument('--raiz-salida', default=OUT_DEFAULT)
    parser.add_argument('--raiz-referencia', default=REF_DEFAULT,
                        help='Solo se usa para replicar el árbol de subdirectorios.')
    parser.add_argument('--limit', type=int, default=None, help='Procesar solo N lesiones (debug)')
    parser.add_argument('--no-overwrite', action='store_true')
    parser.add_argument('--sin-comprimir', action='store_true', help='TIFF sin compresión LZW')
    parser.add_argument('--verificar', action='store_true',
                        help='Tras generar, compara cada .tiff con su PNG de referencia')
    args = parser.parse_args()

    raiz_in = Path(args.raiz_entrada)
    raiz_out = Path(args.raiz_salida)
    raiz_ref = Path(args.raiz_referencia)
    raiz_out.mkdir(parents=True, exist_ok=True)
    ruta_log = raiz_out / 'log_procesamiento.txt'

    t0 = time.time()
    print(f'[1/3] Leyendo estructura de {raiz_ref}...')
    nombres = listar_nombres_referencia(raiz_ref)
    print(f'      -> {len(nombres)} subdirectorios en la referencia')

    if args.limit:
        nombres = nombres[:args.limit]
        print(f'      -> limitado a {len(nombres)} lesiones para debug')

    # Agrupar por (pid, lat) para leer cada mamografía una sola vez.
    print('[2/3] Agrupando lesiones por (paciente, lateralidad)...')
    grupos = defaultdict(list)
    sin_parsear = []
    for nombre in nombres:
        meta = parsear_nombre(nombre)
        if meta is None:
            sin_parsear.append(nombre)
            continue
        pid, lat, n = meta
        grupos[(pid, lat)].append((n, nombre))
    for k in grupos:
        grupos[k].sort(key=lambda x: x[0])
    print(f'      -> {len(grupos)} grupos paciente+lateralidad')

    print('[3/3] Procesando lesiones...')
    entradas_ok, entradas_err = [], []
    cont = {'caso_a': 0, 'caso_b': 0, 'ok': 0, 'err': 0}

    for nombre in sin_parsear:
        entradas_err.append(f'[ERR]  {nombre}  motivo=nombre de carpeta no reconocido')
        cont['err'] += 1

    hechas = 0
    total = sum(len(v) for v in grupos.values())
    for (pid, lat), lista in sorted(grupos.items()):
        ruta_img, _ = resolver_rutas(pid, lat, 0, raiz_in)
        imagen = None
        err_img = None
        if not ruta_img.exists():
            err_img = f'imagen MLO ausente: {ruta_img}'
        else:
            try:
                imagen = cargar_imagen(ruta_img)
            except Exception as e:
                err_img = f'fallo cargar imagen: {type(e).__name__}: {e}'

        for n, nombre in lista:
            hechas += 1
            if hechas % 100 == 0:
                print(f'      {hechas}/{total} lesiones...')

            if err_img is not None:
                entradas_err.append(f'[ERR]  {nombre}  motivo={err_img}')
                cont['err'] += 1
                continue

            try:
                if args.no_overwrite and (raiz_out / nombre / f'{nombre}.tiff').exists():
                    continue

                _, ruta_mask = resolver_rutas(pid, lat, n, raiz_in)
                if not ruta_mask.exists():
                    entradas_err.append(f'[ERR]  {nombre}  motivo=máscara ausente: {ruta_mask}')
                    cont['err'] += 1
                    continue

                mascara = cargar_mascara(ruta_mask)
                if mascara.shape != imagen.shape:
                    entradas_err.append(
                        f'[ERR]  {nombre}  motivo=shape mismatch '
                        f'img={imagen.shape} mask={mascara.shape}')
                    cont['err'] += 1
                    continue

                bbox_info = bbox_centroide(mascara)
                if bbox_info is None:
                    entradas_err.append(f'[ERR]  {nombre}  motivo=máscara vacía (0 píxeles blancos)')
                    cont['err'] += 1
                    continue

                x0, y0, x1, y1, cx, cy, area = bbox_info
                w, h = x1 - x0, y1 - y0
                parche, metodo = extraer_parche(imagen, mascara, bbox_info)
                out_path = escribir_parche(gray_a_rgb(parche), raiz_out, nombre,
                                           comprimir=not args.sin_comprimir)
                entradas_ok.append(
                    f'[OK]   {nombre:<18} bbox={w}x{h:<6} area_blanca={area:<8}px  '
                    f'metodo={metodo:<20} -> {out_path}')
                cont['ok'] += 1
                if metodo == 'A_centrado':
                    cont['caso_a'] += 1
                else:
                    cont['caso_b'] += 1
            except Exception as e:
                tb = traceback.format_exception_only(type(e), e)[-1].strip()
                entradas_err.append(f'[ERR]  {nombre}  motivo={tb}')
                cont['err'] += 1

    elapsed = time.time() - t0

    header = (
        '=' * 60 + '\n'
        'PIPELINE PARCHES INTRATUMORALES CMMD-MSC -> TIFF\n'
        f'Ejecutado: {time.strftime("%Y-%m-%d %H:%M:%S")}\n'
        f'Raíz entrada:     {raiz_in}\n'
        f'Raíz salida:      {raiz_out}\n'
        f'Árbol replicado de: {raiz_ref} (solo nombres de carpeta)\n'
        f'Parámetros: tamaño={PATCH_SIZE}, enmascarado=SI, margen=0%, '
        f'modo_rgb=stack-3x, numeracion=N_original, formato=TIFF'
        f'{"" if args.sin_comprimir else " (LZW)"}\n'
        + '=' * 60 + '\n'
    )
    resumen = {
        'Grupos paciente+lateralidad procesados:': len(grupos),
        'Lesiones totales encontradas:': cont['ok'] + cont['err'],
        'Parches generados con éxito:': cont['ok'],
        'Caso A (bbox <= 224, centrado):': cont['caso_a'],
        'Caso B (bbox > 224, resize exacto):': cont['caso_b'],
        'Errores:': cont['err'],
        'Tiempo total:': f'{int(elapsed // 60)} min {elapsed % 60:.1f} s',
    }
    escribir_log(ruta_log, header, resumen, entradas_ok, entradas_err)

    print('\n=== RESUMEN ===')
    for k, v in resumen.items():
        print(f'  {k:<45} {v}')
    print(f'Log guardado en: {ruta_log}')

    if args.verificar:
        ok = verificar(raiz_out, raiz_ref, nombres)
        return 0 if ok else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
