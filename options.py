import argparse

def parse_tuple(input_string):
    try:
        return tuple(map(int, input_string.split(',')))  # Convierte los valores a int
    except:
        raise argparse.ArgumentTypeError("Las entradas deben ser números enteros separados por comas.")

def get_options():
    """
    Function to parse command line arguments and return them as a namespace.

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    
    # Default directory paths
    images_dir          = "/mnt/Datos/Master_Camilo/DL/intratumoral-patches"
    csv_data_path       = "/mnt/Datos/Master_Camilo/DL/data"
    csv_subtype         = "/mnt/Datos/Master_Camilo/DL/data/radiomics_features_lesion_sin_calc_dist_lipoma.csv"
    csv_doctor          = "/mnt/Datos/Master_Camilo/DL/data/splits_tnbc_doctor.csv"
    csv_cercanos        = "/mnt/Datos/Master_Camilo/DL/data/TN_cercanos_vs_BN_cercanos.csv"
    result_dir          = "/mnt/Datos/Master_Camilo/DL/results"
    path_model          = "/mnt/Datos/Master_Camilo/DL/weights/RadImageNet_pytorch/ResNet50.pt"  # Path to the weights file if needed #"/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/03-Challenges/01-MAMA-MIA/01-Code/RadImageNet_pytorch/01-Pytorch/ResNet50.pt"

    # Available options
    images_model_choices    = ["ResNet"]
    weights_type_choices    = ["custom", "imagenet"]
    activation_choices      = ["Linear", "ReLU", "Sigmoid", "LeakyReLU", "Tanh", "Gelu"]

    # Create argument parser with description
    parser = argparse.ArgumentParser(description="Código para entrenar y evaluar modelos para el proyecto cesm-synthesis")

    # Add arguments for general configurations
    parser.add_argument("--exp_name",           type=str, default="Test_Code", help="Nombre de experimento")
    parser.add_argument("--images_dir",         type=str, default=images_dir, help="Dirección del directorio de las imágenes")
    parser.add_argument("--csv_data_path",      type=str, default=csv_data_path, help="Ruta del directorio con los csv de particiones (train.csv, test.csv, val.csv)")
    parser.add_argument("--csv_train",          type=str, default=csv_subtype, help="Ruta del archivo csv con los training samples")
    parser.add_argument("--csv_test",           type=str, default=csv_subtype, help="Ruta del archivo csv con los test samples")
    parser.add_argument("--csv_val",            type=str, default=csv_subtype, help="Ruta del archivo csv con los validation samples")
    parser.add_argument("--csv_subtype",        type=str, default=csv_subtype, help="Ruta del archivo csv con los subtipos moleculares de cada lesión")
    parser.add_argument("--csv_doctor",         type=str, default=csv_doctor, help="Ruta del archivo csv con los doctores")
    parser.add_argument("--csv_cercanos",       type=str, default=csv_cercanos, help="Ruta del archivo csv con los pacientes cercanos")
    parser.add_argument("--result_dir",         type=str, default=result_dir, help="Direccion en donde se guardaran los resultados. Default = %(default)s")
    parser.add_argument("--tag_exp",            type=str, nargs='+', default=["Test"], help="Etiquetas de wandb para el experimento")
    parser.add_argument("--train",              help="Entrenamiento?", default=True, action="store_true")
    parser.add_argument("--test",               help="Prueba?", default=False, action="store_true")

    # Configuration of images model
    parser.add_argument("--activation_image_model", type=str, default="ReLU", choices=activation_choices, help="Función de activación interna del modelo de imagen: %(choices)s")
    parser.add_argument("--image_model",            type=str, default="ResNet", choices=images_model_choices, help="Seleccione el modelo generador a utilizar: %(choices)s")
    parser.add_argument("--path_image_model",       type=str, default=path_model, help="Ruta al archivo de pesos del modelo preentrenado")
    parser.add_argument("--weights_type",           type=str, default="custom", choices=weights_type_choices, help="Origen de los pesos del modelo de imagen: %(choices)s")
    parser.add_argument("--num_freeze",             type = int, default=5, help="Congelar la base del modelo?", )
    
    # Configuration of final model
    parser.add_argument("--hidden_layers",              type=int, nargs="+", default=[512, 512], help="Capas ocultas del modelo final")
    parser.add_argument("--output_size",                type=int, default=2, help="Número de clases de salida del modelo final")
    parser.add_argument("--activation",                 type=str, default="LeakyReLU", choices=activation_choices, help="Función de activación del modelo final: %(choices)s")
    parser.add_argument("--dropout",                    type=float, default=0.5, help="Tasa de abandono del modelo final")

    # Add arguments for data loader configuration
    parser.add_argument("--augmentation",   help="Utilizar aumento de datos en el entrenamiento?", default=False, action="store_true")

    # Add arguments for training parameters
    parser.add_argument("--init_epoch",     type=int, default=0, help="Epoca desde donde se inicia el entrenamiento")
    parser.add_argument("--n_epochs",       type=int, default=2, help="Número de epocas de entrenamiento")
    parser.add_argument("--batch_size",     type=int, default=32, help="Numero de lotes por cada paso de entrenamiento")
    parser.add_argument("--lr",             type=float, default=1e-4, help="Taza de aprendizaje del modelo generador (Unet-GAN)")
    parser.add_argument("--b1",             type=float, default=0.5, help="Adam: decaimiento del impulso de primer orden del gradiente")
    parser.add_argument("--b2",             type=float, default=0.999, help="Adam: decaimiento del impulso de primer orden del gradiente")
    
    # Configuración EarlyStopping
    parser.add_argument("--patience_early", type=int, default=20, help="Patience para EarlyStopping")

    # Configuración de Scheduler
    parser.add_argument("--min_lr",         type=float, default=1e-6, help="Taza de aprendizaje mínima para el Scheduler")
    
    # Configuracion perdida
    parser.add_argument("--loss",           type=str, default="BCE", help="Función de perdida: %(choices)s")
    
    # Configuracion de balanceo de clases BCE
    parser.add_argument("--class_balance",  help="Utilizar balanceo de clases?", default=True, action="store_true")
    parser.add_argument("--pos_weight",     type=float, default=0.59, help="Peso para la clase positiva en la función de pérdida BCE")
    parser.add_argument("--neg_weight",     type=float, default=3.22, help="Peso para la clase negativa en la función de pérdida BCE")
    parser.add_argument("--gamma",          type=float, default=2.0, help="Gamma para la función de pérdida focal")

    # Parse command line arguments and return as a namespace
    args = parser.parse_args()

    return args
