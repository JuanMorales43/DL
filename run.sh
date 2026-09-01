path_image_model_resnet="/mnt/Datos/Master_Camilo/DL/weights/RadImageNet_pytorch/ResNet50.pt"

result_dir="/mnt/Datos/Master_Camilo/DL/results"

# python3 main.py --exp_name "Classification Images_HL1024-512_lr1e-4_dr0.5_unfreezelayer4" \
#                 --images_dir "/mnt/Datos/Master_Camilo/DL/intratumoral-patches" \
#                 --csv_data_path "/mnt/Datos/Master_Camilo/DL/data" \
#                 --csv_subtype "/mnt/Datos/Master_Camilo/DL/data/radiomics_features_lesion_sin_calc_dist_lipoma.csv" \
#                 --csv_doctor "/mnt/Datos/Master_Camilo/DL/data/splits_tnbc_doctor.csv" \
#                 --csv_cercanos "/mnt/Datos/Master_Camilo/DL/data/TN_cercanos_vs_BN_cercanos.csv" \
#                 --result_dir $result_dir \
#                 --tag_exp "Classification Images" \
#                 --activation_image_model "LeakyReLU" \
#                 --image_model "ResNet" \
#                 --path_image_model $path_image_model_resnet \
#                 --num_freeze 129 \
#                 --hidden_layers 1024 512\
#                 --output_size 2 \
#                 --activation "LeakyReLU" \
#                 --dropout 0.5 \
#                 --augmentation \
#                 --n_epochs 200 \
#                 --batch_size 64 \
#                 --lr 1e-4 \
#                 --patience_early 100 \
#                 --min_lr 1e-6 \
#                 --loss "BCE" \
#                 --train \
#                 --class_balance


python3 main.py --exp_name "Classification Images_HL512-512_lr1e-4_dr0.5_unfreezelayer4_negweight4.0" \
                --images_dir "/mnt/Datos/Master_Camilo/DL/intratumoral-patches" \
                --csv_data_path "/mnt/Datos/Master_Camilo/DL/data" \
                --csv_subtype "/mnt/Datos/Master_Camilo/DL/data/radiomics_features_lesion_sin_calc_dist_lipoma.csv" \
                --csv_doctor "/mnt/Datos/Master_Camilo/DL/data/splits_tnbc_doctor.csv" \
                --csv_cercanos "/mnt/Datos/Master_Camilo/DL/data/TN_cercanos_vs_BN_cercanos.csv" \
                --result_dir $result_dir \
                --tag_exp "Classification Images" \
                --activation_image_model "LeakyReLU" \
                --image_model "ResNet" \
                --path_image_model $path_image_model_resnet \
                --num_freeze 129 \
                --hidden_layers 512 512\
                --output_size 2 \
                --activation "LeakyReLU" \
                --dropout 0.5 \
                --augmentation \
                --n_epochs 200 \
                --batch_size 64 \
                --lr 1e-4 \
                --patience_early 100 \
                --min_lr 1e-6 \
                --loss "BCE" \
                --train \
                --neg_weight 4.0 \
                --class_balance


python3 main.py --exp_name "Classification Images_HL512-512_lr1e-4_dr0.5_unfreezelayer4_negweight5" \
                --images_dir "/mnt/Datos/Master_Camilo/DL/intratumoral-patches" \
                --csv_data_path "/mnt/Datos/Master_Camilo/DL/data" \
                --csv_subtype "/mnt/Datos/Master_Camilo/DL/data/radiomics_features_lesion_sin_calc_dist_lipoma.csv" \
                --csv_doctor "/mnt/Datos/Master_Camilo/DL/data/splits_tnbc_doctor.csv" \
                --csv_cercanos "/mnt/Datos/Master_Camilo/DL/data/TN_cercanos_vs_BN_cercanos.csv" \
                --result_dir $result_dir \
                --tag_exp "Classification Images" \
                --activation_image_model "LeakyReLU" \
                --image_model "ResNet" \
                --path_image_model $path_image_model_resnet \
                --num_freeze 129 \
                --hidden_layers 512 512\
                --output_size 2 \
                --activation "LeakyReLU" \
                --dropout 0.5 \
                --augmentation \
                --n_epochs 200 \
                --batch_size 64 \
                --lr 1e-4 \
                --patience_early 100 \
                --min_lr 1e-6 \
                --loss "BCE" \
                --train \
                --neg_weight 5.0 \
                --class_balance


python3 main.py --exp_name "Classification Images_HL512-512_lr1e-4_dr0.5_unfreezelayer4_negweight5" \
                --images_dir "/mnt/Datos/Master_Camilo/DL/intratumoral-patches" \
                --csv_data_path "/mnt/Datos/Master_Camilo/DL/data" \
                --csv_subtype "/mnt/Datos/Master_Camilo/DL/data/radiomics_features_lesion_sin_calc_dist_lipoma.csv" \
                --csv_doctor "/mnt/Datos/Master_Camilo/DL/data/splits_tnbc_doctor.csv" \
                --csv_cercanos "/mnt/Datos/Master_Camilo/DL/data/TN_cercanos_vs_BN_cercanos.csv" \
                --result_dir $result_dir \
                --tag_exp "Classification Images" \
                --activation_image_model "LeakyReLU" \
                --image_model "ResNet" \
                --path_image_model $path_image_model_resnet \
                --num_freeze 129 \
                --hidden_layers 1024 512 256\
                --output_size 2 \
                --activation "LeakyReLU" \
                --dropout 0.5 \
                --augmentation \
                --n_epochs 200 \
                --batch_size 64 \
                --lr 1e-4 \
                --patience_early 100 \
                --min_lr 1e-6 \
                --loss "BCE" \
                --train \
                --neg_weight 5.0 \
                --class_balance