path_image_model_resnet="/mnt/Datos/Master_Camilo/DL/weights/RadImageNet_pytorch/ResNet50.pt"
path_best_model_image="/mnt/Datos/Master_Camilo/DL/weights/RadImageNet_pytorch/Best_Model_image.pth"

result_dir="/mnt/Datos/Master_Camilo/DL/results"

# python3 main.py --exp_name "Classification Images_HL1024-1024-512-512_lr1e-4_dr0.5_unfreezelayer4_negweight3.22_posweight0.59_tiff" \
#                 --images_dir "/mnt/Datos/Master_Camilo/DL/intratumoral-patches-tiff" \
#                 --csv_data_path "/mnt/Datos/Master_Camilo/DL/data" \
#                 --csv_subtype "/mnt/Datos/Master_Camilo/DL/data/radiomics_features_lesion_sin_calc_dist_lipoma.csv" \
#                 --csv_doctor "/mnt/Datos/Master_Camilo/DL/data/splits_tnbc_doctor.csv" \
#                 --csv_cercanos "/mnt/Datos/Master_Camilo/DL/data/TN_cercanos_vs_BN_cercanos.csv" \
#                 --result_dir $result_dir \
#                 --tag_exp "Classification Images" \
#                 --activation_image_model "LeakyReLU" \
#                 --image_model "ResNet" \
#                 --path_image_model $path_image_model_resnet \
#                 --num_freeze 159 \
#                 --hidden_layers 1024 1024 512 512 \
#                 --output_size 2 \
#                 --activation "LeakyReLU" \
#                 --dropout 0.5 \
#                 --augmentation \
#                 --n_epochs 200 \
#                 --batch_size 64 \
#                 --lr 5e-4 \
#                 --patience_early 200 \
#                 --min_lr 1e-6 \
#                 --loss "BCE" \
#                 --train \
#                 --neg_weight 3.22 \
#                 --pos_weight 0.59 \
#                 --class_balance

# python3 main.py --exp_name "Classification Images_HL2048-2048-1024-1024-512-512_lr1e-4_dr0.5_unfreezelayer4_negweight3.22_posweight0.59_tiff" \
#                 --images_dir "/mnt/Datos/Master_Camilo/DL/intratumoral-patches-tiff" \
#                 --csv_data_path "/mnt/Datos/Master_Camilo/DL/data" \
#                 --csv_subtype "/mnt/Datos/Master_Camilo/DL/data/radiomics_features_lesion_sin_calc_dist_lipoma.csv" \
#                 --csv_doctor "/mnt/Datos/Master_Camilo/DL/data/splits_tnbc_doctor.csv" \
#                 --csv_cercanos "/mnt/Datos/Master_Camilo/DL/data/TN_cercanos_vs_BN_cercanos.csv" \
#                 --result_dir $result_dir \
#                 --tag_exp "Classification Images" \
#                 --activation_image_model "LeakyReLU" \
#                 --image_model "ResNet" \
#                 --path_image_model $path_image_model_resnet \
#                 --num_freeze 159 \
#                 --hidden_layers 2048 2048 1024 1024 512 512 \
#                 --output_size 2 \
#                 --activation "LeakyReLU" \
#                 --dropout 0.5 \
#                 --augmentation \
#                 --n_epochs 200 \
#                 --batch_size 64 \
#                 --lr 5e-4 \
#                 --patience_early 200 \
#                 --min_lr 1e-6 \
#                 --loss "BCE" \
#                 --train \
#                 --neg_weight 3.22 \
#                 --pos_weight 0.59 \
#                 --class_balance

python3 main.py --exp_name "Classification Images_HL2048-2048-1024-1024-512-512_lr1e-4_dr0.5_unfreezelayer4_negweight3.22_posweight0.59_tiff_pseudotrain" \
                --images_dir "/mnt/Datos/Master_Camilo/DL/intratumoral-patches-tiff" \
                --csv_data_path "/mnt/Datos/Master_Camilo/01-ML/data/benigno_vs_maligno_balanceado_pseudotrain_mahalanobis_voto/dl" \
                --csv_subtype "/mnt/Datos/Master_Camilo/DL/data/radiomics_features_lesion_sin_calc_dist_lipoma.csv" \
                --csv_doctor "/mnt/Datos/Master_Camilo/DL/data/splits_tnbc_doctor.csv" \
                --csv_cercanos "/mnt/Datos/Master_Camilo/DL/data/TN_cercanos_vs_BN_cercanos.csv" \
                --result_dir $result_dir \
                --tag_exp "Classification Images" \
                --activation_image_model "LeakyReLU" \
                --image_model "ResNet" \
                --path_image_model $path_image_model_resnet \
                --num_freeze 159 \
                --hidden_layers 2048 2048 1024 1024 512 512 \
                --output_size 2 \
                --activation "LeakyReLU" \
                --dropout 0.5 \
                --augmentation \
                --n_epochs 200 \
                --batch_size 64 \
                --lr 5e-4 \
                --patience_early 200 \
                --min_lr 1e-6 \
                --loss "BCE" \
                --train \
                --neg_weight 3.22 \
                --pos_weight 0.59 \
                --class_balance

python3 main.py --exp_name "Classification Images_HL1024-1024-512-512_lr1e-4_dr0.5_unfreezelayer4_negweight3.22_posweight0.59_tiff_pseudotrain" \
                --images_dir "/mnt/Datos/Master_Camilo/DL/intratumoral-patches-tiff" \
                --csv_data_path "/mnt/Datos/Master_Camilo/01-ML/data/benigno_vs_maligno_balanceado_pseudotrain_mahalanobis_voto/dl" \
                --csv_subtype "/mnt/Datos/Master_Camilo/DL/data/radiomics_features_lesion_sin_calc_dist_lipoma.csv" \
                --csv_doctor "/mnt/Datos/Master_Camilo/DL/data/splits_tnbc_doctor.csv" \
                --csv_cercanos "/mnt/Datos/Master_Camilo/DL/data/TN_cercanos_vs_BN_cercanos.csv" \
                --result_dir $result_dir \
                --tag_exp "Classification Images" \
                --activation_image_model "LeakyReLU" \
                --image_model "ResNet" \
                --path_image_model $path_image_model_resnet \
                --num_freeze 159 \
                --hidden_layers 1024 1024 512 512 \
                --output_size 2 \
                --activation "LeakyReLU" \
                --dropout 0.5 \
                --augmentation \
                --n_epochs 200 \
                --batch_size 64 \
                --lr 5e-4 \
                --patience_early 200 \
                --min_lr 1e-6 \
                --loss "BCE" \
                --train \
                --neg_weight 3.22 \
                --pos_weight 0.59 \
                --class_balance