import torch
from torch import nn
import wandb
from dataloaders.dataloader_images import Loader
import time
import sys
import os
from metrics import Metrics
import pandas as pd
import subsets
import seaborn as sns
import matplotlib.pyplot as plt
from models.get_model import get_model
from early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from losses import get_loss

# Columnas de las predicciones por lesión, comunes al csv del test y al de los cruces.
PREDICTION_COLUMNS = ["lesion_ID", "label", "prediction", "prob_benigno", "prob_maligno"]

class TrainModel():
	
	"""
	Class to train and test the model
	Args:
		options (dict): Dictionary with the training options
	"""

	def __init__(self, options:dict):
		
		# Set options
		self.options        = options
		self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		os.makedirs(os.path.join(options.result_dir, options.exp_name, "Saved_Models"), exist_ok=True)

		
		# Initialize wandb
		wandb.init(
			project = "Master-Classification-Images",
			entity  = "juanmorales288981-institucion-universitaria-itm",
			name    = options.exp_name,
			config  = vars(options),
			tags    = options.tag_exp,
		)

		# Initialize model, print summary and save configuration
		self.model = get_model(options)
		self.model.to(self.device)
  
  		# Get loss function
		if (options.class_balance):
			
			self.criterion = get_loss(
				name = options.loss,
				positive_weight = options.pos_weight,
				negative_weight = options.neg_weight,
				gamma 			= options.gamma
			)
   
			self.criterion.to(self.device)
			
		else:
			self.criterion = nn.CrossEntropyLoss()
			self.criterion.to(self.device)
		
		# Get loaders. El loader se guarda para poder recuperar los lesion_ID del
		# test en el mismo orden en que los devuelve su dataloader.
		self.loader 		= Loader(options.images_dir, options.csv_data_path, options.augmentation)
		self.train_loader 	= self.loader.train_dataloader(batch_size=options.batch_size)
		self.val_loader 	= self.loader.val_dataloader(batch_size=options.batch_size)
		self.test_loader 	= self.loader.test_dataloader(batch_size=options.batch_size)

		# Define optimizer and lr scheduler
		self.optimizer 	= torch.optim.Adam(self.model.parameters(), lr=options.lr, betas=(options.b1, options.b2))
		self.metrics  	= Metrics(self.device, options.neg_weight, options.pos_weight)
		self.best_loss  = 0.
  
		self.early_stopping = EarlyStopping(
	  							patience 	= options.patience_early,
				  				dir_save	= os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models")
	   						)

		self.scheduler 		= CosineAnnealingLR(
	  							self.optimizer, 
	  							T_max 	= options.n_epochs,
	  							eta_min = options.min_lr
						 	)
  
		show_batch_images(self.train_loader, save_dir=os.path.join(self.options.result_dir, self.options.exp_name))
  
		if(options.test and options.best_model):
			self.model.load_state_dict(torch.load(os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models", "Best_Model.pth")))
		elif(options.test and not options.best_model):
			self.model.load_state_dict(torch.load(os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models", "Last_Model.pth")))
	
	def train_model(self):
		
		print("\n [*] -> Starting training....\n\n")

		self.prev_time = time.time()

		for self.epoch in range(self.options.init_epoch, self.options.n_epochs):

			self.epoch_stats = {
				"Loss-BCE"	: [],
				"Train_Accuracy"	: [],
				"Train_Sensitivity": [],
				"Train_Specificity": [],
				"Train_F1-Score"	: [],
				"Train_PPV"			: []
			}

			for batch_idx, data in enumerate(self.train_loader):

				# Get data
				inputs, targets,   	= data
				inputs, targets     = inputs.to(self.device), targets.to(self.device)
				inputs, targets		= inputs.float(), targets.long()

				self.model.train()

				# Train model
				logits 		= self.model(inputs)
				loss 		= self.criterion(logits, targets)
				
				# Backpropagation
				self.optimizer.zero_grad()				
				loss.backward()
				self.optimizer.step()

				# Get predictions 
				probs 	= torch.softmax(logits, dim=1)
				preds  	= torch.argmax(probs, dim=1)

				# Update epoch stats
				self.epoch_stats["Loss-BCE"].append(loss.item())
				self.epoch_stats["Train_Accuracy"].append(self.metrics.accuracy(preds, targets.long()))
				self.epoch_stats["Train_Sensitivity"].append(self.metrics.sensitivity(preds, targets.long()))
				self.epoch_stats["Train_Specificity"].append(self.metrics.specificity(preds, targets.long()))
				self.epoch_stats["Train_F1-Score"].append(self.metrics.f1_score(preds, targets.long()))
				self.epoch_stats["Train_PPV"].append(self.metrics.ppv(preds, targets.long()))

				# Compute the elapsed time since the last log
				elapsed_time = time.time() - self.prev_time

				# Calculate the estimated time remaining for the epoch
				hours   = elapsed_time // 3600
				minutes = (elapsed_time % 3600) // 60
				seconds = elapsed_time % 60
				
				# Format the progress information
				progress_str = (
					f"\r[Epoch {self.epoch}/{self.options.n_epochs}] "
					f"[Batch {batch_idx}/{len(self.train_loader)}] "
					f"Lr {self.optimizer.param_groups[0]['lr']:.6f} "
					f"[BCE loss: {loss.item():.4f}] "
					f"ETA: {int(hours)}h{int(minutes)}m{int(seconds)}s"
				)

				# Write the progress information to the console
				sys.stdout.write(progress_str)

				# Move the cursor to the beginning of the line to overwrite the previous progress information
				sys.stdout.flush()
				sys.stdout.write('\r')
				sys.stdout.flush()
				
				if batch_idx % 5 == 0:
					step_log = {
						"Loss-BCE"	: loss.item(),
					}
					wandb.log(step_log)
	 
			# Log epoch stats
			for key, value in self.epoch_stats.items():
				self.epoch_stats[key] = torch.mean(torch.tensor(value)).item()
			
			self.epoch_stats["epoch"] = self.epoch
			self.epoch_stats["lr"] = self.optimizer.param_groups[0]['lr']
			wandb.log(self.epoch_stats)
			self.validation(plot=False)
			self.scheduler.step()

			if self.early_stopping.early_stop:
				print("Deteniendo entrenamiento temprano")

				# Compute the elapsed time since the last log
				elapsed_time = time.time() - self.prev_time

				# Calculate the estimated time remaining for the epoch
				hours   = elapsed_time // 3600
				minutes = (elapsed_time % 3600) // 60
			
				# Save training time to the configuration file
				with open(os.path.join(self.options.result_dir, self.options.exp_name, 'config.txt'), 'a') as f:
					f.write("\n--------------- Time Training ------------------\n")
					f.write("Tiempo de entrenamiento: {} horas y {} minutos\n".format(int(hours), int(minutes)))

				print("\n [✓] -> Done Training! \n\n")
	
				break

			# Save the model after every epoch_chkpt
			if self.epoch % 10 == 0:
	   
				dir_save = os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models")
				os.makedirs(dir_save, exist_ok=True)

				torch.save(self.model[0].state_dict(), os.path.join(dir_save, f"{self.options.image_model}_{self.epoch:03d}.pth"))
				torch.save(self.model[1].state_dict(), os.path.join(dir_save, f"classifier_{self.epoch:03d}.pth"))
				torch.save(self.model.state_dict(), os.path.join(dir_save, f"Model_{self.epoch:03d}.pth"))

		# Compute the elapsed time since the last log
		elapsed_time = time.time() - self.prev_time

		# Calculate the estimated time remaining for the epoch
		hours   = elapsed_time // 3600
		minutes = (elapsed_time % 3600) // 60
	
		# Save training time to the configuration file
		with open(os.path.join(self.options.result_dir, self.options.exp_name, 'config.txt'), 'a') as f:
			f.write("\n--------------- Time Training ------------------\n")
			f.write("Tiempo de entrenamiento: {} horas y {} minutos\n".format(int(hours), int(minutes)))

		print("\n [✓] -> Done Training! \n\n")
	

	def validation(self, plot=False):

		# Set the generator to training mode
		self.model.eval()

		all_targets = []
		all_probs   = []

		with torch.no_grad():

			# Iterate over the training data
			for batch_idx, data in enumerate(self.val_loader):

				inputs, targets  	= data
				inputs, targets     = inputs.to(self.device), targets.to(self.device)
				inputs, targets		= inputs.float(), targets.float()

				# Get predictions
				logits 		= self.model(inputs)
				probs 		= torch.softmax(logits, dim=1)

				# Acumulamos para calcular las metricas sobre todo el conjunto
				all_targets.append(targets.long())
				all_probs.append(probs)

		# Calculate the metrics over every validation sample
		targets = torch.cat(all_targets)
		probs 	= torch.cat(all_probs)
		preds  	= torch.argmax(probs, dim=1)

		metrics = self.metrics.get_metrics(preds, targets, probs, "Val")

		for key, value in metrics.items():
			self.epoch_stats[key] = value.item()

		self.early_stopping(self.epoch_stats["Val_F1-Score"], self.model, self.epoch)
		self.epoch_stats["epoch"] = self.epoch_stats["epoch"]
		wandb.log(self.epoch_stats)

	def test_model(self):

		# Load the best model
		self.model.load_state_dict(torch.load(os.path.join(self.options.result_dir, self.options.exp_name, "Saved_Models", f"Best_Model.pth")))
		print("Modelo cargado Mejor")

		# Set the generator to training mode
		self.model.eval()

		all_targets = []
		all_probs   = []

		with torch.no_grad():

			# Iterate over the training data
			for batch_idx, data in enumerate(self.test_loader):

				inputs, targets  	= data
				inputs, targets     = inputs.to(self.device), targets.to(self.device)
				inputs, targets		= inputs.float(), targets.float()

				# Train Generator
				logits 		= self.model(inputs)
				probs 		= torch.softmax(logits, dim=1)

				# Acumulamos para las metricas, las curvas y la matriz de confusion
				all_targets.append(targets.long())
				all_probs.append(probs)

		# Metricas sobre todas las muestras del test, no promediadas por lote
		targets = torch.cat(all_targets)
		probs 	= torch.cat(all_probs)
		preds  	= torch.argmax(probs, dim=1)

		metrics = self.metrics.get_metrics(preds, targets, probs, "Test")
		curves 	= self.metrics.get_curves(targets, probs)

		dir_save = os.path.join(self.options.result_dir, self.options.exp_name, "Results")
		os.makedirs(dir_save, exist_ok=True)

		# --- 1) Resumen de métricas ---
		summary = {metric: value.item() for metric, value in metrics.items()}

		df_summary = pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])
		df_summary.index.name = "Metric"

		csv_path = os.path.join(dir_save, "test_summary.csv")
		df_summary.to_csv(csv_path)

		# Subir en wandb
		wandb.log(summary)

		print(f"✅ Resumen de métricas guardado en {csv_path}")
		print(df_summary)

		# --- 2) Matriz de confusión ---
		cm = self.metrics.get_confusion_matrix(preds, targets).cpu().numpy()
		plt.figure(figsize=(10,8))
		sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
					xticklabels=["No Cancer=0","Cancer=1"],
					yticklabels=["No Cancer=0","Cancer=1"])
		plt.xlabel("Predicho")
		plt.ylabel("Verdadero")
		plt.title("Matriz de Confusión")
		plt.tight_layout()
		path_cm = os.path.join(dir_save, "confusion_matrix.png")
		plt.savefig(path_cm, dpi=300)
		plt.close()
		print(f"✅ Matriz de confusión guardada en {path_cm}")
		#plt.show()

		# --- 3) Curvas ROC–AUC y Precisión–Recall ---
		test_curves = {"Test": {**curves, "auc": summary["Test_AUC"], "auprc": summary["Test_AU-PRC"]}}

		path_roc = os.path.join(dir_save, "roc_curve.png")
		plot_roc_curves(test_curves, "Curva ROC", path_roc)

		path_prc = os.path.join(dir_save, "pr_curve.png")
		plot_pr_curves(test_curves, "Curva Precisión-Recall", path_prc)

		print(f"✅ ROC–AUC: {summary['Test_AUC']:.3f} | AU-PRC: {summary['Test_AU-PRC']:.3f}")

		# --- 4) Predicciones por lesión y evaluación por subconjuntos ---
		# El test dataloader no baraja ni descarta muestras, así que su orden es
		# exactamente el de las filas del csv de la partición.
		lesions = self.loader.test_dataset.data
		assert len(lesions) == len(targets), "el orden del test loader no coincide con el de su csv"

		predictions = pd.DataFrame({
			"lesion_ID"		: lesions["lesion_ID"].to_numpy(),
			"label"			: targets.cpu().numpy(),
			"prediction"	: preds.cpu().numpy(),
			"prob_benigno"	: probs[:, 0].cpu().numpy(),
			"prob_maligno"	: probs[:, 1].cpu().numpy()
		})

		self.evaluate_subsets(predictions, dir_save)

	def evaluate_subsets(self, predictions: pd.DataFrame, dir_save: str):

		"""
		Evaluate the model on every crossing defined in `subsets.CROSSINGS`.

		No se vuelve a inferir: cada cruce se aísla con una máscara sobre las
		predicciones que ya calculó `test_model`.

		Args:
			- predictions: (pd.DataFrame), per lesion predictions of the test split.
			- dir_save: (str), directory where the test results are written.
		"""

		dir_subsets = os.path.join(dir_save, "Subsets")
		os.makedirs(dir_subsets, exist_ok=True)

		# Pertenencia de cada lesión del test a los grupos que forman los cruces
		table = subsets.build_subset_table(
			lesions 		= predictions,
			csv_subtype 	= self.options.csv_subtype,
			csv_doctor 		= self.options.csv_doctor,
			csv_cercanos	= self.options.csv_cercanos
		)

		path_predictions = os.path.join(dir_save, "test_predictions.csv")
		table.to_csv(path_predictions, index=False)
		print(f"✅ Predicciones del test guardadas en {path_predictions}")

		subset_metrics 		= {}
		subset_predictions 	= []
		subset_curves 		= {}
		rows 				= []

		for crossing in subsets.CROSSINGS:

			subset = table[subsets.get_crossing_mask(table, crossing)]

			# Un cruce con una sola clase deja el AUC y el MCC sin definir
			if subset["label"].nunique() < 2:
				print(f"⚠ Cruce {crossing} omitido: sus {len(subset)} muestras son de una sola clase")
				continue

			targets = torch.tensor(subset["label"].to_numpy(), dtype=torch.long, device=self.device)
			preds 	= torch.tensor(subset["prediction"].to_numpy(), dtype=torch.long, device=self.device)
			probs 	= torch.tensor(subset[["prob_benigno", "prob_maligno"]].to_numpy(), dtype=torch.float, device=self.device)

			metrics = self.metrics.get_metrics(preds, targets, probs, crossing)
			curves 	= self.metrics.get_curves(targets, probs)

			# Las claves prefijadas con el nombre del cruce se suben a wandb; la
			# fila del csv las guarda sin prefijo para que todos los cruces
			# compartan las mismas columnas.
			subset_metrics.update({key: value.item() for key, value in metrics.items()})

			row = {key.removeprefix(f"{crossing}_"): value.item() for key, value in metrics.items()}
			rows.append({
				"Crossing"	: crossing,
				"n"			: len(subset),
				"n_benigno"	: int((subset["label"] == 0).sum()),
				"n_maligno"	: int((subset["label"] == 1).sum()),
				**row
			})

			subset_predictions.append(subset.assign(crossing=crossing)[["crossing"] + PREDICTION_COLUMNS])

			# Un mismo diccionario alimenta las dos curvas del cruce
			subset_curves[crossing] = {**curves, "auc": row["AUC"], "auprc": row["AU-PRC"]}

			plot_roc_curves({crossing: subset_curves[crossing]}, f"Curva ROC - {crossing}", os.path.join(dir_subsets, f"{crossing}_roc.png"))
			plot_pr_curves({crossing: subset_curves[crossing]}, f"Curva Precisión-Recall - {crossing}", os.path.join(dir_subsets, f"{crossing}_prc.png"))

		if not rows:
			print("⚠ Ningún cruce evaluable, no se generan resultados por subconjunto")
			return

		# Subir en wandb
		wandb.log(subset_metrics)

		# Figuras con los cruces superpuestos
		plot_roc_curves(subset_curves, "Curvas ROC por subconjunto", os.path.join(dir_subsets, "roc_curves.png"))
		plot_pr_curves(subset_curves, "Curvas Precisión-Recall por subconjunto", os.path.join(dir_subsets, "prc_curves.png"))

		# Métricas y predicciones de todos los cruces
		df_subsets 		= pd.DataFrame(rows)
		df_predictions 	= pd.concat(subset_predictions, ignore_index=True)

		path_metrics 			= os.path.join(dir_subsets, "subset_metrics.csv")
		path_subset_predictions = os.path.join(dir_subsets, "subset_predictions.csv")

		df_subsets.to_csv(path_metrics, index=False)
		df_predictions.to_csv(path_subset_predictions, index=False)

		print(f"✅ Métricas por subconjunto guardadas en {path_metrics}")
		print(f"✅ Predicciones por subconjunto guardadas en {path_subset_predictions}")
		print(df_subsets.to_string(index=False))



def plot_roc_curves(curves, title, path_save):

	"""
	Plot one or several ROC curves in the same figure.

	Args:
		- curves: dict, name -> curve, where every curve holds the keys 'fpr', 'tpr' and 'auc'.
		- title: str, title of the figure.
		- path_save: str, path of the png file.
	"""

	plt.figure(figsize=(7,5))

	for name, curve in curves.items():
		plt.plot(curve["fpr"], curve["tpr"], label=f"{name} (AUC = {curve['auc']:.3f})")

	plt.plot([0,1],[0,1], 'k--', label="Aleatorio")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title(title)
	plt.legend(loc="lower right", fontsize="small")
	plt.tight_layout()
	plt.savefig(path_save, dpi=300)
	plt.close()


def plot_pr_curves(curves, title, path_save):

	"""
	Plot one or several precision-recall curves in the same figure.

	Args:
		- curves: dict, name -> curve, where every curve holds the keys 'recall', 'precision' and 'auprc'.
		- title: str, title of the figure.
		- path_save: str, path of the png file.
	"""

	plt.figure(figsize=(7,5))

	for name, curve in curves.items():
		plt.plot(curve["recall"], curve["precision"], label=f"{name} (AU-PRC = {curve['auprc']:.3f})")

	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.title(title)
	plt.legend(loc="lower left", fontsize="small")
	plt.tight_layout()
	plt.savefig(path_save, dpi=300)
	plt.close()


def show_batch_images(train_dataloader, num_images=10, save_dir=None):
	
	"""
	Show a batch of images from the dataloader.

	Args:
		- train_dataloader: DataLoader, training dataloader.
		- num_images: int, number of images to show.
		- save_dir: str, directory to save the images."""
	
	images_shown = 0

	plt.figure(figsize=(15, 6))

	for images, labels in train_dataloader:
		# images shape: [B, C, H, W]
		batch_size = images.shape[0]

		for i in range(batch_size):
			if images_shown >= num_images:
				plt.savefig(os.path.join(save_dir, "batch_images.png"), dpi=300)
				return

			img = images[i]
			print(img.size())

			# Convertir a CPU numpy
			img_np = img[0,:,:].detach().cpu().numpy()

			plt.subplot(2, 5, images_shown + 1)
			plt.imshow(img_np, cmap="gray", vmin=-1, vmax=1)
			plt.title(f"Label: {labels[i].item()}")
			plt.axis("off")

			images_shown += 1
