from models.image_models import get_image_model
from torch import nn
import torch
from types import SimpleNamespace
from models.mlp_models import MLP

def get_model(options:dict):
    
    image_model = get_image_model(
        model_name      = options.image_model,
        weigths_file    = options.path_image_model,
        weights_type    = options.weights_type,
        num_freeze      = options.num_freeze
    )
    
    classifier  = MLP(input_size=image_model.features, hidden_layers=options.hidden_layers, output_size=options.output_size, activation=options.activation, dropout=options.dropout)
    final_model = nn.Sequential(image_model, classifier)

    return final_model


if __name__ == "__main__":
    
    options = {
        "strategy"          : "images",
        "image_model"       : "ResNet",
        "path_image_model"  : "/mnt/Datos/Master_Camilo/DL/weights/RadImageNet_pytorch/ResNet50.pt",
        "weights_type"      : "custom",
        "freeze_backbone"   : True,
    }
    
    options = SimpleNamespace(**options)
    
    model   = get_model(options)
    image   = torch.randn(1, 3, 224, 224)
    output  = model(image)
    print(model)