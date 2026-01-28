from train import *
from data import *

import argparse
import os

import torch
from torchvision.transforms import Compose, ToTensor, Normalize


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    type = str,
    help = 'Path to data. Must contain folder named NG and OK. Inside, it should contain images'
)
parser.add_argument(
    '--batch_size',
    type = int,
    help = 'Number of data batch size',
    default = 8
)
parser.add_argument(
    '--epochs',
    type = int,
    help = 'Number of train epochs',
    default = 100
)
args = parser.parse_args()

def main():
    base_dir = args.data_dir
    train_dir = os.path.join(base_dir,'train')

    try:
        print("train data folder exists? ", os.path.exists(train_dir))
    except Exception as e:
        raise ValueError(e)
    
    data_transformation = Compose(
        [
            ToTensor(),
            #Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]
    )
    train_dataloader, num_classes_exist = load_Image_Dataloader(train_dir, batch_size=args.batch_size, transform_func=data_transformation)
    print("Finished loading train data!")

    MaskRCNNModule = MaskRCNNTrainer(
        lr= 0.005, 
        momentum = 0.9,
        weight_decay = 0.0005, 
        lr_decay_start= 5,
        early_stopping_delta= 0.00001,
        early_stopping_monitor= 'epoch_train_loss',
        early_stopping_patience= 20,
        early_stopping_mode = "min",
        early_stopping_save_best= True,
        num_classes=num_classes_exist,
        use_pretrained=True)
    MaskRCNNModule.train(train_dataloader, args.epochs)

    #get the model
    final_model = MaskRCNNModule.maskrcnn_mod

    if final_model != None:
        print("Saving model!")
        try:
            base = 'exported_models/final_models'
            os.makedirs(base, exist_ok = True)
            torch.save(final_model.state_dict(), f'{base}/final_model.pth')
            print(f"Final MaskRCNN saved to {base}/final_model.pth")
        except Exception as e:
            raise ValueError(e)

if __name__ == '__main__':
    main()