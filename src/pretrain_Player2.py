import torch
import lightning.pytorch as pl
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from data_processing.dataset_ValaisCD import Valais_Dataset
from data_processing.dataset_HRSCD import HRSCD_Dataset
from data_processing.dataset_LEVIRCD import LEVIRCD_Dataset
from data_processing.dataset_WHU import WHUCD_Dataset
from lightning.pytorch import seed_everything
from models.player2 import AE
from argparse import ArgumentParser


class AEModule(pl.LightningModule):

    def __init__(self, lr):
        super().__init__()
        self.player2 = AE()

        # Training parameters
        self.lr = lr
        self.reconstruction_loss = torch.nn.MSELoss()
        self.writer = SummaryWriter()   

    def forward(self, im1):
        reconstructed_im = self.player2(im1)
        return reconstructed_im
                

    def training_step(self, batch):
        im1, im2, lab = batch 
        reconstructed_im = self.player2(im1)

        #loss
        loss = self.reconstruction_loss(im1, reconstructed_im)

        self.log("train_loss", loss,prog_bar=True,on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch):
        im1, im2, lab = batch
        reconstructed_im = self.player2(im1)

        #loss
        loss = self.reconstruction_loss(im1, reconstructed_im)

        self.log("val_loss", loss,prog_bar=True,on_epoch=True, on_step=True)
        return loss


    def configure_optimizers(self):
        optimzizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimzizer
    
    def on_training_epoch_end(self, training_step_outputs):
        # Log gradients and weights at the end of each epoch
        for name, param in self.named_parameters():
            self.writer.add_histogram(name, param, global_step=self.current_epoch)

if __name__ == "__main__": 

    parser = ArgumentParser()
    parser.add_argument("--dataset", type = str)
    args = parser.parse_args()


    seed_everything(42)

    torch.use_deterministic_algorithms(True, warn_only=True)

    # Datasets

    # Datasets
    if args.dataset == 'HRSCD':
        ds_train = HRSCD_Dataset(type='train', refined_labels=True, given_list='test_imgs_filtered.pkl')
        ds_val = HRSCD_Dataset(type='val', refined_labels=True, given_list='test_imgs_filtered.pkl')

    if args.dataset == 'Valais':
        ds_train = Valais_Dataset(type='train',cropsize=256,given_list='test_imgs_filtered_5000_2.pkl')
        ds_val = Valais_Dataset(type='val', cropsize=256, given_list='test_imgs_filtered_5000_2.pkl')

    if args.dataset == 'LEVIR':
        ds_train = LEVIRCD_Dataset(cropsize=256, type='train')
        ds_val = LEVIRCD_Dataset(cropsize=256, type='val')
    
    if args.dataset == 'WHUCD':
        ds_train = WHUCD_Dataset(cropsize=256, type='train')
        ds_val = WHUCD_Dataset(cropsize=256, type='val')

    # Dataloaders
    train_loader = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=24) 
    val_loader = DataLoader(ds_val, batch_size=8, num_workers=24)

    # Model
    model = AEModule(lr=1e-4)

    chkpt_folder = './checkpoints/player2/' 

    # Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint()
    checkpoint_callback3 = ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(default_root_dir=chkpt_folder, max_epochs=10,check_val_every_n_epoch=1, callbacks=[lr_monitor, checkpoint_callback, checkpoint_callback3],accelerator="gpu", devices=1,deterministic=True) 
    
    # Train model
    trainer.fit(model, train_loader,val_loader)

    print(trainer.callback_metrics)