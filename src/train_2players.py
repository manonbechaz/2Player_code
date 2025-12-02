import torch
from torch.utils.data import DataLoader
from models.players import PlayersGame_W_Probability
from data_processing.dataset_HRSCD import HRSCD_Dataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from argparse import ArgumentParser, ArgumentTypeError
from lightning.pytorch import seed_everything
from data_processing.dataset_ValaisCD import Valais_Dataset
from data_processing.dataset_LEVIRCD import LEVIRCD_Dataset
import warnings
from rasterio.errors import NotGeoreferencedWarning
from data_processing.dataset_WHU import WHUCD_Dataset


VAL_FOLDER = '/data/manon/LEVIR-CD/val'
TRAIN_FOLDER = '/data/manon/LEVIR-CD/train'

PATH_TRAIN_DS_HRSCD = '/data/manon/HRSCD_D35/train'
PATH_TEST_DS_HRSCD = '/data/manon/HRSCD_D35/test'
PATH_VAL_DS_HRSCD = '/data/manon/HRSCD_D35/val'

PATH_DEV_DS_VALAIS = '/data/manon/Valais_CD_Large/dev'
PATH_TRAIN_DS_VALAIS = '/data/manon/Valais_CD_Large/train'
PATH_TEST_DS_VALAIS = '/data/manon/Valais_CD_Large/test'
PATH_VAL_DS_VALAIS = '/data/manon/Valais_CD_Large/val'

if __name__ == "__main__": 

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise ArgumentTypeError('Boolean value expected.')
        
    def if0None(v):
        if v.lower() in ('0'):
            return None
        else:
            return v

    parser = ArgumentParser()
    parser.add_argument("--dataset", type = str)
    parser.add_argument("--pretrained_P2", type=if0None, default=None)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--freeze", type=str2bool)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--ckpt", type=if0None)
    parser.add_argument("--P1", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    seed_everything(42)

    torch.use_deterministic_algorithms(True, warn_only=True)

    # Datasets
    if args.dataset == 'HRSCD':
        ds_train = HRSCD_Dataset(folderpath=PATH_TRAIN_DS_HRSCD, refined_labels=True, probability_maps = True, given_list='test_imgs_filtered.pkl')
        ds_val = HRSCD_Dataset(folderpath=PATH_TEST_DS_HRSCD, refined_labels=True, probability_maps = True, given_list='test_imgs_filtered.pkl')
        if args.pretrained_P2 is None:
            pretrained_AE = '~/checkpoints/D35/AE/lightning_logs/version_0/checkpoints/epoch=4-step=1095.ckpt'
        else: 
            pretrained_AE = args.pretrained_P2

    if args.dataset == 'Valais':
        ds_train = Valais_Dataset(folderpath=PATH_TRAIN_DS_VALAIS,cropsize=256,probability_maps=True,given_list='test_imgs_filtered_5000_2.pkl')
        ds_val = Valais_Dataset(folderpath=PATH_VAL_DS_VALAIS, cropsize=256,probability_maps=True, given_list='test_imgs_filtered_5000_2.pkl')
        if args.pretrained_P2 is None:
            pretrained_AE = '~/checkpoints/ValaisCD/AE/lightning_logs/version_1/checkpoints/epoch=9-step=4380.ckpt'
        else: 
            pretrained_AE = args.pretrained_P2
        

    if args.dataset == 'LEVIR':
        ds_train = LEVIRCD_Dataset(cropsize=256, type='train',probability_maps=True)
        ds_val = LEVIRCD_Dataset(cropsize=256, type='val',probability_maps=True)
        if args.pretrained_P2 is None:
            pretrained_AE = '~/checkpoints/LEVIRCD/AE/lightning_logs/version_11/checkpoints/epoch=9-step=560.ckpt'
        else: 
            pretrained_AE = args.pretrained_P2
        
    if args.dataset == 'WHUCD':
        ds_train = WHUCD_Dataset(cropsize=256, type='train',probability_maps=True)
        ds_val = WHUCD_Dataset(cropsize=256, type='val',probability_maps=True)
        if args.pretrained_P2 is None:
            pretrained_AE = '~/checkpoints/WHUCD/AE/lightning_logs/version_1/checkpoints/epoch=9-step=6050.ckpt'
        else: 
            pretrained_AE = args.pretrained_P2
        
    
    # Dataloaders
    train_loader = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=24) 
    val_loader = DataLoader(ds_val, batch_size=8, shuffle = True, num_workers=24)

    # Model
    model = PlayersGame_W_Probability(lr=args.lr, P1=args.P1, lamb = args.lamb, lamb_prob=args.lamb_prob, alpha=args.alpha, beta=args.beta, pretrained_AE=pretrained_AE, freeze = args.freeze)
    
    chkpt_folder = './checkpoints/2player/'

    # Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor="val_F1", mode='max', save_top_k=1)
    checkpoint_callback4 = ModelCheckpoint()
    checkpoint_callback3 = ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(default_root_dir=chkpt_folder, max_epochs=args.epochs,check_val_every_n_epoch=1, callbacks=[lr_monitor,checkpoint_callback,checkpoint_callback3, checkpoint_callback4],accelerator="gpu", devices=1,deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Train model
    trainer.fit(model, train_loader, val_loader,ckpt_path=args.ckpt)

    print(trainer.callback_metrics)
