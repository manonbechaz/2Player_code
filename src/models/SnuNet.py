# Kaiyu Li
# https://github.com/likyoo
#

import torch.nn as nn
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from data_processing.dataset_HRSCD import HRSCD_Dataset


class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)



class SNUNet_ECAM(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, out_ch=2):
        super(SNUNet_ECAM, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.Up4_0 = up(filters[4])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

        self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)

        self.conv_final = nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.sm = nn.Sigmoid()
        self.temperature = 1 #0.01


    def forward(self, xA, xB):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)

        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        out = self.conv_final(out)

        proba = self.sm(out/self.temperature)

        return proba

###########################################################################################################
###########################################################################################################
#################### Lightning Module #####################################################################
###########################################################################################################
###########################################################################################################

class SNuNetModule(pl.LightningModule):

    def __init__(self, lr):
        super().__init__()
    
        self.player1 = SNUNet_ECAM(in_ch=3, out_ch=1)

        # Training parameters
        self.lr = lr
        #self.loss = nn.NLLLoss(weight=torch.Tensor([0.0141,1.9859])) # We could add weights [0.1851,1.8149]
        #self.loss = nn.CrossEntropyLoss(weight=torch.Tensor([0.0141,1.9859]))
        #self.loss = nn.BCELoss() # la il faut bien avoir sigmoids attention
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([140])) #pos_weight=torch.Tensor([10])
        
        self.writer = SummaryWriter()
    
    def lambda_rule(self, epoch):
        lr_l = 1.0 - epoch / float(self.max_epochs + 1)
        return lr_l

    def forward(self, im1, im2): 
        '''
        Generates a change map given two images im1 and im2
        '''

        pred = self.player1(im1,im2)
        #pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        return pred
    
    def training_step(self, batch):
        
        im1, im2, lab = batch

        # Compute change maps
        output = self.player1(im1, im2)
        loss = self.loss(output, lab.float())

        self.log("train_loss", loss,prog_bar=True,on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch):
        im1, im2, lab = batch

        # Compute change maps
        output = self.player1(im1, im2)
        loss = self.loss(output, lab.float()) 

        #pred = torch.argmax(proba.data, dim=1)
        #logs
        sample_change_maps = output[:2]
        changegrid = torchvision.utils.make_grid(sample_change_maps)
        self.logger.experiment.add_image("generated_changemaps", changegrid, self.global_step)
        sample_change_maps = lab[:2]
        changegrid = torchvision.utils.make_grid(sample_change_maps)
        self.logger.experiment.add_image("ground_truth", changegrid, self.global_step)
        sample_reconstructed = im1[:2] 
        reconstructedgrid = torchvision.utils.make_grid(sample_reconstructed)
        self.logger.experiment.add_image("image1", reconstructedgrid, self.global_step)
        sample_reconstructed = im2[:2]
        reconstructedgrid = torchvision.utils.make_grid(sample_reconstructed)
        self.logger.experiment.add_image("image2", reconstructedgrid, self.global_step)


        self.log("val_loss", loss,prog_bar=True,on_epoch=True, on_step=True)
        return loss


    def configure_optimizers(self):
        '''player1_opt = torch.optim.Adam(self.player1.parameters(), lr=self.lr)
        player2_opt = torch.optim.Adam(self.player2.parameters(), lr=self.lr)
        return [player1_opt, player2_opt],[]'''

        optimzizer = torch.optim.Adam(self.parameters(), self.lr, (0.9, 0.99), weight_decay = 1e-4) #, weight_decay = 1e-4
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimzizer, lr_lambda=self.lambda_rule)

        #return {"optimizer": optimzizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return optimzizer
    
    def on_training_epoch_end(self, training_step_outputs):
        # Log gradients and weights at the end of each epoch
        for name, param in self.parameters():
            self.writer.add_histogram(name, param, global_step=self.current_epoch)



PATH_TRAIN_DS_HRSCD = '/data/manon/processed_dataset/train'
PATH_TEST_DS_HRSCD = '/data/manon/processed_dataset/test'
PATH_VAL_DS_HRSCD = '/data/manon/processed_dataset/val'


if __name__ == "__main__": 

    # Datasets
    ds_train = HRSCD_Dataset(folderpath=PATH_TRAIN_DS_HRSCD,changed_only=False,patchnorm=False,vegidx=False)
    ds_val = HRSCD_Dataset(folderpath=PATH_VAL_DS_HRSCD, changed_only=False,patchnorm=False,vegidx=False)

    # Dataloaders
    train_loader = DataLoader(ds_train, batch_size=16, shuffle=True, num_workers=24) 
    val_loader = DataLoader(ds_val, batch_size=16, shuffle = True, num_workers=24)

    # Model
    model = SNuNetModule(lr=1e-4) 

    chkpt_folder = '~/checkpoints/SNUNet'
    # Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint()
    trainer = pl.Trainer(default_root_dir=chkpt_folder, max_epochs=400,check_val_every_n_epoch=1, callbacks=[lr_monitor, checkpoint_callback],accelerator="gpu", devices=1) #profiler="simple", #EarlyStopping(monitor="val_loss", patience = 2000, mode="min"),

    # Train model
    trainer.fit(model, train_loader, val_loader)

    print(trainer.callback_metrics)

    # If we want to resume training:  #TODO add to config
    # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")
    
