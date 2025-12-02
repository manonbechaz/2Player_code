import torch

# Player2
class AE(torch.nn.Module):

    def __init__(self,n_input_channels=3):
        super().__init__()

        self.x = None
        self.height = None
        self.width = None
        self.out_channels = None
        self.kernel_size_cv2d = None
        self.stride_cv2d = None
        self.padding_cv2d = None
        self.kernel_size_mp2d = None
        self.stride_mp2d = None
        self.padding_mp2d = None
        self.alpha = None
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()

        self.enc0 = torch.nn.Conv2d(in_channels=n_input_channels, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc1 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc2 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc3 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc4 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc5 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc6 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc7 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')

        self.dec6 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec6b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec5 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec5b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec4 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec4b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec3 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec3b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec2 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec2b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1a = torch.nn.Conv2d(in_channels=96+n_input_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1b = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1 = torch.nn.Conv2d(in_channels=32, out_channels=n_input_channels, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')

        
        

        self.upscale2d = torch.nn.UpsamplingNearest2d(scale_factor=2)


    def encoder(self,X):
        x = torch.clone(X)
        skips = [x]
        n = x

        n = self.leaky(self.enc0(n))
        n = self.leaky(self.enc1(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc2(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc3(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc4(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc5(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc6(n))
        n = self.pool(n)
        n = self.leaky(self.enc7(n))

        return n, skips

    def decoder(self,X,skips):
        n = torch.clone(X)

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec6(n))
        n = self.leaky(self.dec6b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec5(n))
        n = self.leaky(self.dec5b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec4(n))
        n = self.leaky(self.dec4b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec3(n))
        n = self.leaky(self.dec3b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec2(n))
        n = self.leaky(self.dec2b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)

        return n

    def denoising_head(self,X):
        n = torch.clone(X)

        n = self.leaky(self.dec1a(n))
        n = self.leaky(self.dec1b(n))

        n = self.dec1(n)

        return n

    def forward(self,input_image):
        

        n, skips = self.encoder(input_image)
        n = self.decoder(n,skips)
        n = self.denoising_head(n)
        return n
    