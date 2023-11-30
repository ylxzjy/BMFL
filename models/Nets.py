import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(8, 256, 2, batch_first=True)
        self.drop = nn.Dropout()
        self.out = nn.Linear(256, 80)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.embed(x)
        x, hidden = self.lstm(x)
        x = self.drop(x)
        return self.out(x[:, -1, :])



#bm_fl
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

#for mnist and 8fashion
class Generatormnist(nn.Module):
    def __init__(self,args):
        super(Generatormnist, self).__init__()
        self.args = args 
        self.label_emb = nn.Embedding(args.num_classes, args.latent_dim)

        self.init_size = 28 // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    # def __init__(self,args):
    #     super(Generatormnist, self).__init__()
    #     self.args = args 
    #     self.label_emb = nn.Embedding(args.num_classes, args.latent_dim)

    #     self.init_size = args.img_size // 4  # Initial size before upsampling
    #     self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

    #     self.conv_blocks1 = nn.Sequential(
    #         nn.BatchNorm2d(128),
    #         nn.Upsample(scale_factor=2),
    #         nn.Conv2d(128, 128, 3, stride=1, padding=1),
    #         nn.BatchNorm2d(128, 0.8),
    #         nn.LeakyReLU(0.2, inplace=True),
    #     )

    #     self.conv_blocks2 = nn.Sequential(
    #         nn.Upsample(scale_factor=2),
    #         nn.Conv2d(128, 64, 3, stride=1, padding=1),
    #         nn.BatchNorm2d(64, 0.8),
    #         nn.LeakyReLU(0.2, inplace=True),
    #         nn.Conv2d(64, 3, 3, stride=1, padding=1),  # Output 3 channels for RGB image
    #         nn.Tanh(),
    #     )

    #     self.conv_blocks3 = nn.Sequential(
    #         nn.Upsample(scale_factor=2),
    #         nn.Conv2d(128, 64, 3, stride=1, padding=1),
    #         nn.BatchNorm2d(64, 0.8),
    #         nn.LeakyReLU(0.2, inplace=True),
    #         nn.Conv2d(64, 1, 3, stride=1, padding=1),  # Output 1 channel for grayscale image
    #         nn.Tanh(),
    #     )

    # def forward(self, noise, labels):
    #     gen_input = torch.mul(self.label_emb(labels), noise)
    #     out = self.l1(gen_input)
    #     out = out.view(out.shape[0], 128, self.init_size, self.init_size)

    #     if self.args.dataset == "cifar10":
    #         img = self.conv_blocks1(out)
    #         img = self.conv_blocks2(img)
    #     else:
    #         img = self.conv_blocks1(out)
    #         img = self.conv_blocks3(img)
    #     #print("img.shape: ",img.shape)
    #     return img
    
    
        
#for mnist and 8fashion
class Discriminatormnist(nn.Module):
    def __init__(self,args):
        super(Discriminatormnist, self).__init__()
        self.args = args
        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 2
        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, args.num_classes), nn.Softmax(dim=1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        #print(out.shape)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label
    
    
    # def __init__(self, args):
    #     super(Discriminatormnist, self).__init__()
    #     self.args = args

    #     def discriminator_block(in_filters, out_filters, bn=True):
    #         """Returns layers of each discriminator block"""
    #         block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
    #         if bn:
    #             block.append(nn.BatchNorm2d(out_filters, 0.8))
    #         return block

    #     # Input channel adapter
    #     if args.dataset == "cifar10":
    #         self.channel_adapter = None
    #     else:
    #         self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1)
        

    #     self.conv_blocks = nn.Sequential(
    #         *discriminator_block(3, 16, bn=False),
    #         *discriminator_block(16, 32),
    #         *discriminator_block(32, 64),
    #         *discriminator_block(64, 128),
    #     )

    #     # The height and width of downsampled image
    #     ds_size = 2
    #     # Output layers
    #     self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
    #     self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, args.num_classes), nn.Softmax(dim=1))

    # def forward(self, img):
    #     # Apply channel adapter if necessary
    #     if self.args.dataset != "cifar10":
    #         img = self.channel_adapter(img)

    #     out = self.conv_blocks(img)
    #     out = out.view(out.shape[0], -1)
    #     validity = self.adv_layer(out)
    #     label = self.aux_layer(out)

    #     return validity, label



    


class Generatorcifar(nn.Module):
    # initializers
    def __init__(self, args, d=128):
        super(Generatorcifar, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, args.channels, 4, 2, 1)


    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        return x

class Discriminatorcifar(nn.Module):
    # initializers
    def __init__(self, args, d=128):
        super(Discriminatorcifar, self).__init__()
        self.args = args
        self.conv1_1 = nn.Conv2d(args.channels, d//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(10, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

        self.aux_layer = nn.Sequential(nn.Linear(128 * 8 ** 2, args.num_classes), nn.Softmax(dim=1))

    # def forward(self, input):
    def forward(self, input, label):

        x = F.leaky_relu(self.conv1_1(input), 0.2)

        y = F.leaky_relu(self.conv1_2(label), 0.2)

        x = torch.cat([x, y], 1)

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)

        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        #print("x: ",x.shape)
        label = x.reshape(self.args.train_bs, -1)
        #print("label:", label.shape)
        label = self.aux_layer(label)
        x = F.sigmoid(self.conv4(x))

        return x,label