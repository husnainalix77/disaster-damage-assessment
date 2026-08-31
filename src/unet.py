import torch
import torch.nn as nn

# Create an encoder block
class EncoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__() # because class inherits from Pytorch's nn.module
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # creates 1st convolution
        self.relu1 = nn.ReLU() # introducing non-linearity, network learns more complex features
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # creates 2nd convolution
        self.relu2 = nn.ReLU()
        # Network needs to learn complex features before downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # pooling halves height and width only, not channels
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        skip = x # save feature map before pooling for decoder
        x = self.pool(x)
        
        return x, skip

# Create a bottleneck block
class Bottleneck(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))

        return x    

# Create a decoder block
class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Unsampling (increases feature map spatial resolution and decreases the number of channels)
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
            )
        # Combine the upsampled feature map channels with the skip connection
        self.conv1 = nn.Conv2d(
            out_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1
            )
        
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
            )
        self.relu2 = nn.ReLU()
    
    def forward(self, x, skip):
        
        x = self.up(x)
        # Combine with the skip connection (concatenate the channels of x and skip)
        x = torch.cat([x, skip], dim=1) # dim=1 means along channel dimension
        
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))

        return x    

class UNet(nn.Module):
    
    def __init__(self, base_channels=64):
        super().__init__()
        c = base_channels  # shorthand
        
        # Encoder
        self.block1 = EncoderBlock(3, c)
        self.block2 = EncoderBlock(c, c * 2)
        self.block3 = EncoderBlock(c * 2, c * 4)
        
        # Bottleneck
        self.bottleneck = Bottleneck(c * 4, c * 8)
        
        # Decoder
        self.decoder_block3 = DecoderBlock(in_channels=c * 8, skip_channels=c * 4, out_channels=c * 4)
        self.decoder_block2 = DecoderBlock(in_channels=c * 4, skip_channels=c * 2, out_channels=c * 2)
        self.decoder_block1 = DecoderBlock(in_channels=c * 2, skip_channels=c, out_channels=c)
        
        # Final output layer — collapses c channels down to 1 (building / not-building)
        self.final_conv = nn.Conv2d(c, 1, kernel_size=1)
    
    def forward(self, x):
        x, skip1 = self.block1(x)
        x, skip2 = self.block2(x)
        x, skip3 = self.block3(x)
        
        x = self.bottleneck(x)
        
        x = self.decoder_block3(x, skip3)
        x = self.decoder_block2(x, skip2)
        x = self.decoder_block1(x, skip1)
        
        x = self.final_conv(x)
        return x            
            


