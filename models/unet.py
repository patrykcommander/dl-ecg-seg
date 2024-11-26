import torch
import torch.nn as nn
import torch.nn.functional as F

# downsampling encoder block
class SingleEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SingleEncoderBlock, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        f = self.encode(x)
        p = self.pool(f)

        return (f, p)

# upsampling decoder block
class SingleDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, transpose_stride):
        super(SingleDecoderBlock, self).__init__()
        
        self.transpose = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=transpose_stride, output_padding=1, padding=1)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
        )

    def forward(self, x, f):
        x = self.transpose(x)
        x = torch.cat((x, f), dim=1)
        x = self.conv(x)
        return x
    
# UNET for 1D sequence
class UNET(nn.Module):
    def __init__(self, input_dim=1, output_dim=4, filters=[8, 16, 32, 64, 128], kernels=[3, 3, 3, 3, 3]):
        super(UNET, self).__init__()

        # Encoder layers (downsampling)
        self.down_1 = SingleEncoderBlock(in_channels=input_dim, out_channels=filters[0], kernel_size=kernels[0])
        self.down_2 = SingleEncoderBlock(in_channels=filters[0], out_channels=filters[1], kernel_size=kernels[1])
        self.down_3 = SingleEncoderBlock(in_channels=filters[1], out_channels=filters[2], kernel_size=kernels[2])
        self.down_4 = SingleEncoderBlock(in_channels=filters[2], out_channels=filters[3], kernel_size=kernels[3])

        # Bottleneck (central part of the U-Net)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=filters[3], out_channels=filters[4], kernel_size=kernels[4], padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=filters[4], out_channels=filters[4], kernel_size=kernels[4], padding="same"),
            nn.ReLU(),
        )

        # Decoder layers (upsampling)
        self.up_1 = SingleDecoderBlock(in_channels=filters[4], out_channels=filters[3], kernel_size=kernels[4], transpose_stride=2)
        self.up_2 = SingleDecoderBlock(in_channels=filters[3], out_channels=filters[2], kernel_size=kernels[2], transpose_stride=2)
        self.up_3 = SingleDecoderBlock(in_channels=filters[2], out_channels=filters[1], kernel_size=kernels[1], transpose_stride=2)
        self.up_4 = SingleDecoderBlock(in_channels=filters[1], out_channels=filters[0], kernel_size=kernels[0], transpose_stride=2)

        # Final classifier layer (output)
        self.classifier = nn.Linear(in_features=filters[0], out_features=output_dim) #nn.Conv1d(in_channels=filters[0], out_channels=output_dim, kernel_size=1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, x):
        # Encoder part: downsample the input and save skip connections
        x = x.transpose(1,2)
        f1, p1 = self.down_1(x)
        f2, p2 = self.down_2(p1)
        f3, p3 = self.down_3(p2)
        f4, p4 = self.down_4(p3)

        # Bottleneck
        neck = self.bottleneck(p4)

        # Decoder part: upsample and concatenate with the skip connections
        x = self.up_1(neck, f4)
        x = self.up_2(x, f3)
        x = self.up_3(x, f2)
        x = self.up_4(x, f1)

        x = x.transpose(1,2)
        x = self.classifier(x)
        x = x.transpose(1,2)

        return x


# Attention mechanism from Attention UNET paper
class Attention(nn.Module):
    def __init__(self, skip_connection_channels, features_channels):
        super(Attention, self).__init__()
        self.conv_connection_features = nn.Conv1d(in_channels=skip_connection_channels, out_channels=features_channels, kernel_size=3, stride=2, padding=3//2)
        self.conv_features_connection = nn.Conv1d(in_channels=1, out_channels=skip_connection_channels, kernel_size=3, padding="same")

        self.conv_1x1 = nn.Conv1d(in_channels=features_channels, out_channels=1, kernel_size=1, stride=1, padding="same")

        self.resample = nn.ConvTranspose1d(in_channels=skip_connection_channels, out_channels=skip_connection_channels, kernel_size=3, stride=2, output_padding=1, padding=1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device) 

    def forward(self, features, skip_connection):
        skip_connecton_transformed =  self.conv_connection_features(skip_connection) # (batch_size, features_channels, features_length)
        x = skip_connecton_transformed + features # (batch_size, features_channels, features_length)
        x = F.relu(x) # (batch_size, features_channels, features_length)
        x = self.conv_1x1(x) # (batch_size, 1, features_length)
        x = F.sigmoid(x) # (batch_size, 1, features_length)

        x = self.conv_features_connection(x) # (batch_size, skip_connection_channels, features_length)

        x = self.resample(x)

        x = x *  skip_connection # (batch_size, skip_connection_channels, skip_connection_length)
        return x 

# Attention UNET for 1D sequence
class Attention_UNET(UNET):
    def __init__(self, input_dim, output_dim, filters=[8, 16, 32, 64, 128], kernels=[3, 3, 3, 3, 3]):
        super(Attention_UNET, self).__init__(input_dim, output_dim, filters, kernels)

        self.attention_1 = Attention(features_channels=filters[4], skip_connection_channels=filters[3])
        self.attention_2 = Attention(features_channels=filters[3], skip_connection_channels=filters[2])
        self.attention_3 = Attention(features_channels=filters[2], skip_connection_channels=filters[1])
        self.attention_4 = Attention(features_channels=filters[1], skip_connection_channels=filters[0])

    def forward(self, x):
        # Encoder part: downsample the input and save skip connections
        x = x.transpose(1,2)
        f1, p1 = self.down_1(x)
        f2, p2 = self.down_2(p1)
        f3, p3 = self.down_3(p2)
        f4, p4 = self.down_4(p3)

        # Bottleneck
        neck = self.bottleneck(p4)

        # Decoder part: upsample and concatenate with the skip connections
        att_f4 = self.attention_1(neck, f4)
        x = self.up_1(neck, att_f4)

        att_f3 = self.attention_2(x, f3)
        x = self.up_2(x, att_f3)

        att_f2 = self.attention_3(x, f2)
        x = self.up_3(x, att_f2)

        att_f1 = self.attention_4(x, f1)
        x = self.up_4(x, att_f1)

        x = x.transpose(1,2)
        x = self.classifier(x)
        x = x.transpose(1,2)

        return x
