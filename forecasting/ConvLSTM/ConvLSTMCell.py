import torch
import torch.nn as nn

# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTMCell, self).__init__()  

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
        
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding)           

        # Initialize weights for Hadamard Products
        test = torch.zeros(out_channels, *frame_size)
        # print('test', torch.any(test.isnan()))
        self.W_ci = nn.Parameter(torch.zeros(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.zeros(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.zeros(out_channels, *frame_size))
        # print('----------')
        # print(self.W_ci, self.W_co, self.W_cf)
        # print(torch.any(self.W_ci.isnan()), torch.any(self.W_co.isnan()), torch.any(self.W_cf.isnan()))
        # print(torch.sum(self.W_ci.isnan()))
        # exit()

    def forward(self, X, H_prev, C_prev):

        # print('1', torch.any(X.isnan()), torch.any(H_prev.isnan()), torch.any(C_prev.isnan()))
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))
        # print('2', torch.any(conv_output.isnan()))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)
        
        # print('3', torch.any(i_conv.isnan()), torch.any(self.W_ci.isnan()), torch.any(C_prev.isnan()))

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )

        # print('4', torch.any(f_conv.isnan()), torch.any(self.W_cf.isnan()), torch.any(C_prev.isnan()))

        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

        # print('5', torch.any(input_gate.isnan()), torch.any(forget_gate.isnan()))

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C