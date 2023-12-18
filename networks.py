import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# TODO :  debug the whole


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.Conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=(3,3), pad_size=(1,1), bias=False)
        self.BN1 = nn.BatchNorm2d(in_ch,in_ch)

        self.Conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=(3,3), pad_size=(1,1), bias=False)
        self.BN2 = nn.BatchNorm2d(in_ch, out_ch)
        self.Conv3 = nn.Conv2d(in_ch,out_ch, kernel_size = (1,1))

    def forward(self,x):
        y = self.Conv1(x)
        y = self.BN1(y)
        y = self.Conv2(y)
        y =  self.BN2(y)
        y_c = self.Conv3(x)

        return y +  y_c



class RepreNet(nn.Module):
    def __init__(self, input_h, input_w, in_ch, channels = [16,16,32,64,128]):
        super().__init__()
        self.in_h = input_h
        self.in_w = input_w
        self.in_ch = in_ch
        self.channels = channels

        self.InitConv = nn.Conv2d(in_ch, channels[0],kernel_size=(8,8), padding = (4,4), pad_type='mirror', bias=True)
        self.MP1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.ResStack1 = nn.Sequential(
            [ResBlock(channels[1], channels[1]) for i in range(6)]
        )

        self.ResStack2 = nn.Sequential(
            [nn.DownSample(kernel_size=(2,2),mode = 'bilinear'), ResBlock(channels[1], channels[2])] + [ResBlock(channels[2], channels[2]) for i in range(5)]
        )
        self.ResStack3 = nn.Sequential(
            [nn.DownSample(kernel_size=(2,2),mode = 'bilinear'), ResBlock(channels[2], channels[3])] + [ResBlock(channels[3], channels[3]) for i in range(5)]
        )

        self.ResStack4 = nn.Sequential(
            [nn.DownSample(kernel_size=(2,2),mode = 'bilinear'), ResBlock(channels[3], channels[4])] + [ResBlock(channels[4], channels[4] for i in range(5))]
        )

    def forward(self,x):

        y = self.InitConv(x)
        y = self.MP1(y)
        y = self.ResStack1(y)
        y = self.ResStack2(y)
        y = self.ResStack3(y)
        y = self.ResStack4(y)

        return y

class comparNet(nn.Module):
    def __init__(self, in_ch, mid_ch, in_h, in_w):
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.in_h = in_h
        self.in_w = in_w

        self.Conv1 = nn.Conv2d(in_ch,mid_ch,kernel_size=(3,3), padding=(1,1), bias=False)
        self.Conv2 = nn.Conv2d(mid_ch,mid_ch,kernel_size=(3,3), padding=(1,1), bias=False)

        self.Linear = nn.Linear(self.mid_ch * self.in_h * self.in_w)

    def forward(self,x):

        y = self.Conv1(x)
        y = self.Conv2(y)

        y = torch.flatten(y)

        y = self.Linear(y)

        y = F.SoftMax(y)

        return y

class AtmoDistNet(nn.Module):
    def __init__(self,in_ch, inter_ch, in_h, in_w):
        super().__init__()

        self.Repre = RepreNet(in_h, in_w, in_ch)
        self.Compar = comparNet(inter_ch * 2, inter_ch, in_h//32, in_w//32)  # * 2 --> stacking

    def forward(self,x1,x2):
        """ Assuming siamese behavior with shared weigths"""

        y1 = self.Repre(x1)
        y2 = self.Repre(x2)

        y_f = torch.cat((y1, y2), dim=1)

        y_f = self.Compar(y_f)

        return y_f
    




