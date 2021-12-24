import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from modules import *
from NONLOCAL.non_local_embedded_gaussian import NONLocalBlock2D
import scipy.io as sio

device = 'cuda'



class InSSSRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio):
        super(InSSSRBlock, self).__init__()

        self.alpha_hidden = nn.Parameter(torch.rand(1, requires_grad=True, device=device))
        self.beta = nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        self.gamma = nn.Parameter(torch.randn(1, requires_grad=True, device=device))

        self.upsample1 = Upsample(in_channels, ratio)
        self.upsample2 = Upsample(in_channels, ratio)
        self.upsample3 = Upsample(out_channels, ratio)

        self.downsample1 = Downsample(in_channels, ratio)
        self.downsample2 = Downsample(out_channels, ratio)

        self.nl = NONLocalBlock2D(in_channels=in_channels)

        self.correct = nn.Sequential(
            resblock(out_channels, 3),
            resblock(out_channels, 3),
            resblock(out_channels, 3),
        )

        self.channels_inc = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1, requires_grad=True, device=device))
        self.channels_dec = nn.Parameter(torch.randn(in_channels, out_channels, 1, 1, requires_grad=True, device=device))
        self.conv1x1_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, w):
        alpha = self.alpha_hidden
        ltw = self.upsample1(w)
        d_ltw = self.downsample1(2*alpha*ltw)
        nl_d_wt = self.nl(d_ltw)
        res_y = self.upsample2(nl_d_wt)
        y = self.relu((2*alpha*ltw-(2*alpha/self.beta)*res_y)/self.beta)
        g = F.conv2d(-y, self.channels_inc, stride=1)
        wrt = F.conv2d(w, self.channels_inc, stride=1)
        z = self.relu(self.conv1x1_1(2*(1-alpha)*wrt))
        h = self.upsample3(-z)
        x = self.relu(self.correct(self.beta * g + self.gamma * h))
        xr = F.conv2d(x, self.channels_dec, stride=1)
        u = xr - y
        lx = self.downsample2(x)
        v = lx - z

        return x, y, z, u, v


class MidSSSRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio):
        super(MidSSSRBlock, self).__init__()

        self.alpha_hidden = nn.Parameter(torch.rand(1, requires_grad=True, device=device))
        self.beta = nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        self.gamma = nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        self.eta = nn.Parameter(torch.randn(1, requires_grad=True, device=device))

        self.upsample1 = Upsample(in_channels, ratio)
        self.upsample2 = Upsample(in_channels, ratio)
        self.upsample3 = Upsample(out_channels, ratio)
        self.upsample4 = Upsample(out_channels, ratio)

        self.downsample1 = Downsample(in_channels, ratio)
        self.downsample2 = Downsample(out_channels, ratio)
        self.downsample3 = Downsample(out_channels, ratio)
        self.downsample4 = Downsample(out_channels, ratio)

        self.nl = NONLocalBlock2D(in_channels=in_channels)

        self.correct = nn.Sequential(
            resblock(out_channels, 3),
            resblock(out_channels, 3),
            resblock(out_channels, 3),
        )

        self.channels_inc = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1, requires_grad=True, device=device))
        self.channels_dec = nn.Parameter(torch.randn(in_channels, out_channels, 1, 1, requires_grad=True, device=device))
        self.conv1x1_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, w, x, u, v):
        alpha = self.alpha_hidden
        eta = self.relu(self.eta)
        ltw = self.upsample1(w)

        xkr = F.conv2d(x, self.channels_dec, stride=1)

        y = 2*alpha*ltw + self.beta* (u + xkr)
        d_y = self.downsample1(y)
        nl_d_y = self.nl(d_y)
        res_y = self.upsample2(nl_d_y)

        y = self.relu((y-(2*alpha/self.beta)*res_y)/self.beta)
        g = F.conv2d(u - y, self.channels_inc, stride=1)



        lx = self.downsample2(x)
        ltlx = self.upsample3(lx)

        xkrrt = F.conv2d(xkr, self.channels_inc, stride=1)

        lxk = self.downsample3(x)

        wrt = F.conv2d(w, self.channels_inc, stride=1)
        z = self.gamma * (lxk+v) + 2*(1-alpha)*wrt
        z = self.relu(self.conv1x1_1(z))
        h = self.upsample4(v - z)

        xk1 = self.relu(self.correct(x - eta * ((self.beta * g + self.gamma * ltlx) + (self.beta * xkrrt + self.gamma * h))))

        xk1r = F.conv2d(xk1, self.channels_dec, stride=1)
        uk1 = u + xk1r - y
        lxk1 = self.downsample4(xk1)
        vk1 = v + lxk1 - z

        return xk1, y, z, uk1, vk1


class OutSSSRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio):
        super(OutSSSRBlock, self).__init__()
        self.alpha_hidden = nn.Parameter(torch.rand(1, requires_grad=True, device=device))
        self.beta = nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        self.gamma = nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        self.eta = nn.Parameter(torch.randn(1, requires_grad=True, device=device))

        self.upsample1 = Upsample(in_channels, ratio)
        self.upsample2 = Upsample(in_channels, ratio)
        self.upsample3 = Upsample(out_channels, ratio)
        self.upsample4 = Upsample(out_channels, ratio)

        self.downsample1 = Downsample(in_channels, ratio)
        self.downsample2 = Downsample(out_channels, ratio)
        self.downsample3 = Downsample(out_channels, ratio)

        self.nl = NONLocalBlock2D(in_channels=in_channels)

        self.correct = nn.Sequential(
            resblock(out_channels, 3),
            resblock(out_channels, 3),
            resblock(out_channels, 3),
        )

        self.channels_inc = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1, requires_grad=True, device=device))
        self.channels_dec = nn.Parameter(torch.randn(in_channels, out_channels, 1, 1, requires_grad=True, device=device))
        self.conv1x1_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, w, x, u, v):
        alpha = self.alpha_hidden
        eta = self.relu(self.eta)
        ltw = self.upsample1(w)
        xkr = F.conv2d(x, self.channels_dec, stride=1)
        y = 2*alpha*ltw + self.beta * (u + xkr)
        d_y = self.downsample1(y)
        nl_d_y = self.nl(d_y)
        res_y = self.upsample2(nl_d_y)
        y = self.relu((y-(2*alpha/self.beta)*res_y)/self.beta)
        g = F.conv2d(u - y, self.channels_inc, stride=1)

        lx = self.downsample2(x)
        ltlx = self.upsample3(lx)

        xkrrt = F.conv2d(xkr, self.channels_inc, stride=1)

        lxk = self.downsample3(x)

        wrt = F.conv2d(w, self.channels_inc, stride=1)
        z = self.gamma * (lxk+v) + 2*(1-alpha)*wrt
        z = self.relu(self.conv1x1_1(z))
        h = self.upsample4(v - z)

        xk1 = self.relu(self.correct(x - eta * ((self.beta * g + self.gamma * ltlx) + (self.beta * xkrrt + self.gamma * h))))


        return xk1, y, z


class S3RNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, ratio=4):
        super(S3RNet, self).__init__()
        self.body1 = InSSSRBlock(in_channels, out_channels, ratio)
        self.body2 = MidSSSRBlock(in_channels, out_channels, ratio)
        self.body3 = OutSSSRBlock(in_channels, out_channels, ratio)

        self.res_x = nn.Sequential(
            Upsample(in_channels, ratio),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )

        self.convx = nn.Sequential(
            nn.Conv2d(in_channels=4 * out_channels, out_channels= out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.convxx = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, w):
        listX = []
        listY = []
        listZ = []
        res_x = self.res_x(w)
        x1, y1, z1, u, v = self.body1(w)
        listX.append(x1)
        listY.append(y1)
        listZ.append(z1)
        x2, y2, z2, u, v = self.body2(w, x1, u, v)
        listX.append(x2)
        listY.append(y2)
        listZ.append(z2)
        x3, y3, z3, u, v = self.body2(w, x2, u, v)
        listX.append(x3)
        listY.append(y3)
        listZ.append(z3)
        x4, y4, z4 = self.body3(w, x3, u, v)
        listX.append(x4)
        listY.append(y4)
        listZ.append(z4)

        xx = torch.cat([x1, x2, x3, x4], dim=1)
        xx = self.convx(xx)
        x = xx + res_x
        x = self.convxx(x)

        return x, y4, z4, listX, listY, listZ


if __name__ == '__main__':
    x = torch.randn(2, 3, 6, 6)
    model = S3RNet(3, 6, 2)
    x, listx = model(x)
    print(x.size())

