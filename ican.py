import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
import math
import copy
from collections import OrderedDict
from torch.autograd import Variable

NUM_CLASS = 31
BATCH_SIZE = 16
INI_DISC_WEIGHT_SCALE = 200**4
INI_DISC_BIAS = 0.5
LAST_WEIGHT_LIMIT = -2


class Contrast_ReLU_activate(nn.Module):

    def __init__(self, initWeightScale, initBias):

        super(Contrast_ReLU_activate, self).__init__()

        self.dom_func_weight = nn.Parameter(torch.ones(1),requires_grad=True)
        self.dom_func_bias = Variable(torch.FloatTensor([0]).cuda())

        self.weight_scale = initWeightScale
        self.add_bias = initBias

    def forward(self, dom_res, dom_label, init_weight):

        w = (self.dom_func_weight * self.weight_scale).expand_as(init_weight)
        b = (self.dom_func_bias + self.add_bias).expand_as(init_weight)

        dom_prob = F.sigmoid(dom_res).squeeze()
        dom_variance = torch.abs(dom_prob - 0.5)

        act_weight = 0.8 - w * dom_variance**4  + b
        
        # Minimise function to zero(target)
        zeros_var = b
        f_weight = torch.max(act_weight, zeros_var)

        final_weight = f_weight
        
        return final_weight, w.squeeze().data[0], b.squeeze().data[0]


class Discriminator_Weights_Adjust(nn.Module):

    def __init__(self):

        super(Discriminator_Weights_Adjust, self).__init__()

        self.main_var = Variable(torch.FloatTensor([0]).cuda())
        self.l1_var = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.l2_var = nn.Parameter(torch.zeros(1),requires_grad=True)

        self.k_var = Variable(torch.FloatTensor([-0.9]).cuda())

        self.l1_rev = Variable(torch.ones(1).cuda())
        self.l2_rev = Variable(torch.ones(1).cuda())
        self.l3_rev = Variable(torch.ones(1).cuda())

    def forward(self, main_weight, l1_weight, l2_weight, l3_weight):

        w_main = main_weight + self.main_var

        w_l1 = l1_weight + self.l1_var
        w_l2 = l2_weight + self.l2_var
        w_l3 = (w_main - self.k_var) - w_l1 - w_l2

        l1_rev = self.l1_rev
        l2_rev = self.l2_rev
        l3_rev = self.l3_rev
        if w_l1.data[0] < 0:
            l1_rev[0] = -1
        if w_l2.data[0] < 0:
            l2_rev[0] = -1
        if w_l3.data[0] < 0:
            l3_rev[0] = -1

        return torch.abs(w_main), torch.abs(w_l1), torch.abs(w_l2), torch.abs(w_l3), l1_rev, l2_rev, l3_rev


class ICAN(nn.Module):
    def __init__(self, pre_trained):
        super(ICAN, self).__init__()
        self.num_class = NUM_CLASS

        self.disc_activate = Contrast_ReLU_activate(INI_DISC_WEIGHT_SCALE, INI_DISC_BIAS)
        self.disc_weight = Discriminator_Weights_Adjust()

        self.conv1 = pre_trained.conv1
        self.bn1 = pre_trained.bn1
        self.relu = pre_trained.relu
        self.maxpool = pre_trained.maxpool

        self.layer1 = pre_trained.layer1
        self.layer2 = pre_trained.layer2
        self.layer3 = pre_trained.layer3
        self.layer4 = pre_trained.layer4

        self.domain_pred = nn.Sequential(nn.Linear(256, 3072), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(2048, 1))
        self.domain_pred_l1 = nn.Sequential(nn.Linear(256, 3072), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(2048, 1))
        self.domain_pred_l2 = nn.Sequential(nn.Linear(256, 3072), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(2048, 1))
        self.domain_pred_l3 = nn.Sequential(nn.Linear(256, 3072), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(2048, 1))

        self.process = pre_trained.avgpool

        self.process_l1 = nn.AvgPool2d(kernel_size=56)
        self.process_l2 = nn.AvgPool2d(kernel_size=28)
        self.process_l3 = nn.AvgPool2d(kernel_size=14)

        self.source_bottleneck = nn.Sequential(nn.Linear(pre_trained.fc.in_features, 256))

        self.l1_bottleneck = nn.Sequential(nn.Linear(256, 256))
        self.l2_bottleneck = nn.Sequential(nn.Linear(512, 256))
        self.l3_bottleneck = nn.Sequential(nn.Linear(1024, 256))

        self.source_classifier = nn.Sequential(nn.Linear(256, self.num_class))

        # ----- data parallel (multi-gpu) -------
        self.conv1 = nn.DataParallel(self.conv1)
        self.layer1 = nn.DataParallel(self.layer1)
        self.layer2 = nn.DataParallel(self.layer2)
        self.layer3 = nn.DataParallel(self.layer3)
        self.layer4 = nn.DataParallel(self.layer4)

        self.source_bottleneck = nn.DataParallel(self.source_bottleneck)
        self.l1_bottleneck = nn.DataParallel(self.l1_bottleneck)
        self.l2_bottleneck = nn.DataParallel(self.l2_bottleneck)
        self.l3_bottleneck = nn.DataParallel(self.l3_bottleneck)

        self.domain_pred = nn.DataParallel(self.domain_pred)
        self.domain_pred_l1 = nn.DataParallel(self.domain_pred_l1)
        self.domain_pred_l2 = nn.DataParallel(self.domain_pred_l2)
        self.domain_pred_l3 = nn.DataParallel(self.domain_pred_l3)

    def forward(self, cond, x1, x2=None, l=None, dom_label=None, 
                init_weight=None, init_w_main=None, init_w_l1=None,
                init_w_l2=None, init_w_l3=None,):

        base1 = self.conv1(x1)
        base1 = self.bn1(base1)
        base1 = self.relu(base1)
        base1 = self.maxpool(base1)
        l1 = self.layer1(base1)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        process = self.process(l4).view(l4.size(0), -1)


        if (cond == 'source_train' or cond == 'pretrain' or cond == 'source_pseudo_train'):

            bottle = self.source_bottleneck(process)
            class_pred = self.source_classifier(bottle)

            base2 = self.conv1(x2)
            base2 = self.bn1(base2)
            base2 = self.relu(base2)
            base2 = self.maxpool(base2)
            l1_2 = self.layer1(base2)
            l2_2 = self.layer2(l1_2)
            l3_2 = self.layer3(l2_2)
            l4_2 = self.layer4(l3_2)

            process_2 = self.process(l4_2)
            process_2 = process_2.view(l4_2.size(0), -1)
            bottle_2 = self.source_bottleneck(process_2)
            grad_inverse_hook_2 = bottle_2.register_hook(lambda grad: grad * -1*l)

            process_l1 = self.process_l1(l1_2).view(l1_2.size(0), -1)
            bottle_l1 = self.l1_bottleneck(process_l1)

            process_l2 = self.process_l2(l2_2).view(l2_2.size(0), -1)
            bottle_l2 = self.l2_bottleneck(process_l2)

            process_l3 = self.process_l3(l3_2).view(l3_2.size(0), -1)
            bottle_l3 = self.l3_bottleneck(process_l3)

            disc_main, disc_l1, disc_l2, disc_l3, l1_rev, l2_rev, l3_rev = self.disc_weight(init_w_main, init_w_l1,
                                                                    init_w_l2, init_w_l3)

            grad_hook_l1 = bottle_l1.register_hook(lambda grad: grad *l*l1_rev)
            grad_hook_l2 = bottle_l2.register_hook(lambda grad: grad *l*l2_rev)
            grad_hook_l3 = bottle_l3.register_hook(lambda grad: grad *l*l3_rev)

            dom_pred = self.domain_pred(bottle_2)
            dom_pred_l1 = self.domain_pred_l1(bottle_l1)
            dom_pred_l2 = self.domain_pred_l2(bottle_l2)
            dom_pred_l3 = self.domain_pred_l3(bottle_l3)

            # print("process: ", process.grad)

            return class_pred, dom_pred.squeeze(), dom_pred_l1.squeeze(), \
                               dom_pred_l2.squeeze(), dom_pred_l3.squeeze(), \
                               disc_main, disc_l1, disc_l2, disc_l3, l1_rev, l2_rev, l3_rev


        elif (cond == 'pseudo_discriminator'):

            bottle_2 = self.source_bottleneck(process)
            class_pred = self.source_classifier(bottle_2)
            dom_pred = self.domain_pred(bottle_2)

            disc_w, weight, bias = self.disc_activate(dom_pred, dom_label, init_weight)

            return class_pred, dom_pred.squeeze(), disc_w, weight, bias

        else:
            # test and pseudo label

            bottle = self.source_bottleneck(process)
            class_pred = self.source_classifier(bottle)

            return class_pred
