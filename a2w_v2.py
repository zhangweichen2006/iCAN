from __future__ import print_function, division

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.model_zoo as model_zoo

import matplotlib
from matplotlib.offsetbox import AnchoredText
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import time, math
import copy
import os,errno
# import bisect
from operator import itemgetter
from ican_v2 import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description="iCAN")
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path of the checkpoint')

BATCH_SIZE = 16
NUM_CLASS = 31
BASE_LR = 0.0015

PRETRAIN_SAMPLE = 60000
TRAIN_SAMPLE = 240000

# int(EPOCHS) / 10

INI_CONFID_THRESH = 0.8

# LOW_DISCRIM_THRESH_T = 0.495
# UP_DISCRIM_THRESH_T = 0.505
INI_DISCRIM_THRESH_LAMBDA = 0.005

LOW_DISCRIM_THRESH_S = 0.0
UP_DISCRIM_THRESH_S = 1.0

INI_L1_THRESH = 0.4/3
INI_L2_THRESH = 0.4/3
INI_L3_THRESH = 0.4/3
INI_MAIN_THRESH = -0.8
PRETRAIN_THRESH = 0

VERSION = 'EVO_16_pre'

######################################################################
# Load Data

# source_set = 'webcam'
# target_set = 'amazon'
#
source_set = 'amazon'
target_set = 'webcam'

print('Office-31: ' + source_set + ' To ' + target_set)

data_transforms = {
    'source': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'target': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/home/wzha8158/datasets/Office/domain_adaptation_images/'
save_dir = './models/'

dsets = {}
dsets['source'] = datasets.ImageFolder(data_dir+source_set+'/images', data_transforms['source'])
dsets['target'] = datasets.ImageFolder(data_dir+target_set+'/images', data_transforms['target'])
dsets['target_test'] = datasets.ImageFolder(data_dir+target_set+'/images', data_transforms['test'])
dsets['pseudo_source'] = datasets.ImageFolder(data_dir+source_set+'/images', data_transforms['source'])
dsets['pseudo'] = []

dsets['test'] = datasets.ImageFolder(data_dir+target_set+'/images', data_transforms['test'])

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=int(BATCH_SIZE / 2),
                shuffle=True) for x in ['source', 'target', 'pseudo_source', 'test']}

source_batches_per_epoch = np.floor(len(dsets['source']) * 2 / BATCH_SIZE).astype(np.int16)
target_batches_per_epoch = np.floor(len(dsets['target']) * 2 / BATCH_SIZE).astype(np.int16)

pre_epochs = int(PRETRAIN_SAMPLE / len(dsets['source']))
total_epochs = int(pre_epochs + TRAIN_SAMPLE / len(dsets['source']))
 # + 20

######################################################################
# Finetuning the convnet
model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
                'densenet161': 'https://download.pytorch.org/models/densenet161-17b70270.pth',
                'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
                'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',}

def train_model(model, optimizer, pseudo_optimizer, lr_scheduler, pseu_lr_scheduler, num_epochs=500, start_epoch=0, best_acc=0, loaded_model=False, draw_dict=None):
    since = time.time()

    # ----- initialise variables ----

    best_model_wts = model.state_dict()

    step_count = 0

    epoch_acc = 0.0
    epoch_acc_s = 0.0
    epoch_lr_mult = 0.0

    target_dict = [(i,j) for (i,j) in dset_loaders['target']]
    pseudo_source_dict = [(i,j) for (i,j) in dset_loaders['pseudo_source']]

    # for inputs, labels in dset_loaders['train']:
    #     inputs_var, labels_var = Variable(inputs.cuda()), Variable(labels.cuda())
    #     inputs_list, labels_list = inputs.numpy(), labels.numpy().tolist()
    #     for i in range(len(inputs_list)):
    #         Source_set.append((torch.from_numpy(np.array(inputs_list[i])), labels_list[i]))

    # ------------------------- two model train ----------------------------------

    for epoch in range(start_epoch, num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        draw_dict['epoch_point'] = [i for i in range(epoch+1)]

        # ----------------------------------------------------------
        # --------------- Training and Testing Phase ---------------
        # ----------------------------------------------------------

        for phase in ['train', 'test']:

            # ----- initialise common variables -----
            epoch_loss = 0.0
            epoch_corrects = 0

            # ----------------------------------------------------------
            # ------------------ Training Phase ------------------------
            # ----------------------------------------------------------

            if phase == 'train':

                # ----- initialise common variables -----

                domain_epoch_loss = 0.0
                total_epoch_loss = 0.0
                domain_epoch_corrects = 0
                epoch_discrim_lambda = copy.deepcopy(INI_DISCRIM_THRESH_LAMBDA)
                epoch_discrim_bias = 0.0

                target_pointer = 0
                pseudo_pointer = 0
                pseudo_source_pointer = 0

                batch_count = 0
                class_count = 0
                domain_counts = 0

                total_iters = 0

                domain_epoch_loss_l1 = 0.0
                domain_epoch_loss_l2 = 0.0
                domain_epoch_loss_l3 = 0.0

                domain_epoch_corrects_l1 = 0
                domain_epoch_corrects_l2 = 0
                domain_epoch_corrects_l3 = 0

                w_main = 0.0 
                w_l1 = 0.0 
                w_l2 = 0.0 
                w_l3 = 0.0 

                confid_threshold = copy.deepcopy(INI_CONFID_THRESH)

                # -----------------------------------------------
                # ----------------- Pre-train -------------------
                # -----------------------------------------------

                if epoch < pre_epochs:

                    source_batchsize = int(BATCH_SIZE / 2)

                    total_iters = source_batches_per_epoch * total_epochs

                else:
                    # -----------------------------------------------
                    # ------------ Pseudo Labelling -----------------
                    # -----------------------------------------------

                    # ------- with iterative pseudo sample filter --------

                    model.train(False)
                    model.eval()

                    Pseudo_set = []

                    # base on the results of last epoch's test accuracy

                    confid_threshold = 1 / (1 + np.exp(-2.4*epoch_acc_s))

                    dset_loaders['target_test'] = torch.utils.data.DataLoader(dsets['target_test'],
                                                        batch_size=1, shuffle=True)

                    total_pseudo_errors = 0

                    for target_inputs, target_labels in dset_loaders['target_test']:

                        target_inputs = target_inputs.cuda()

                        domain_labels_t = torch.FloatTensor([0.]*len(target_inputs)).cuda()

                        class_t, domain_out_t, disc_weight_t, w_t, b_t = model('pseudo_discriminator', 
                                                                               x1=target_inputs,dom_label=domain_labels_t)
                        _, preds_t = torch.max(class_t, 1)

                        # for i in range(len(class_t)):

                        if disc_weight_t[0] > b_t:
                            top_prob, top_label = torch.topk(F.softmax(class_t,dim=1)[0], 1)
                            if float(top_prob[0]) >= float(confid_threshold):
                                s_tuple = (target_inputs[0].cpu(), (top_label[0].cpu(), float(disc_weight_t)))
                                Pseudo_set.append(s_tuple)

                                # ---- no training, just for testing the number of error pseudos
                                total_pseudo_errors += int(top_label[0]) != int(target_labels)

                    # ----------------- reload pseudo set ---------------------------

                    dsets['pseudo'] = Pseudo_set

                    print("Pseudo error/total = {}/{}, confid_threshold: {:.4f}".format(total_pseudo_errors, len(Pseudo_set), 
                                                                                        confid_threshold)) 

                    draw_dict['confid_threshold_point'].append(float("%.4f" % confid_threshold))

                    # -----------------------------------------------
                    # -------- Pseudo Threshold Training ------------
                    # -----------------------------------------------

                    if (len(dsets['pseudo']) > 0):

                        model.disc_activate.train(True)

                        for param in model.parameters():
                            param.requires_grad = False

                        for param in model.disc_activate.parameters():
                            param.requires_grad = True

                        pseudo_batch_count = 0
                        pseudo_sample_count = 0
                        pseudo_epoch_loss = 0.0
                        pseudo_epoch_acc = 0
                        pseudo_epoch_corrects = 0
                        pseudo_avg_loss = 0.0

                        dset_loaders['pseudo'] = torch.utils.data.DataLoader(dsets['pseudo'],
                                                            batch_size=int(BATCH_SIZE / 2), shuffle=True)

                        for pseudo_inputs, pseudo_labels in dset_loaders['pseudo']:

                            pseudo_batch_count += 1
                            pseudo_sample_count += len(pseudo_inputs)
                            # if (pseudo_batch_count * BATCH_SIZE / 2 > len(dsets['pseudo'])):
                            #     continue

                            pseudo_labels, pseudo_weights = pseudo_labels[0], pseudo_labels[1]

                            pseudo_p = (epoch - pre_epochs) / (num_epochs - pre_epochs)
                            pseudo_lr_mult = (1. + 10 * pseudo_p)**(-0.75) 
                            pseudo_optimizer, pseudo_epoch_lr_mult = pseu_lr_scheduler(pseudo_optimizer, pseudo_lr_mult, 
                                                                        200*(len(dsets['source']) / len(dsets['target'])))

                            pseudo_inputs, pseudo_labels = pseudo_inputs.cuda(), pseudo_labels.cuda()
                            # pseudo_weights = pseudo_weights.cuda().squeeze()

                            domain_labels = torch.FloatTensor([0.]*len(pseudo_inputs)).cuda()

                            pseudo_class, pseudo_domain_out, pseudo_disc_weight, pseudo_ww, pseudo_bb = model(
                                            'pseudo_discriminator',x1=pseudo_inputs,dom_label=domain_labels)

                            pseudo_optimizer.zero_grad()

                            _, pseudo_preds = torch.max(pseudo_class, 1)

                            pseudo_class_loss = compute_new_loss(pseudo_class, pseudo_labels, pseudo_disc_weight)

                            pseudo_epoch_loss += float(pseudo_class_loss)
                            pseudo_epoch_corrects += int(torch.sum(pseudo_preds.squeeze() == pseudo_labels.squeeze()))

                            pseudo_loss = pseudo_class_loss

                            pseudo_loss.backward()
                            pseudo_optimizer.step()

                            epoch_discrim_lambda = 1.0 / (abs(pseudo_ww) ** (1. / 4))
                            epoch_discrim_bias = pseudo_bb

                        pseudo_avg_loss = pseudo_epoch_loss / pseudo_batch_count
                        pseudo_epoch_acc = pseudo_epoch_corrects / pseudo_sample_count

                        print('Phase: {} Lr: {:.4f} Loss: {:.4f} Acc: {:.4f} Disc_Lam: {:.6f} Disc_bias: {:.4f} '.format(
                                'Pseudo_train', pseudo_epoch_lr_mult, pseudo_avg_loss, pseudo_epoch_acc, epoch_discrim_lambda, epoch_discrim_bias))

                        for param in model.parameters():
                            param.requires_grad = True

                        for param in model.disc_activate.parameters():
                            param.requires_grad = False

                    # ---------------------------------------------------
                    # ------- Source + Pseudo Dataset Preparation -------
                    # ---------------------------------------------------

                    # -------- reset source and pseudo batch ratio -------

                    source_size = len(dsets['source'])
                    pseudo_size = len(dsets['pseudo'])

                    if pseudo_size == 0:
                        dset_loaders['pseudo'] = []
                        dset_loaders['pseudo_source'] = []
                        source_batchsize = int(BATCH_SIZE / 2)
                        pseudo_batchsize = 0
                    else:
                        source_batchsize = int(int(BATCH_SIZE / 2) * source_size
                                                    / (source_size + pseudo_size))

                        if source_batchsize == int(BATCH_SIZE / 2):
                            source_batchsize -= 1

                        if source_batchsize < int(int(BATCH_SIZE / 2) / 2):
                            source_batchsize = int(int(BATCH_SIZE / 2) / 2)

                        pseudo_batchsize = int(BATCH_SIZE / 2) - source_batchsize

                        # print('Source,Pseudo ratio: %d'%(source_batchsize),',%d'%(pseudo_batchsize))

                        dset_loaders['pseudo'] = torch.utils.data.DataLoader(dsets['pseudo'],
                                                        batch_size=pseudo_batchsize, shuffle=True)
                        dset_loaders['pseudo_source'] = torch.utils.data.DataLoader(dsets['pseudo_source'],
                                                        batch_size=pseudo_batchsize, shuffle=True)

                    dset_loaders['source'] = torch.utils.data.DataLoader(dsets['source'],
                                                        batch_size=source_batchsize, shuffle=True)
                    dset_loaders['target'] = torch.utils.data.DataLoader(dsets['target'],
                                                        batch_size=source_batchsize, shuffle=True)

                    target_dict = [(i,j) for (i,j) in dset_loaders['target']]
                    if pseudo_size > 0:
                        pseudo_dict = [(i,j) for (i,j) in dset_loaders['pseudo']]
                        pseudo_source_dict = [(i,j) for (i,j) in dset_loaders['pseudo_source']]
                    else:
                        pseudo_dict = []
                        pseudo_source_dict = []

                    total_iters = source_batches_per_epoch * pre_epochs + \
                                  source_batches_per_epoch * (total_epochs - pre_epochs) * \
                                  BATCH_SIZE / (source_batchsize * 2)
                # -----------------------------------------------
                # --------------- Training Process --------------
                # -----------------------------------------------

                # -------- loop through source dataset ---------

                model.train(True) # Set model to training mode

                for param in model.parameters():
                    param.requires_grad = True

                for param in model.disc_activate.parameters():
                    param.requires_grad = False

                if epoch < pre_epochs:
                    for param in model.disc_weight.parameters():
                        param.requires_grad = False

                ini_w_main = torch.FloatTensor([float(INI_MAIN_THRESH)]).cuda()
                ini_w_l1 = torch.FloatTensor([float(INI_L1_THRESH)]).cuda()
                ini_w_l2 = torch.FloatTensor([float(INI_L2_THRESH)]).cuda()
                ini_w_l3 = torch.FloatTensor([float(INI_L3_THRESH)]).cuda()
                epoch_batch_count = 0

                for data in dset_loaders['source']:
                    # get the inputs
                    inputs, labels = data

                    step_count += 1
                    # iterations could be changed
                    p = step_count / total_iters
                    l = (2. / (1. + np.exp(-10. * p))) - 1
                    if (epoch == 0):
                        lr_mult = 1 / (1 + np.exp(-3*(step_count / len(dsets['source']))))
                    else:
                        lr_mult = (1. + 10 * p)**(-0.75)
                    # 
                    weight_mult = 0.055**p
                    optimizer, epoch_lr_mult = lr_scheduler(optimizer, lr_mult, weight_mult)

                    # skip the last batch if the length is not enough
                    batch_count += 1
                    if (batch_count * source_batchsize > len(dsets['source'])):
                        continue

                    # ---------------- reset exceeded datasets --------------------

                    if target_pointer >= len(target_dict) - 1:
                        target_pointer = 0
                        target_dict = [(i,j) for (i,j) in dset_loaders['target']]

                    target_inputs = target_dict[target_pointer][0]

                    if epoch < pre_epochs:

                        # -------------------- pretrain model -----------------------

                        domain_inputs = torch.cat((inputs, target_inputs),0)
                        domain_labels = torch.FloatTensor([1.]*int(BATCH_SIZE / 2)
                                                         +[0.]*int(BATCH_SIZE / 2))

                        # Wrap domain inputs and labels (Source and Target)

                        inputs, labels = inputs.cuda(), labels.cuda()
                        domain_inputs, domain_labels = domain_inputs.cuda(), domain_labels.cuda()

                        class_outputs, domain_outputs, domain_outputs_l1, domain_outputs_l2, \
                                domain_outputs_l3, w_main, w_l1, w_l2, w_l3, l1_rev, l2_rev, l3_rev\
                                             = model('pretrain', x1=inputs, x2=domain_inputs, l=l,\
                                                     init_w_main=ini_w_main, init_w_l1=ini_w_l1, \
                                                     init_w_l2=ini_w_l2, init_w_l3=ini_w_l3)

                        # print("-----------------")

                        target_pointer += 1

                        epoch_discrim_bias = 0.5

                        # ------------ training classification statistics --------------
                        criterion = nn.CrossEntropyLoss()


                        _, preds = torch.max(class_outputs, 1)
                        class_count += len(preds)
                        class_loss = criterion(class_outputs, labels)
                        epoch_loss += float(class_loss)
                        epoch_corrects += int(torch.sum(preds.squeeze() == labels.squeeze()))
                        
                    else:

                        # -------------- train with pseudo sample model -------------
                        pseudo_weights = torch.FloatTensor([])

                        if (pseudo_pointer >= len(pseudo_dict) - 1) and (len(pseudo_dict) != 0) :
                            pseudo_pointer = 0
                            pseudo_dict = [(i,j) for (i,j) in dset_loaders['pseudo']]

                        if (pseudo_source_pointer >= len(pseudo_source_dict) - 1) and (len(pseudo_source_dict) != 0):
                            pseudo_source_pointer = 0
                            pseudo_source_dict = [(i,j) for (i,j) in dset_loaders['pseudo_source']]

                        # Wrap domain inputs and labels (Source + Pseudo(Target) and Target+Source)

                        if pseudo_size == 0:
                            domain_inputs = torch.cat((inputs, target_inputs),0)
                            domain_labels = torch.FloatTensor([1.]*int(BATCH_SIZE / 2)+
                                                              [0.]*int(BATCH_SIZE / 2))

                            fuse_inputs = inputs
                            fuse_labels = labels

                        else:
                            pseudo_inputs, pseudo_labels, pseudo_weights = pseudo_dict[pseudo_pointer][0], \
                                          pseudo_dict[pseudo_pointer][1][0], pseudo_dict[pseudo_pointer][1][1]
                            pseudo_source_inputs = pseudo_source_dict[pseudo_source_pointer][0]

                            domain_inputs = torch.cat((inputs, pseudo_inputs, target_inputs, pseudo_source_inputs),0)
                            domain_labels = torch.FloatTensor([1.]*source_batchsize+[0.]*pseudo_batchsize+
                                                              [0.]*source_batchsize+[1.]*pseudo_batchsize)

                            fuse_inputs = torch.cat((inputs, pseudo_inputs),0)
                            fuse_labels = torch.cat((labels, pseudo_labels),0)

                        inputs, labels = fuse_inputs.cuda(), fuse_labels.cuda()
                        domain_inputs, domain_labels = domain_inputs.cuda(), domain_labels.cuda()

                        source_weight_tensor = torch.FloatTensor([1.]*source_batchsize)
                        pseudo_weights_tensor = pseudo_weights.float()
                        class_weights_tensor = torch.cat((source_weight_tensor, pseudo_weights_tensor),0)
                        dom_weights_tensor = torch.FloatTensor([0.]*source_batchsize+[1.]*pseudo_batchsize)

                        ini_weight = torch.cat((class_weights_tensor, dom_weights_tensor),0).squeeze().cuda()

                        class_outputs, domain_outputs, domain_outputs_l1, domain_outputs_l2, \
                            domain_outputs_l3, w_main, w_l1, w_l2, w_l3, l1_rev, l2_rev, l3_rev\
                                = model('source_pseudo_train', x1=inputs, x2=domain_inputs, l=l,\
                                         init_w_main=ini_w_main, init_w_l1=ini_w_l1, \
                                         init_w_l2=ini_w_l2, init_w_l3=ini_w_l3)

                        # ------------ training classification statistics --------------
                        _, preds = torch.max(class_outputs, 1)
                        class_count += len(preds)
                        class_loss = compute_new_loss(class_outputs, labels, ini_weight)

                        epoch_loss += float(class_loss)
                        epoch_corrects += int(torch.sum(preds.squeeze() == labels.squeeze()))

                        target_pointer += 1
                        pseudo_pointer += 1
                        pseudo_source_pointer += 1

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # ----------- calculate pred domain labels and losses -----------

                    domain_criterion = nn.BCEWithLogitsLoss()

                    domain_labels = domain_labels.squeeze()
                    domain_preds = torch.trunc(2*F.sigmoid(domain_outputs))

                    domain_preds_l1 = torch.trunc(2*F.sigmoid(domain_outputs_l1))
                    domain_preds_l2 = torch.trunc(2*F.sigmoid(domain_outputs_l2))
                    domain_preds_l3 = torch.trunc(2*F.sigmoid(domain_outputs_l3))
                    correct_domain = domain_labels

                    # ---------- Pytorch 0.2.0 edit change --------------------------
                    domain_counts += len(domain_preds)
                    domain_epoch_corrects += int(torch.sum(domain_preds == correct_domain))
                    domain_epoch_corrects_l1 += int(torch.sum(domain_preds_l1 == correct_domain))
                    domain_epoch_corrects_l2 += int(torch.sum(domain_preds_l2 == correct_domain))
                    domain_epoch_corrects_l3 += int(torch.sum(domain_preds_l3 == correct_domain))

                    domain_loss = domain_criterion(domain_outputs, domain_labels)
                    domain_loss_l1 = domain_criterion(domain_outputs_l1, domain_labels)
                    domain_loss_l2 = domain_criterion(domain_outputs_l2, domain_labels)
                    domain_loss_l3 = domain_criterion(domain_outputs_l3, domain_labels)

                    domain_epoch_loss += float(domain_loss)
                    domain_epoch_loss_l1 += float(domain_loss_l1)
                    domain_epoch_loss_l2 += float(domain_loss_l2)
                    domain_epoch_loss_l3 += float(domain_loss_l3)

                    # ------ calculate pseudo predicts and losses with weights and threshold lambda -------

                    total_loss = class_loss + w_main*domain_loss+ w_l1*domain_loss_l1 \
                                            + w_l2*domain_loss_l2+ w_l3*domain_loss_l3

                    total_epoch_loss += float(total_loss)

                    #  -------  backward + optimize in training and Pseudo-training phase -------
                    total_loss.backward()
                    optimizer.step()

            # ----------------------------------------------------------
            # ------------------ Testing Phase -------------------------
            # ----------------------------------------------------------

            elif phase == 'test' :
                model.train(False)  # Set model to evaluate mode
                model.eval()

                for data in dset_loaders['test']:
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()
                    class_outputs = model('test', inputs)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # ------------ test classification statistics ------------
                    _, preds = torch.max(class_outputs, 1)

                    criterion = nn.CrossEntropyLoss()
                    class_loss = criterion(class_outputs, labels)
                    epoch_loss += float(class_loss)
                    epoch_corrects += int(torch.sum(preds.squeeze() == labels.squeeze()))

            # ----------------  print statistics results   --------------------

            if phase == 'train':
                epoch_loss = epoch_loss / batch_count
                epoch_acc = epoch_corrects / class_count
                epoch_acc_s = epoch_acc

                domain_avg_loss = domain_epoch_loss / batch_count
                domain_avg_loss_l1 = domain_epoch_loss_l1 / batch_count
                domain_avg_loss_l2 = domain_epoch_loss_l2 / batch_count
                domain_avg_loss_l3 = domain_epoch_loss_l3 / batch_count

                domain_acc = domain_epoch_corrects / domain_counts
                domain_acc_l1 = domain_epoch_corrects_l1 / domain_counts
                domain_acc_l2 = domain_epoch_corrects_l2 / domain_counts
                domain_acc_l3 = domain_epoch_corrects_l3 / domain_counts

                total_avg_loss = total_epoch_loss / batch_count

                print('Phase: {} lr_mult: {:.4f} Loss: {:.4f} D_loss: {:.4f} D1_loss: {:.4f} D2_loss: {:.4f} D3_loss: {:.4f} Acc: {:.4f} D_Acc: {:.4f}'.format(
                      phase, epoch_lr_mult, epoch_loss, domain_avg_loss, 
                      domain_avg_loss_l1, domain_avg_loss_l2, domain_avg_loss_l3, 
                      epoch_acc, domain_acc))
                print("Total loss: {:.4f}, w_main: {:.4f}, 1: {:.4f}, 2: {:.4f}, 3: {:.4f}".format(
                      total_avg_loss, w_main.data[0],w_l1.data[0],w_l2.data[0],w_l3.data[0]))

                draw_dict['class_loss_point'].append(float("%.4f" % epoch_loss))
                draw_dict['domain_loss_point'].append(float("%.4f" % domain_avg_loss))
                draw_dict['source_acc_point'].append(float("%.4f" % epoch_acc))
                draw_dict['domain_acc_point'].append(float("%.4f" % domain_acc))
                draw_dict['lr_point'].append(float("%.4f" % epoch_lr_mult))

                draw_dict['domain_loss_point_l1'].append(float("%.4f" % domain_avg_loss_l1))
                draw_dict['domain_loss_point_l2'].append(float("%.4f" % domain_avg_loss_l2))
                draw_dict['domain_loss_point_l3'].append(float("%.4f" % domain_avg_loss_l3))

                draw_dict['domain_acc_point_l1'].append(float("%.4f" % domain_acc_l1))
                draw_dict['domain_acc_point_l2'].append(float("%.4f" % domain_acc_l2))
                draw_dict['domain_acc_point_l3'].append(float("%.4f" % domain_acc_l3))

            else:
                epoch_loss = epoch_loss / len(dsets['test'])
                epoch_acc = epoch_corrects / len(dsets['test'])
                print('Phase: {} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                draw_dict['target_loss_point'].append(float("%.4f" % epoch_loss))
                draw_dict['target_acc_point'].append(float("%.4f" % epoch_acc))

            # deep copy the model, print best accuracy
            if phase == 'test':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # best_model = copy.deepcopy(model)
                    best_model_wts = model.state_dict()

                    # torch.save(best_model_wts, PATH)
                    # save_model(best_model_wts, os.path.join(save_dir, './%03d_best_model.pth' % epoch))  
                    save_model({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec': best_acc,
                        'draw_dict': draw_dict,
                    }, os.path.join(save_dir, './best_model.pth'))
                
                if epoch == total_epochs - 1:
                    # final_model_wts = model.state_dict()
                    # save_model(final_model_wts, os.path.join(save_dir, './%03d_final_model.pth' % epoch))
                    save_model({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec': best_acc,
                        'draw_dict': draw_dict,
                    }, os.path.join(save_dir, './%03d_final_model.pth' % epoch))

                print('Best Test Accuracy: {:.4f}'.format(best_acc))

        print()

        # ------------------------------------ draw graph ------------------------------------
        # ------------------------------------------------------------------------------------
        if not loaded_model:
            try:
                os.makedirs('./graph/')
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            try:
                os.makedirs('./graph/acc/')
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            # Create plots 0
            fig, ax = plt.subplots()
            ax.plot(draw_dict['epoch_point'], draw_dict['domain_acc_point'],  'k', label='main domain acc',color='r')
            ax.plot(draw_dict['epoch_point'], draw_dict['domain_acc_point_l1'], 'k', label='l1 domain acc',color='g')
            ax.plot(draw_dict['epoch_point'], draw_dict['domain_acc_point_l2'], 'k', label='l2 domain acc',color='b')
            ax.plot(draw_dict['epoch_point'], draw_dict['domain_acc_point_l3'], 'k', label='l3 domain acc',color='y')
            # Now add the legend with some customizations.
            legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., shadow=True)

            # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
            frame = legend.get_frame()
            frame.set_facecolor('0.90')

            # Set the fontsize
            for label in legend.get_texts():
                label.set_fontsize('large')

            for label in legend.get_lines():
                label.set_linewidth(1.5)  # the legend line width

            fig.text(0.5, 0.02, 'EPOCH', ha='center')
            fig.text(0.02, 0.5, 'ACCURACY', va='center', rotation='vertical')

            plt.savefig('graph/'+source_set+'2'+target_set+'_dom_acc_V'+ VERSION +'.png', bbox_inches='tight')

            # ----------------------------------------------------------------

            # Create plots 2
            fig, ax = plt.subplots()
            ax.plot(draw_dict['epoch_point'], draw_dict['source_acc_point'], 'k', label='Source Classification Accuracy',color='r')
            ax.plot(draw_dict['epoch_point'], draw_dict['domain_acc_point'], 'k', label='Domain Accuracy',color='g')
            ax.plot(draw_dict['epoch_point'], draw_dict['target_acc_point'], 'k', label='Test Classification Accuracy',color='b')

            ax.annotate("DANN " + source_set + ' 2 ' + target_set + ' 0.5 Domain', xy=(1.05, 0.7), xycoords='axes fraction')
            ax.annotate('lr: %0.4f Pre epochs: %d Max epochs: %d' % (BASE_LR, pre_epochs, total_epochs), xy=(1.05, 0.65), xycoords='axes fraction')
            # ax.annotate('Pretrain epochs: %d' % PRETRAIN_EPOCH, xy=(1.05, 0.6), xycoords='axes fraction')
            # ax.annotate('Confidence Threshold: %0.3f' % confid_threshold, xy=(1.05, 0.55), xycoords='axes fraction')
            # ax.annotate('Discriminator Threshold: %0.3f ~ %0.3f' % (LOW_DISCRIM_THRESH_T, UP_DISCRIM_THRESH_T), xy=(1.05, 0.5), xycoords='axes fraction')
            ax.annotate('L1,L2,L3,Main Disc_Weight: %0.4f %0.4f %0.4f %0.4f' % \
                        (l1_rev.data[0]*w_l1.data[0], l2_rev.data[0]*w_l2.data[0], l3_rev.data[0]*w_l3.data[0], w_main.data[0]), xy=(1.05, 0.5), xycoords='axes fraction')
            ax.annotate('S_Low,Up, T_Low,Up Threshold: %0.2f %0.2f %0.5f %0.5f' % \
                        (LOW_DISCRIM_THRESH_S, UP_DISCRIM_THRESH_S, (0.5 - epoch_discrim_lambda), (0.5 + epoch_discrim_lambda)), xy=(1.05, 0.45), xycoords='axes fraction')

            if epoch >= 49:
                ax.annotate('50 Epoch Accuracy: %0.4f' % (draw_dict['target_acc_point'][49]), xy=(1.05, 0.35), xycoords='axes fraction')
            if epoch >= 99:
                ax.annotate('50 Epoch Accuracy: %0.4f' % (draw_dict['target_acc_point'][49]), xy=(1.05, 0.35), xycoords='axes fraction')
                ax.annotate('100 Epoch Accuracy: %0.4f' % (draw_dict['target_acc_point'][99]), xy=(1.05, 0.3), xycoords='axes fraction')
            if epoch >= 199:
                ax.annotate('50 Epoch Accuracy: %0.4f' % (draw_dict['target_acc_point'][49]), xy=(1.05, 0.35), xycoords='axes fraction')
                ax.annotate('100 Epoch Accuracy: %0.4f' % (draw_dict['target_acc_point'][99]), xy=(1.05, 0.3), xycoords='axes fraction')
                ax.annotate('200 Epoch Accuracy: %0.4f' % (draw_dict['target_acc_point'][199]), xy=(1.05, 0.25), xycoords='axes fraction')
            if epoch >= 299:
                ax.annotate('50 Epoch Accuracy: %0.4f' % (draw_dict['target_acc_point'][49]), xy=(1.05, 0.35), xycoords='axes fraction')
                ax.annotate('100 Epoch Accuracy: %0.4f' % (draw_dict['target_acc_point'][99]), xy=(1.05, 0.3), xycoords='axes fraction')
                ax.annotate('200 Epoch Accuracy: %0.4f' % (draw_dict['target_acc_point'][199]), xy=(1.05, 0.25), xycoords='axes fraction')
                ax.annotate('300 Epoch Accuracy: %0.4f' % (draw_dict['target_acc_point'][299]), xy=(1.05, 0.2), xycoords='axes fraction')

            ax.annotate('Last Epoch Accuracy: %0.4f' % (epoch_acc), xy=(1.05, 0.1), xycoords='axes fraction', size=14)

            # Now add the legend with some customizations.
            legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., shadow=True)

            # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
            frame = legend.get_frame()
            frame.set_facecolor('0.90')

            # Set the fontsize
            for label in legend.get_texts():
                label.set_fontsize('large')

            for label in legend.get_lines():
                label.set_linewidth(1.5)  # the legend line width

            fig.text(0.5, 0.02, 'EPOCH', ha='center')
            fig.text(0.02, 0.5, 'ACCURACY', va='center', rotation='vertical')

            plt.savefig('graph/acc/'+source_set+'2'+target_set+'_acc_V'+ VERSION +'.png', bbox_inches='tight')

            fig.clf()

            plt.clf()


    time_elapsed = time.time() - since

    try:
        os.makedirs('./Result_txt/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Write to File
    with open('Result_txt/'+source_set+"2"+target_set+"Output_V" + VERSION +'.txt', "w") as text_file:
        text_file.write(str(draw_dict['epoch_point']).strip('[]'))
        text_file.write('\n')
        text_file.write(str(draw_dict['domain_acc_point']).strip('[]'))
        text_file.write('\n')
        text_file.write(str(draw_dict['target_acc_point']).strip('[]'))
        text_file.write('\n')
        text_file.write(str(draw_dict['domain_loss_point']).strip('[]'))
        text_file.write('\n')
        text_file.write(str(draw_dict['target_loss_point']).strip('[]'))
        text_file.write('\n')
        text_file.write(str(draw_dict['lr_point']).strip('[]'))
        text_file.write('\n')

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))



    model.load_state_dict(best_model_wts)

    return model

######################################################################
# Learning rate scheduler
def exp_lr_scheduler(optimizer, lr_mult, weight_mult=1):
    counter = 0
    for param_group in optimizer.param_groups:
        if counter == 0:
            optimizer.param_groups[counter]['lr'] = BASE_LR * lr_mult / 10.0
            # print('{} LR: {:0.6f}'.format('Base', optimizer.param_groups[counter]['lr']))
        elif counter == 1:  
            optimizer.param_groups[counter]['lr'] = BASE_LR * lr_mult * weight_mult
        else:
            optimizer.param_groups[counter]['lr'] = BASE_LR * lr_mult
        counter += 1

    return optimizer, lr_mult

def pseudo_lr_scheduler(optimizer, lr_mult, weight_mult=1):
    counter = 0
    for param_group in optimizer.param_groups:
        optimizer.param_groups[counter]['lr'] = BASE_LR * lr_mult * weight_mult
        counter += 1

    return optimizer, lr_mult


def diff_states(dict_canonical, dict_subset):
    names1, names2 = (list(dict_canonical.keys()), list(dict_subset.keys()))
    #Sanity check that param names overlap
    #Note that params are not necessarily in the same order
    #for every pretrained model
    not_in_1 = [n for n in names1 if n not in names2]
    not_in_2 = [n for n in names2 if n not in names1]
    assert len(not_in_1) == 0
    assert len(not_in_2) == 0

    for name, v1 in dict_canonical.items():
        v2 = dict_subset[name]
        assert hasattr(v2, 'size')
        if v1.size() != v2.size():
            yield (name, v1)

def load_model_merged(name, num_classes):

    model = models.__dict__[name](num_classes=num_classes)

    #Densenets don't (yet) pass on num_classes, hack it in for 169
    if name == 'densenet169':
        model = torchvision.models.DenseNet(num_init_features=64, growth_rate=32, \
                                            block_config=(6, 12, 32, 32), num_classes=num_classes)

    if name == 'densenet201':
        model = torchvision.models.DenseNet(num_init_features=64, growth_rate=32, \
                                            block_config=(6, 12, 48, 32), num_classes=num_classes)
    if name == 'densenet161':
        model = torchvision.models.DenseNet(num_init_features=96, growth_rate=48, \
                                            block_config=(6, 12, 36, 24), num_classes=num_classes)

    pretrained_state = model_zoo.load_url(model_urls[name])

    #Diff
    diff = [s for s in diff_states(model.state_dict(), pretrained_state)]
    print("Replacing the following state from initialized", name, ":", \
          [d[0] for d in diff])

    for name, value in diff:
        pretrained_state[name] = value

    assert len([s for s in diff_states(model.state_dict(), pretrained_state)]) == 0

    #Merge
    model.load_state_dict(pretrained_state)
    return model, diff

def scale_gradients(v, weights): # assumes v is batch x ...
    def hook(g):
        return g*weights.view(*((-1,)+(len(g.size())-1)*(1,))) # probably nicer to hard-code -1,1,...,1
    v.register_hook(hook)

def compute_new_loss(logits, target, weights):
    """ logits: Unnormalized probability for each class.
        target: index of the true class(label)
        weights: weights of weighted loss.
    Returns:
        loss: An average weighted loss value
    """
    # print("l: ",logits)
    # print("t: ",target)
    weights = weights.narrow(0,0,len(target))
    # print("w: ",weights)
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size()) * weights
    # losses = losses * weights
    loss = losses.sum() / len(target)
    # length.float().sum()
    return loss

def save_model(model, path):
    torch.save(model, path)


# model_names = model_urls.keys()

#------------------------ model 1 ------------------------------
def main():
    global args
    args = parser.parse_args()

    model_pretrained, diff = load_model_merged('resnet50', NUM_CLASS)

    model_dann = ICAN(model_pretrained)

    # print(model_dann)

    model_dann = model_dann.cuda()

    for param in model_dann.parameters():
        param.requires_grad = True

    # Observe that all parameters are being optimized
    ignored_params = list(map(id, model_dann.source_bottleneck.parameters()))
    ignored_params.extend(list(map(id, model_dann.source_classifier.parameters())))
    ignored_params.extend(list(map(id, model_dann.domain_pred.parameters())))
    ignored_params.extend(list(map(id, model_dann.disc_weight.parameters())))
    ignored_params.extend(list(map(id, model_dann.disc_activate.parameters())))

    base_params = filter(lambda p: id(p) not in ignored_params,
                         model_dann.parameters())

    if args.resume and os.path.isfile(args.resume):
        loadmodel = True
        print(("=> loading checkpoint '{}'".format(args.resume)))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        draw_dict = checkpoint['draw_dict']
        model_dann.eval()
        model_dann.load_state_dict(checkpoint['state_dict'])
        print(("=> loaded checkpoint epoch {}".format(checkpoint['epoch'])))
    else:
        loadmodel= False
        start_epoch = 0
        best_prec = 0
        draw_dict = {
            "class_loss_point":[],
            "domain_loss_point":[],
            "target_loss_point":[],
            "source_acc_point":[],
            "domain_acc_point":[],
            "target_acc_point":[],
            "set_len_point":[],
            "confid_threshold_point":[],
            "epoch_point":[],
            "lr_point":[],
            "domain_loss_point_l1":[],
            "domain_loss_point_l2":[],
            "domain_loss_point_l3":[],
            "domain_acc_point_l1":[],
            "domain_acc_point_l2":[],
            "domain_acc_point_l3":[],
        }
        print("=> no checkpoint model")

    # print(list(model_dann.parameters()))

    optimizer_ft = optim.SGD([
                {'params': base_params},
                {'params': model_dann.disc_weight.parameters(), 'lr': BASE_LR, 'weight_decay': 0},
                {'params': model_dann.source_bottleneck.parameters(), 'lr': BASE_LR},
                {'params': model_dann.source_classifier.parameters(), 'lr': BASE_LR},
                {'params': model_dann.domain_pred.parameters(), 'lr': BASE_LR},
                ], lr=BASE_LR / 10.0, momentum=0.9, weight_decay=3e-4)
    # , 'weight_decay':5e-4
    optimizer_pseudo = optim.SGD(model_dann.disc_activate.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=3e-4)

    #------------------------- train models ------------------------------

    try:
        os.makedirs('./models/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    model = train_model(model_dann, optimizer_ft, optimizer_pseudo, exp_lr_scheduler, pseudo_lr_scheduler, num_epochs=total_epochs, start_epoch=start_epoch, best_acc=best_prec, loaded_model = loadmodel, draw_dict=draw_dict)

if __name__ == '__main__':
    main()
