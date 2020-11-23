from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn    as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn    as cudnn
from torch.autograd import Variable
import math
import util
import classifier
import classifier2
import sys
import model
import time
import scipy.io    as sio
import datetime
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter

i = datetime.datetime.now()
time_str = time.strftime('%m%d', time.localtime())
time_str1 = time.strftime('%Y%m%d_%H%M%S', time.localtime())
timestr = time.strftime('%m%d%H%M%S', time.localtime())

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2', help='AWA2')
parser.add_argument('--dataroot', default='./data', help='path	to dataset')
parser.add_argument('--matdataset', default=True, help='Data in	matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=200, help='number features to generate per class')
parser.add_argument('--syn_num_train', type=int, default=200, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False,
                    help='enbale	MinMaxScaler on	visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number	of data	loading	workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch	size')
parser.add_argument('--resSize', type=int, default=2048, help='size	of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size	of semantic	features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent	z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size	of the hidden units	in generator')
parser.add_argument('--nch', type=int, default=1024, help='size	of the hidden units	in quality classifier')
parser.add_argument('--nepoch', type=int, default=2000, help='number of	epochs to train	for')
parser.add_argument('--epoch_q', type=int, default=2000, help='number of	epochs to train	quality classifier')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate	to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1	for	adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables	cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number	of GPUs	to use')
parser.add_argument('--pretrain_classifier', default='',
                    help="path	to pretrain	classifier (to continue	training)")
parser.add_argument('--netG', default='', help="path to	netG (to continue training)")
parser.add_argument('--netC', default='', help="path to	netC (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--checkpoint_path', default='./checkpoint/', help='folder	to output data and model checkpoints')
parser.add_argument('--log_path', default='./logs/', help='folder	to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number	of all classes')
parser.add_argument('--nclass_train', type=int, default=200, help='number	of training classes')
parser.add_argument('--need_teacher', action='store_true', default=False, help='need_teacher')
parser.add_argument('--teacher_type', type=str, default='seen_classes', help='seen_classes/ all_classes')
parser.add_argument('--val_split', type=float, default=0.25, help='number	of all classes')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.checkpoint_path)
    os.makedirs(opt.log_path)


except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed:	", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a	CUDA device, so	you	should probably	run	with --cuda")

if opt.gzsl:
    mode = 'GZSL'
else:
    mode = 'ZSL'

# load data
data = util.DATA_LOADER(opt)
# print("# of	training samples: ", data.ntrain)

# initialize generator and quality classifier
netG = model.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netC = model.MLP_C(opt)
if opt.netC != '':
    netC.load_state_dict(torch.load(opt.netC))
print(netC)

input_label = torch.LongTensor(data.ntrain_class * opt.syn_num_train)
input_att = torch.FloatTensor(data.ntrain_class * opt.syn_num_train, opt.attSize)
noise = torch.FloatTensor(data.ntrain_class * opt.syn_num_train, opt.nz)

input_feature = torch.FloatTensor(opt.batch_size, opt.resSize)
quality_label = torch.FloatTensor(opt.batch_size)

cls_criterion = nn.CrossEntropyLoss()

if opt.cuda:
    netG.cuda()
    netC.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    input_label = input_label.cuda()
    input_feature = input_feature.cuda()
    quality_label = quality_label.cuda()
    cls_criterion.cuda()


def sample():
    batch_feature, batch_label = next_batch_quality(opt.batch_size, fake_feature_train, label_q_train)
    input_feature.copy_(batch_feature)
    quality_label.copy_(batch_label)


def load_data():
    train_label, attribute, train_feature = data.load_dataset()
    input_att.copy_(attribute)
    input_label.copy_(util.map_label(train_label, data.seenclasses))


def get_att_and_label(classes, attribute, num):
    nclass = classes.size(0)

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]

        att = iclass_att.repeat(num, 1)
        input_att.narrow(0, i * num, num).copy_(att)
        input_label.narrow(0, i * num, num).copy_(iclass)

    input_label.copy_(util.map_label(input_label, data.seenclasses))


def next_batch_quality(batch_size, feature, label):
    idx = torch.randperm(feature.shape[0])[0:batch_size]
    batch_label = label[idx]
    batch_feature = feature[idx]
    return batch_feature, batch_label


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise, requires_grad=False), Variable(syn_att, requires_grad=False))
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label


def save_teaching_model(model, root):
    save_path = os.path.join(root, opt.teacher_type, opt.dataset, time_str, time_str1)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model, os.path.join(save_path, 'classifier.pth'))

    print("	[*]	save pretrain model SUCCESS")


def save_model(model, root):
    save_path = os.path.join(root, opt.dataset, time_str, time_str1)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model, os.path.join(save_path, 'model.pth'))
    print("	[*]	save model SUCCESS")


def load_model(root):
    loadpath = os.path.join('./' + root, opt.dataset, '1106', '20201106_144445')
    print(os.path.join(loadpath, 'classifier.pth'))
    torch.load(os.path.join(loadpath, 'classifier.pth'))


def save_log(path, mode):
    log_dir = os.path.join(path, mode, opt.dataset, time_str, time_str1)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    return SummaryWriter(log_dir)


def compute_acc(real_label, predicted_label):
    index = (real_label == predicted_label.T)

    total_num = torch.tensor(real_label.size())
    # print('total_num',total_num)
    print('Correct labeled Number:', torch.sum(index).float())
    acc = torch.sum(index).float() / total_num.float()
    # print('acc:  ',acc)
    return acc


# setup	optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerC = optim.Adam(netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

criterion_G = nn.CrossEntropyLoss()
criterion_C = nn.CrossEntropyLoss()
criterion_Cq = nn.BCEWithLogitsLoss()

######  summary writer
writer = save_log(opt.log_path, mode)

# train	a classifier on	seen classes, obtain teacher

# print('train_feature', data.train_feature.shape)

if opt.need_teacher:
    if opt.teacher_type == 'seen_classes':
        print('pretrain classifier for seen classes!')

        pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                             data.test_seen_feature, data.test_seen_label, data.seenclasses, data,
                                             data.seenclasses.size(0),
                                             opt.resSize, opt.cuda, 0.001, 0.5, 100, 100, opt.pretrain_classifier,
                                             opt.teacher_type)
    elif opt.teacher_type == 'all_classes':
        print('pretrain classifier for all classes!')

        #############  train/test split  4/1
        feature_all = torch.cat((data.train_feature, data.test_seen_feature, data.test_unseen_feature), 0)
        label_all = torch.cat((data.train_label, data.test_seen_label, data.test_unseen_label), 0)

        data_num = label_all.shape[0]
        print('all_data_num', data_num)
        indices = list(range(data_num))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[int(data_num * opt.val_split):], indices[:int(data_num * opt.val_split)]
        # print('train_indices',train_indices)
        # print('val_indeces',val_indeces)

        train_feature = feature_all[train_indices]
        train_label = label_all[train_indices]
        test_feature = feature_all[val_indices]
        test_label = label_all[val_indices]

        pretrain_cls = classifier.CLASSIFIER(train_feature, train_label,
                                             test_feature, test_label, data.allclasses, data, opt.nclass_all,
                                             opt.resSize, opt.cuda, 0.001, 0.5, 100, 100, opt.pretrain_classifier,
                                             opt.teacher_type)

    acc = pretrain_cls.acc
    print(' accuracy= ', acc)
    save_teaching_model(pretrain_cls, './classifier_weight')


else:

    loadpath = os.path.join('./classifier_weight', opt.teacher_type, opt.dataset, '1115', '20201115_144529',
                            'classifier.pth')
    pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                         data.test_seen_feature, data.test_seen_label, data.seenclasses, data,
                                         data.seenclasses.size(0),
                                         opt.resSize, opt.cuda, 0.001, 0.5, 100, 100, loadpath, opt.teacher_type)
    # pretrain_cls= load_model('classifier_weight')
    print('pretrain_cls', pretrain_cls)
    print("	[*]	load pretrain model SUCCESS")

# freeze the classifier	during the optimization
for p in pretrain_cls.model.parameters():  # set	requires_grad to False
    p.requires_grad = False

get_att_and_label(data.seenclasses, data.attribute, opt.syn_num_train)
netG.train()

for epoch in range(opt.nepoch):

    print('****************** Epoch  %d *********************************' % (epoch))

    #############################
    #	(1)	 G network
    ###########################
    # load_data()

    #netG.zero_grad()
    input_attv = Variable(input_att)
    noise.normal_(0, 1)
    noisev = Variable(noise)
    fake_feature = netG(noisev, input_attv)
    #	classification result

    cls_result = F.softmax(pretrain_cls.model(fake_feature))
    g_loss = criterion_G(cls_result, input_label)
    print("g_loss = ", g_loss)
    g_loss.backward()
    optimizerG.step()

    # print('cls_result',cls_result)
    # print('cls_result',cls_result.shape)

    _, predict_label = torch.max(cls_result, 1)
    # print('predict_label',predict_label)
    # print('input_label',input_label)
    label_q = (predict_label == input_label)  ##### label for quality classification
    # print('label_q',label_q)
    right_num = torch.sum(label_q) #TODO test optimising netG to maximise this number
    print('right_num',right_num)

    #right_ratio = right_num/8000

    #right_ratio.backward()

    ###########  Too much negative sample, choose a part of negative sample  ###################

    idx_pos = torch.squeeze((label_q == 1).nonzero())
    # print('idx_pos',idx_pos)
    # print('idx_pos',idx_pos.shape)

    idx_neg = torch.squeeze((label_q == 0).nonzero())  ### mislabeled index
    # print('idx_neg',idx_neg.shape)

    label_pos = label_q[idx_pos]
    fake_feat_pos = fake_feature[idx_pos]
    # print('label_pos',label_pos.shape)
    # print('fake_feat_pos',fake_feat_pos.shape)

    fake_pos_num = fake_feat_pos.shape[0]
    fake_neg_num = idx_neg.shape[0]

    neg_indices = list(range(fake_neg_num))
    np.random.shuffle(neg_indices)
    idx_neg_chosen = neg_indices[:int(fake_pos_num * 1.5)]
    # print('idx_neg_chosen',idx_neg_chosen)
    # print('idx_neg_chosen',idx_neg_chosen.shape)

    label_neg = label_q[idx_neg_chosen]
    fake_feat_neg = fake_feature[idx_neg_chosen]
    # print('label_neg',label_neg.shape)
    # print('fake_feat_neg',fake_feat_neg.shape)

    ######  concat pos and neg sample

    fake_feat_chosen = torch.cat((fake_feat_pos, fake_feat_neg), 0)
    label_q_chosen = torch.cat((label_pos, label_neg), 0)

    ####################################################

    print('Epoch: [%d/%d],  [right_num/generated_num]:   [%d/%d]  ' % (
    epoch, opt.nepoch, right_num, fake_feature.shape[0]))
    print("---------------------------------------")




    ############################
    #	(2)	train quality classifier
    ############################

    netC.zero_grad()

    for p in netC.parameters():  # set	requires_grad to False
        p.requires_grad = True

    for epoch_q in range(opt.epoch_q):

        ################### 2.   random split 4/1
        data_num = fake_feat_chosen.shape[0]
        # print('data_num',data_num)
        indices = list(range(data_num))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[int(data_num * opt.val_split):], indices[:int(data_num * opt.val_split)]
        # print('train_indices',train_indices)
        # print('val_indeces',val_indeces)

        print('train_dataset / test_dataset:  [%d/%d]  ' % (
        int(data_num * (1 - opt.val_split)), int(data_num * opt.val_split)))

        label_q_train = label_q_chosen[train_indices]
        fake_feature_train = fake_feat_chosen[train_indices]
        label_q_test = label_q_chosen[val_indices]
        fake_feature_test = fake_feat_chosen[val_indices]
        # print('label_q_train',label_q_train.shape)
        # print('label_q_test',label_q_test.shape)
        # print('fake_feature_train',fake_feature_train.shape)
        # print('fake_feature_test',fake_feature_test.shape)
        class_error = 0
        qual_error = 0
        for i in range(0, data.ntrain, opt.batch_size):
            sample()
            input_feat = Variable(input_feature)
            pred_label, q = netC(input_feat)
            pred_label = pred_label.squeeze()
            q = torch.squeeze(q)  ### tensor dimension
            # print('pred_label',pred_label)
            # print('pred_label',pred_label.shape)
            #errC = criterion_C(pred_label, label_q_train)
            errQ = criterion_Cq(q, quality_label)
            #errQ = criterion_Cq(, qual)

            # print('errC',errC)
            # time.sleep(10)

            #err = errC + errQ
            err = errQ

            err.backward()
            optimizerC.step()
            #optimizerG.step()

        print('[%d/%d]	Loss_C:	%.4f' % (epoch_q, opt.epoch_q, errQ))
        print("---------------------------------------")
        # time.sleep(3)

        # evaluate	the	model, set C to	evaluation mode
        netC.eval()

        pred_tlabel, _ = netC(fake_feature_test)
        pred_tlabel = torch.squeeze(pred_tlabel).float()
        # print('pred_tlabel',pred_tlabel)
        pred_testlabel = (pred_tlabel >= 0.5)

        # print('pred_testlabel',pred_testlabel)
        # print(' Number of Pos Sapmle:',torch.sum(pred_testlabel))
        # print('Number of test sample: ',label_q_test.shape)

        cls_acc = compute_acc(label_q_test, pred_testlabel)
        print('quality classification accuracy=   %.4f' % (cls_acc))

        if errQ <= 0.0001 or cls_acc >= 99.99:
            print('Finish quality classifier training and turn to nex step !')
            print('')
            print('')
            time.sleep(2)
            break

        else:
            # reset C to training mode
            netC.train()


