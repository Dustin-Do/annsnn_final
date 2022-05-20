import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import models
import json
import modules
from utils import *
import numpy as np
import torch.nn.functional as F
import spikingjelly.clock_driven.functional as functional
import matplotlib.pyplot as plt
import spikingjelly.clock_driven.neuron as neuron

########################################################################################################################
#
# MODEL INITIALIZATION
#
########################################################################################################################
model_name = 'vgg16'
dataset = 'cifar10'

device = 'cuda'
#device = 'cpu' # Duc
optimizer = 'sgd'

momentum = 0.9
lr = 0.1    #leaning rate
schedule = [100, 150]
gammas = [0.1, 0.1]
decay = 1e-4
batch_size = 50
epoch = 5
acc_tolerance = 0.1
lam = 0.1
sharescale = True
scale_init = 2.5


# --------------------------------- File management --------------------------------------------------------------------
conf = [model_name,dataset]
save_name = '_'.join(conf) # 'save_name' = concatenation of all elements of 'conf'
log_dir = 'train_' + save_name
if not os.path.isdir(log_dir): # if 'log_dir' is not a directory
    os.makedirs(log_dir) # create path 'log_dir'

# The SummaryWriter class is your main entry to log data for consumption and visualization by TensorBoard
writer = SummaryWriter(log_dir)
# ----------------------------------------------------------------------------------------------------------------------



# 'load_cv_data' is function defined in 'utils.py'
# datasets are defined: 'cifar10', 'cifar100', 'mnist', 'imagenet'
train_dataloader, test_dataloader = load_cv_data(data_aug=False,
                 batch_size=batch_size,
                 workers=0,
                 dataset=dataset,
                 data_target_dir=datapath[dataset]
                 )
print('type of train_dataloader', type(train_dataloader))
#print('train_dataloader', train_dataloader)
print('type of test_dataloader', type(test_dataloader))
#print('test_dataloader', test_dataloader)


best_acc = 0.0
start_epoch = 0
sum_k = 0.0
cnt_k = 0.0
last_k = 0
best_avg_k = 1e5
train_batch_cnt = 0
test_batch_cnt = 0

# Define the model to be used is 'vgg16'
model = models.__dict__[model_name](num_classes=10, dropout=0)

# 'replace_maxpool2d_by_avgpool2d' & 'replace_relu_by_spikingnorm' are defined in 'modules.py'
model = modules.replace_maxpool2d_by_avgpool2d(model)
model = modules.replace_relu_by_spikingnorm(model,True)

for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)): # if 'm' is 'nn.Conv2d' (AND/OR)? 'nn.Linear'
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #Create weight explicitly by creating a random matrix based on Kaiming initialization
                                                                               #Read more:https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138
        if hasattr(m,'bias') and m.bias is not None: # if 'm' has 'bias' and 'm.bias' is not none
            nn.init.zeros_(m.bias) # set value of 'm.bias' to 0
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, val=1)
        nn.init.zeros_(m.bias)

# --------------------- Define simulating configuration-----------------------------------------------------------------
model.to(device)
device = torch.device(device)
if device.type == 'cuda':
#if device.type == 'cpu': #Duc
    print(f"=> CUDA memory allocated: {torch.cuda.memory_allocated(device.index)}")
# ----------------------------------------------------------------------------------------------------------------------


# Index 'ann_train_module' and 'snn_train_module' as modules which are already defined
ann_train_module = nn.ModuleList()
snn_train_module = nn.ModuleList()
# same as 'divide_trainable_modules' function in 'fast_train.py'
def divide_trainable_modules(model):
    print('divide_trainable_modules')
    global ann_train_module,snn_train_module
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = divide_trainable_modules(module)
        if module.__class__.__name__ != "Sequential":
            if module.__class__.__name__ == "SpikingNorm":
                snn_train_module.append(module)
            else:
                ann_train_module.append(module)
    return model
divide_trainable_modules(model)


# ---------------------- Define loss function ---------------------------------------------------------------------------
# same as 'new_loss_function' in 'fast_train.py'
# this func used to calculate loss between 'ann_out' and 'snn_out' by different methods
def new_loss_function(ann_out, snn_out, k, func='cos'):
    print('new_loss_function')
    if func == 'mse':
        f = nn.MSELoss()
        diff_loss = f(ann_out, snn_out) # assign 'diff_loss' equal to MSE between 'ann_out' and 'snn_out'
    elif func == 'cos':
        f = nn.CosineSimilarity(dim=1, eps=1e-6) # read more about CosineSimilarity func: https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
        diff_loss = 1.0 - torch.mean(f(ann_out, snn_out))
    else:
        assert False
    loss = diff_loss + lam * k
    return loss, diff_loss

# 'loss_function1' used to compute the cross entropy loss between the output of ANN training and target
loss_function1 = nn.CrossEntropyLoss()
# 'loss_function2' computes loss between 'ann_out' & 'snn_out' by MSE or CosineSimilarity as defined above
loss_function2 = new_loss_function
# ----------------------------------------------------------------------------------------------------------------------


# ---------------------- Define 'optimizer1' ---------------------------------------------------------------------------
if optimizer == 'sgd':
    optimizer1 = optim.SGD(ann_train_module.parameters(),
                               momentum=momentum,
                               lr=lr,
                               weight_decay=decay)
elif optimizer == 'adam':
    optimizer1 = optim.Adam(ann_train_module.parameters(),
                           lr=lr,
                           weight_decay=decay)
elif optimizer == 'adamw':
    optimizer1 = optim.AdamW(ann_train_module.parameters(),
                           lr=lr,
                           weight_decay=decay)
# ----------------------------------------------------------------------------------------------------------------------

########################################################################################################################
#
# SOME FUNCTIONS
#
########################################################################################################################

# same as 'adjust_learning_rate' function in 'fast_train.py'
def adjust_learning_rate(optimizer, epoch):
    print('\nadjust_learning_rate\n')
    global lr
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# same as 'layerwise_k' function in 'fast_train.py'
def layerwise_k(a, max=1.0):
    # print('Start layerwise_k')
    return torch.sum(a / max) / (torch.pow(torch.norm(a / max, 2), 2) + 1e-5)


# same as 'hook' function in 'fast_train.py'
def hook(module, input, output):
    # print('Start hook')
    global sum_k,cnt_k
    sum_k += layerwise_k(output)
    cnt_k += 1
    return


# 1.same as 'ann_train' function in 'para_train.py'
# 2.'ann_train' use variable 'train_dataloader' to train the model. The outputs below are returned. They are used to
#    evaluate the training result of ANN
# 3.outputs:
#       'ann_train_loss': loss between 'ann_outputs' and 'targets'
#       'ann_correct': nb of same elements between 'ann_predicted' and 'targets'
def ann_train(epoch):
    print('\n\n***** START ANN TRAINING (func ann_train) *****')
    global sum_k,cnt_k,train_batch_cnt
    net = model.to(device)

    #print('*** Epoch: %d' % epoch)
    net.train()
    ann_train_loss = 0
    ann_correct = 0
    total = 0
    # 1. In ANN training, we use 'train_dataloader' as training dataset. 'train_dataloader' has 50000 32x32 images. The
    #    below 'for' loop will loop 1000 time. Size of 'inputs' is [50, 3, 32, 32], it means that in each loop, 'inputs'
    #    contains 50 images, each image has 3 color channels, and each channel has size 32x32 pixel (matrix 32x32).
    # 2. 'targets' has the size [50,1] with elements varying from 0 to 9 corresponding to one of 10 classes. Each image
    #    has its targeted class.
    # 3. 'ann_outputs' has the size [50,10]. Each image corresponds to one row which contains 10 possibilities of being
    #    a class. So that we use 'ann_predicted' to find max possibility of each image, then compare with corresponding
    #    target
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader)): #tqdm is a library in Python which is used for creating Progress Meters or Progress Bars
        inputs, targets = inputs.to(device), targets.to(device) # size of 'inputs': torch.Size([50, 3, 32, 32])
                                                                # size of 'targets': torch.Size([50,1])
        ann_outputs = net(inputs) # size of 'ann_outputs': torch.Size([50, 10])
        ann_loss = loss_function1(ann_outputs, targets) # 'ann_loss' is a float number
        ann_train_loss += (ann_loss.item()) # Sum up all 'ann_loss' patterns
        _, ann_predicted = ann_outputs.max(1) # find all cases along dim 1 (max in each row) of 'ann_outputs'
                                              # size of 'ann_predicted': torch.Size([50,1])
        tot = targets.size(0) # 'tot' equals to size of 1st dimension of 'targets'
        total += tot
        ac = ann_predicted.eq(targets).sum().item() # find elements of 'ann_predicted' that equal to 'target, then sum them up
        ann_correct += ac

        # --------------------------------------------------------------------------------------------------------------
        # print('type of inputs of ann_train', type(inputs))
        # print('size of inputs of ann_train', inputs.size())
        # print('type of targets of ann_train', type(targets))
        # print('size of targets of ann_train', targets)
        # print('size of ann_outputs', ann_outputs)
        # print('ann_predicted', ann_predicted)
        # --------------------------------------------------------------------------------------------------------------

        # -------------------------Don't understand---------------------------------------------------------------------
        optimizer1.zero_grad()
        ann_loss.backward()
        # torch.nn.utils.clip_grad_norm_(ann_train_module.parameters(), 50)
        # --------------------------------------------------------------------------------------------------------------

        # All optimizers implement a step() method, that updates the parameters. It can be used as beside syntax
        optimizer1.step()
        if np.isnan(ann_loss.item()) or np.isinf(ann_loss.item()):
            print('Fail to calculate ann_loss', ann_loss)
            return False

        # Use 'train_batch_cnt' to count loop. 'train_batch_cnt' varies from 0 to 999.
        writer.add_scalar('Train/Acc', ac / tot, train_batch_cnt)
        writer.add_scalar('Train/Loss', ann_loss.item(), train_batch_cnt)
        train_batch_cnt += 1

    print('\n*** ANN training results (epoch %d): Loss:%.3f Acc:%.3f' % (epoch,
                                                      ann_train_loss,
                                                      ann_correct / total))

    writer.add_scalar('Train/EpochAcc', ann_correct / total, epoch)
    return


# 1.same as 'val' function in 'para_train.py'
# 2.'para_train_val' use variable 'test_dataloader' to test the model which was trained by 'ann_train'. It returns loss,
#   accuracy which are the result of the test
# 3.outputs:
#       'ann_test_loss': loss between 'ann_outputs' and 'targets'
#       'ann_correct': nb of same elements between 'ann_predicted' and 'targets'
#       'sum_k', 'cnt_k', 'last_k': WEIGHTs?
def para_train_val(epoch):
    print('\n\n***** RUN ANN TRAINED MODEL WITH TESTING DATA (func para_train_val) *****')
    global sum_k,cnt_k,test_batch_cnt,best_acc
    net = model.to(device) # Define 'net' as a CNN model which will be processed by 'device'. Both 'model' and 'device'
                           # are defined above

    #print('*** Epoch: %d' % epoch)

    handles = []
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            # '.register_forward_hook' registers a global forward hook for all the modules. It adds global state to the
            # nn.module module and it is only intended for debugging/profiling purposes.
            handles.append(m.register_forward_hook(hook))

    net.eval()
    ann_test_loss = 0
    ann_correct = 0
    total = 0

    with torch.no_grad():
        # 1. Size of input: [50,3,32,32], of target: [50,1], of ann_outputs: [50,10]. All sizes are similar with in 'ann_train'.
        #    The difference here is the 'for' loop 200 times instead of 1000 times as in 'ann_train'
        # 2. In each loop, the 'ann_outputs' of 50 images will be calculated based on the weights available inside ANN trained
        #    model.
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_dataloader)):
            sum_k = 0
            cnt_k = 0
            inputs, targets = inputs.to(device), targets.to(device)
            ann_outputs = net(inputs)
            ann_loss = loss_function1(ann_outputs, targets)

            if np.isnan(ann_loss.item()) or np.isinf(ann_loss.item()):
                print('Fail to calculate ann_loss', ann_loss)
                return False

            predict_outputs = ann_outputs.detach() # The 'detach()' method constructs a new view on a tensor which is
                                                   # declared not to need gradients, i.e., it is to be excluded from
                                                   # further tracking of operations, and therefore the subgraph involving
                                                   # this view is not recorded.
            ann_test_loss += (ann_loss.item())
            _, ann_predicted = predict_outputs.max(1) # '.max(1)' return max elements of rows in 'ann_predicted' and their positions

            tot = targets.size(0) # 'tot' = nb of rows in matrix 'targets'
            total += tot
            ac = ann_predicted.eq(targets).sum().item() # count nb of same elements between 'ann_predicted' and 'targets'
            ann_correct += ac

            # ----------------------------------------------------------------------------------------------------------
            #print('size of inputs', inputs.size())
            #print('size of targets', targets.size())
            #print('size of ann_outputs', ann_outputs.size())
            #-----------------------------------------------------------------------------------------------------------

            # 'layerwise_k':greedy layer-wise pretraining that
            # allowed very deep neural networks to be successfully trained
            # 'layerwise_k' is defined above
            last_k = layerwise_k(F.relu(ann_outputs), torch.max(ann_outputs))

            # The SummaryWriter class ('writer') is your main entry to log data for consumption and visualization by TensorBoard
            # Log 4 parameters of each loop for later consumption and visualization
            # syntax: writer.add_scalar('',y,x)
            writer.add_scalar('Test/Acc', ac / tot, test_batch_cnt)
            writer.add_scalar('Test/Loss', ann_test_loss, test_batch_cnt)
            writer.add_scalar('Test/AvgK', (sum_k / cnt_k).item(), test_batch_cnt)
            writer.add_scalar('Test/LastK', last_k, test_batch_cnt)
            test_batch_cnt += 1

        print('\n*** ANN trained model testing result (epoch %d): Loss:%.3f Acc:%.3f AvgK:%.3f LastK:%.3f' % (epoch,
                                                             ann_test_loss,
                                                             ann_correct / total,
                                                             sum_k / cnt_k, last_k))

    # Log 'ann_correct' for each epoch
    writer.add_scalar('Test/EpochAcc', ann_correct / total, epoch)

    # --------------------Save checkpoint-------------------------------------------------------------------------------
    # We just save the checkpoint when there is a better accuracy appears ('if acc > best_acc')
    # Parameters we save here are:
    #       'net.state_dict()': ???
    #       'acc': accuracy of the testing
    #       'epoch'
    acc = 100.*ann_correct/total  # 'acc' is percentage of correct elements over all elements
    if acc > best_acc:
        print('Saving checkpoint (para_train_val)...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir) # create directory 'log_dir'

        # save 'state' = ['net', 'acc', 'epoch'] to path train_vgg16_cifar10/vgg16_cifar10_para_train.pth
        # only have 1 file which saves the result
        torch.save(state, log_dir + '/%s_para_train.pth'%(save_name))
        best_acc = acc
    # ------------------------------------------------------------------------------------------------------------------

    # --------------------Schedule save checkpoint----------------------------------------------------------------------
    # We save checkpoint after every 10 epochs
    avg_k = ((sum_k + last_k) / (cnt_k + 1)).item()
    if (epoch + 1) % 10 == 0:
        print('Schedule saving checkpoint (para_train_val)...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'avg_k': avg_k
        }
        # save 'state' = ['net', 'acc', 'epoch', 'avg_k'] to path train_vgg16_cifar10/vgg16_cifar10_pt_scheduled.pth
        # only have 1 file which saves the result
        torch.save(state, log_dir + '/%s_para_train_scheduled.pth' % (save_name))
    for handle in handles:
        handle.remove()
    # ------------------------------------------------------------------------------------------------------------------


# 1.same as 'snn_train' function in 'fast_train.py'
# 2.'snn_train' uses 'train_dataloader' to train the model, it trains ANN and SNN, then compare results
# 3.outputs:
#       'snn_dist_loss', 'snn_fast_loss':
#                Two losses are considered ('fast_loss' and 'dist_lost'), which are loss between ANN & SNN training outputs
#                'snn_dist_loss' is cumulation of 'dist_loss'
#       'snn_correct': nb of same elements between 'snn_predicted' and 'targets' of SNN (not output of ANN training)
def snn_train(epoch):
    print('\n\n***** START SNN TRAINING (func snn_train) *****')
    global sum_k, cnt_k, train_batch_cnt, last_k
    net = model.to(device)

    print('*** Epoch: %d' % epoch)
    net.train()
    snn_fast_loss = 0
    snn_dist_loss = 0
    snn_correct = 0
    total = 0

    handles = []
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            handles.append(m.register_forward_hook(hook))

    # Size of 'inputs': [50,3,32,32], of 'targets': [50,1], of 'ann_outputs': [50,10], of 'snn_outputs': [50,10]
    # Every size are the same as in ANN training
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader)):
        sum_k = 0
        cnt_k = 0
        # ----------------------Run ANN training------------------------------------------------------------------------
        inputs, targets = inputs.to(device), targets.to(device)
        ann_outputs = net(inputs)
        ann_loss = loss_function1(ann_outputs, targets)

        if np.isnan(ann_loss.item()) or np.isinf(ann_loss.item()):
            print('Fail to calculate ann_loss', ann_loss)
            return False

        # 'detach()' method constructs a new view on a tensor which is declared not to need gradients, i.e., it is
        # to be excluded from further tracking of operations, and therefore the subgraph involving this view is not recorded.
        predict_outputs = ann_outputs.detach()
        _, ann_predicted = predict_outputs.max(1)
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # print('ANN TRAINING parameters\n')
        # print('size of input', inputs.size())
        # print('size of target', targets.size())
        # print('size of ann_outputs', ann_outputs.size())
        # --------------------------------------------------------------------------------------------------------------


        # ----------------------Run SNN training------------------------------------------------------------------------
        snn_outputs = net(inputs)
        check = (ann_outputs == snn_outputs)
        print('Is ann_outputs = snn_outputs', check)
        # print('SNN TRAINING parameters\n')
        # print('size of snn_outputa', snn_outputs.size())

        # 'F.relu(snn_outputs)' returns positive elements, others are set to 0
        # 'torch.mac(snn_outputs)' return max element of 'snn_outputs'
        # 'layerwise_k' is defined above
        last_k = layerwise_k(F.relu(snn_outputs), torch.max(snn_outputs))

        # 'predict_outputs' is output of ANN (i.e. 'ann_outputs'), 'snn_outputs' is output of SNN
        # 'loss_function2' uses MSE or CosineSimilarity technique to calculates difference between 'predict_outputs'
        #  (returned by ANN) and 'snn_outputs'.
        # fast_loss = dist_loss + lam * [(sum_k + last_k) / (cnt_k + 1)]
        fast_loss, dist_loss = loss_function2(predict_outputs, snn_outputs, (sum_k + last_k) / (cnt_k + 1))

        snn_dist_loss += dist_loss.item()
        snn_fast_loss += fast_loss.item()
        optimizer2.zero_grad()
        fast_loss.backward()
        optimizer2.step()

        _, snn_predicted = snn_outputs.max(1)
        tot = targets.size(0)
        total += tot
        sc = snn_predicted.eq(targets).sum().item()
        snn_correct += sc
        # --------------------------------------------------------------------------------------------------------------

        # The SummaryWriter class ('writer') is your main entry to log data for consumption and visualization by TensorBoard
        # Log 4 parameters of each loop (1 LOOP FOR 1 PICTURE ?) for later consumption and visualization
        # syntax: writer.add_scalar('',y,x)
        writer.add_scalar('Train/Acc', sc / tot, train_batch_cnt)
        writer.add_scalar('Train/DistLoss', dist_loss, train_batch_cnt)
        writer.add_scalar('Train/AvgK', (sum_k / cnt_k).item(), train_batch_cnt)
        writer.add_scalar('Train/LastK', last_k, train_batch_cnt)
        train_batch_cnt += 1

        # 'inspect_interval' is a time interval which is used to observe the data progress
        if train_batch_cnt % inspect_interval == 0:
            if not snn_val(train_batch_cnt):
                return False
            # 'net.train()' tells your model that you are training the model. So effectively layers like dropout,
            # batchnorm etc. which behave different on the train and test procedures know what is going on and hence
            # can behave accordingly.
            net.train()
    print('*** SNN training result (epoch %d): Loss:%.3f Acc:%.3f' % (epoch,
                                                      snn_dist_loss,
                                                      snn_correct / total))

    writer.add_scalar('Train/EpochAcc', snn_correct / total, epoch)
    for handle in handles:
        handle.remove()
    return True


# 1.same as 'get_acc' function in 'fast_train.py'
# 2.output:
#       'snn_acc': nb of same elements between 'predicted' (i.e. testing output of 'model' on 'val_dataloader') and 'targets'
# 3.Used to update the best accuracy to save the checkpoint in 'snn_val'.
# 4.Why in 'para_train_val.py', the update is contained in the file, not in separate file like in SNN case???
def get_acc(val_dataloader):
    print('\nget_acc\n')
    global model
    net = model
    net.to(device)

    net.eval()
    correct = 0
    total = 0
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            m.lock_max = True
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    snn_acc = correct / total
    return snn_acc


# 1.same as 'val' function in 'fast_train.py'
# 2.same as 'para_train_val' defined above
# 3.used for test dataset
# 3.why 'snn_val' uses ANN training instead of SNN???
def snn_val(iter):
    print('\nsnn_val\n')
    global sum_k, cnt_k, test_batch_cnt, best_acc, last_k, best_avg_k
    net = model.to(device)

    handles = []
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            handles.append(m.register_forward_hook(hook))

    net.eval()
    ann_test_loss = 0
    ann_correct = 0
    total = 0
    with torch.no_grad(): #disable gradient calculation.
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_dataloader)):
            sum_k = 0
            cnt_k = 0
            inputs, targets = inputs.to(device), targets.to(device)

            # ----------------------------ANN---------------------------------------------------------------------------
            ann_outputs = net(inputs)
            ann_loss = loss_function1(ann_outputs, targets)

            if np.isnan(ann_loss.item()) or np.isinf(ann_loss.item()):
                print('Fail to calculate ann_loss', ann_loss)
                return False

            predict_outputs = ann_outputs.detach()
            ann_test_loss += (ann_loss.item())
            _, ann_predicted = predict_outputs.max(1)

            tot = targets.size(0)
            total += tot
            ac = ann_predicted.eq(targets).sum().item()
            ann_correct += ac
            # ----------------------------------------------------------------------------------------------------------

            last_k = layerwise_k(F.relu(ann_outputs), torch.max(ann_outputs))
            # SummaryWriter class ('writer') is a main entry to log data for consumption, visualization by TensorBoard
            writer.add_scalar('Test/Acc', ac / tot, test_batch_cnt)
            writer.add_scalar('Test/Loss', ann_test_loss, test_batch_cnt)
            writer.add_scalar('Test/AvgK', (sum_k / cnt_k).item(), test_batch_cnt)
            writer.add_scalar('Test/LastK', last_k, test_batch_cnt)
            test_batch_cnt += 1
            #–––––-----------–––––-----------–––––-----------–––––-----------–––––-----------–––––-----------
        print('Test Iter %d: Loss:%.3f Acc:%.3f AvgK:%.3f LastK:%.3f' % (iter,
                                                                         ann_test_loss,
                                                                         ann_correct / total,
                                                                         sum_k / cnt_k, last_k))
    writer.add_scalar('Test/IterAcc', ann_correct / total, iter)

    # --------------------Save checkpoint-------------------------------------------------------------------------------
    # We just save the checkpoint when there is a better accuracy appears ('if acc > best_acc')
    # Parameters we save here are:
    #       'net.state_dict()': ???
    #       'acc': accuracy of the testing
    #       'epoch'
    avg_k = ((sum_k + last_k) / (cnt_k + 1)).item()
    acc = 100. * ann_correct / total
    if acc < (best_acc - acc_tolerance)*100.:
        return False
    if acc > (best_acc - acc_tolerance)*100. and best_avg_k > avg_k:
        test_acc = get_acc(test_dataloader)
        print('Saving checkpoint (snn_val)...')
        state = {
            'net': net.state_dict(),
            'acc': test_acc * 100,
            'epoch': epoch,
            'avg_k': avg_k
        }
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        torch.save(state, log_dir + '/%s_fast_train_[%.3f_%.3f_%.3f].pth' % (save_name,
                                                                       lam,test_acc * 100,
                                                                       ((sum_k + last_k) / (cnt_k + 1)).item() ))
        best_avg_k = avg_k
    # ------------------------------------------------------------------------------------------------------------------

    # --------------------Schedule save checkpoint----------------------------------------------------------------------
    # We save checkpoint after every 10 epochs
    if (epoch + 1) % 10 == 0:
        print('Schedule saving checkpoint (snn_val)...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, log_dir + '/%s_fast_train_scheduled.pth' % (save_name))
    for handle in handles:
        handle.remove()
    return True
    # ------------------------------------------------------------------------------------------------------------------


# 1. inputs:
#       'net': weights of model used to simulate. It would be defined in 'simulate_by_filename'. It can be loaded from
#              checkpoint file of (no schedule or schedule) ANN training or SNN training
#       'T': given parameter
#       'save_name': filename of the dataset which is input of 'simulate_by_filename' function
#       'log_dir': folder to save simulated results
#       'ann_baseline': check if we plot the figures or not
# 2. This function uses 'test_dataloader' dataset to run the simulation based on the weights have been found in ANN training
def simulate(net, T, save_name, log_dir, ann_baseline=0.0):
    print('\nStart simulation (func simulate)\n')
    net.to(device) # link to device for simulation
    functional.reset_net(net)
    correct_t = {}

    # 'torch.no_grad(): Context-manager that disabled gradient calculation.
    # Disable gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward().
    # It will reduce memory consumption for computations that would otherwise have requires_grad=True.
    with torch.no_grad():
        # 'net.eval()' is a kind of switch for some specific layers/parts of the model that behave differently during
        # training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn
        # off them during model evaluation, and '.eval()' will do it for you. In addition, the common practice for
        # evaluating/validation is using 'torch.no_grad()' in pair with 'model.eval()' to turn off gradients computation
        net.eval()
        correct = 0.0
        total = 0.0

        # 1. Size of inputs: [50,3,32,32], of targets: [50,1]
        for batch, (inputs, targets) in enumerate(test_dataloader):
            # Loop 'for t in range(T)':
            #           1. Variable 't' will be looped from 0 to 'T'-1 (in this case, from 0-99)
            #           2. Size of 'out' (output of testing results) is [50,10], of 'out_spikes_counter' is [50,10]
            #           3. In each loop of 't', 'out' will be predicted. Each 'out' after a loop will be sum up and save
            #           to 'out_spikes_counter'.
            #           4. After summing up of 'out', we always compare 'out_spikes_counter' with 'targets' to find nb of
            #           similar elements and save to 'correct_t'
            for t in range(T):
                # print('t', t)
                out = net(inputs.to(device))
                #print('size of output of simulate', out.size())
                #print('output of simulate', out)
                if isinstance(out, tuple) or isinstance(out, list):
                    out = out[0]
                if t == 0:
                    out_spikes_counter = out
                else:
                    out_spikes_counter += out
                # 'keys()' method returns a view object. The view object contains the keys of the dictionary, as a list.
                if t not in correct_t.keys():
                    # 'out_spikes_counter.max(1)' return the value and indices of max elements of each row of 'out_spikes_counter'
                    # 'out_spikes_counter.max(1)[1]' return the indices of max elements of each row of 'out_spikes_counter'
                    # 'float().sum().item()' sums up all float elements
                    # 'correct_t' saves nb of similar elements between 'out_spikes_counter' and 'targets' in each loop of 't'
                    # 'correct_t' will be (0:... 1:... 2:... ...)
                    # Normally, elements of 'correct_t' will be the same and equal to 'correct'
                    correct_t[t] = (out_spikes_counter.max(1)[1] == targets.to(device)).float().sum().item()
                else:
                    correct_t[t] += (out_spikes_counter.max(1)[1] == targets.to(device)).float().sum().item()

            # 'correct' counts the nb of correct classified images. The value of 'correct' is a cumulation of correct
            # classified images after each iteration of classifying 50 images
            correct += (out_spikes_counter.max(1)[1] == targets.to(device)).float().sum().item()
            #print('correct', correct)
            total += targets.numel() # '.numel()' returns the total number of elements in the input tensor
            #print('total', total)
            functional.reset_net(net)

            #------------------------------Plot testing result of SNN --------------------------------------------------
            fig = plt.figure()
            x = np.array(list(correct_t.keys())).astype(np.float32) + 1
            y = np.array(list(correct_t.values())).astype(np.float32) / total * 100
            plt.plot(x, y, label='SNN', c='b')
            # ----------------------------------------------------------------------------------------------------------

            #------------------------------Plot testing result of ANN --------------------------------------------------
            if ann_baseline != 0:
                plt.plot(x, np.ones_like(x) * ann_baseline, label='ANN', c='g', linestyle=':')
                plt.text(0, ann_baseline + 1, "%.3f%%" % (ann_baseline), fontdict={'size': '8', 'color': 'g'})
            #-----------------------------------------------------------------------------------------------------------

            plt.title("Weights from file: %s \n[test samples:%.1f%%]" % (save_name, 100 * total / len(test_dataloader.dataset)))
            plt.xlabel("T")
            plt.ylabel("Accuracy(%)")
            plt.legend()


            argmax = np.argmax(y)
            disp_bias = 0.3 * float(T) if x[argmax] / T > 0.7 else 0
            plt.text(x[argmax] - 0.8 - disp_bias, y[argmax] + 0.8, "MAX:%.3f%% T=%d" % (y[argmax], x[argmax]),
                     fontdict={'size': '12', 'color': 'r'})

            plt.scatter([x[argmax]], [y[argmax]], c='r')
            print('[SNN Simulating... %.2f%%] Acc:%.3f' % (100 * total / len(test_dataloader.dataset), correct / total))
            acc_list = np.array(list(correct_t.values())).astype(np.float32) / total * 100
            np.save(log_dir + '/snn_acc-list' + ('-constant'), acc_list)
            plt.savefig(log_dir + '/sim_' + save_name + ".jpg", dpi=1080)

            from PIL import Image
            im = Image.open(log_dir + '/sim_' + save_name + ".jpg")
            totensor = transforms.ToTensor()
            plt.close()
            # ----------------------------------------------------------------------------------------------------------
        acc = correct / total
        print('\nSNN SIMULATING ACCURACY:%.3f\n' % (acc))


# same as 'replace_spikingnorm_by_ifnode' in 'modules.py'
def replace_spikingnorm_by_ifnode(model):
    print('\nreplace_spikingnorm_by_ifnode\n')
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_spikingnorm_by_ifnode(module)
        if module.__class__.__name__ == "SpikingNorm":
            #'neuron.IFNode' represents the Integrate and Fire neuron
            model._modules[name] = neuron.IFNode(v_threshold=module.calc_v_th().data.item(),v_reset=None)
    return model


# Used to set up and call the 'simulate' function
def simulate_by_filename(save_name):
    print('\n***** SIMULATE BY FILENAME *****')
    print('Filename: %s' %save_name)
    model = models.__dict__[model_name](num_classes=10, dropout=0)
    model = modules.replace_maxpool2d_by_avgpool2d(model) #function from 'modules.py'
    model = modules.replace_relu_by_spikingnorm(model,True) #function from 'modules.py'
    state_dict = torch.load('train_vgg16_cifar10/%s.pth' % save_name)
    # In PyTorch, the learnable parameters (i.e. weights and biases) of a torch.nn.Module model are contained in the
    # model’s parameters (accessed with model.parameters()). A state_dict is simply a Python dictionary object that maps
    # each layer to its parameter tensor.
    ann_acc = state_dict['acc']
    model.load_state_dict(state_dict['net'])
    model = replace_spikingnorm_by_ifnode(model)
    simulate(model.to(device), T=100, save_name='%s' % save_name, log_dir=log_dir, ann_baseline=ann_acc)

########################################################################################################################
#
# PHASE 1: ANN TRAINING FOR WEIGHTS OF MODEL and SAVE CHECKPOINTS OF ANN TRAINED MODEL
#
########################################################################################################################
print('\n\n\n########################################################################################################')
print('PHASE 1: ANN TRAINING FOR WEIGHTS OF MODEL and SAVE CHECKPOINTS OF ANN TRAINED MODEL')
print('########################################################################################################')

# Number of epochs depends on the value of 'epoch' defined above
for epoch in range(start_epoch, start_epoch + epoch):
    print('\n*********************************************')
    print('Epoch: ', epoch)
    print('*********************************************')

    adjust_learning_rate(optimizer1, epoch)

    if epoch==start_epoch:
        para_train_val(epoch)
    ret = ann_train(epoch)
    if ret == False:
        break
    # output of 'para_train_val': 'epoch', 'ann_test_loss', 'ann_correct', 'sum_k', 'cnt_k', 'last_k'
    para_train_val(epoch)
    print("\nThres:")
    for n, m in model.named_modules():
        if isinstance(m, modules.SpikingNorm):
            print('thres', m.calc_v_th().data, 'scale', m.calc_scale().data)

########################################################################################################################
#
# PHASE 2: SNN TRAINING
#
########################################################################################################################
print('\n\n\n########################################################################################################')
print('PHASE 2: SNN TRAINING')
print('########################################################################################################')

dataset = train_dataloader.dataset
# divide 'dataset' (50000 images) into 'train_set' (40000 images) and 'val_set' (10000 images)
train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000])

# load data from 'train_set' and 'val_set' and save to 'train_dataloader' and 'val_data_loader'
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# In PyTorch, the learnable parameters (i.e. weights and biases) of torch.nn.Module model are contained in the model’s
# parameters (accessed with model.parameters()). A state_dict is simply a Python dictionary object that maps each layer
# to its parameter tensor.
# Load weights from file having path 'train_vgg16_cifar10/vgg16_cifar10_para_train.pth', which contains:
#       'net.state_dict()': ???
#       'acc': accuracy of the testing
#       'epoch'
model.load_state_dict(torch.load('train_vgg16_cifar10/vgg16_cifar10_para_train.pth')['net'])

# ----------------------------Don't understand--------------------------------------------------------------------------
if sharescale:
    first_scale = None
    sharescale = nn.Parameter(torch.Tensor([scale_init]))
    for m in model.modules():
        if isinstance(m, modules.SpikingNorm):
            setattr(m, 'scale', sharescale) # set the 'scale' of 'm' equals to 'sharescale'
            m.lock_max = True
# ----------------------------------------------------------------------------------------------------------------------

divide_trainable_modules(model)

# define opt2
lr = 0.001
inspect_interval = 100

# -------------------------- Define 'optimizer2' -----------------------------------------------------------------------
if optimizer == 'sgd':
    optimizer2 = optim.SGD(snn_train_module.parameters(),
                           momentum=momentum,
                           lr=lr,
                           weight_decay=decay)
elif optimizer == 'adam':
    optimizer2 = optim.Adam(snn_train_module.parameters(),
                           lr=lr,
                           weight_decay=decay)
# ----------------------------------------------------------------------------------------------------------------------
# 'get_acc' used to get the best testing accuracy returned by ANN
best_acc = get_acc(val_dataloader)

for e in range(0, epoch): # 'epoch'=200 as defined in line 35
    print('\n*********************************************')
    print('Epoch: ', e)
    print('*********************************************')

    adjust_learning_rate(optimizer2, e)
    # In here, the 'snn_train' uses the 'model' with parameters saved in 'train_vgg16_cifar10/vgg16_cifar10_para_train.pth'.
    # These parameters were obtained from the ANN training (line 781)
    ret = snn_train(e)
    if ret == False:
        break
    print("\nThres:")
    for n, m in model.named_modules():
        if isinstance(m, modules.SpikingNorm):
            print('thres', m.calc_v_th().data, 'scale', m.calc_scale().data, 'scale_t',m.scale.data)
            # '.calc_v_th()' and '.calc_scale()' are 2 functions defined in 'modules.py'
########################################################################################################################
#
# SIMULATION
#
########################################################################################################################
print('\n\n\n########################################################################################################')
print('START SIMULATION ON TRAINED WEIGHTS')
print('########################################################################################################')

# simulate_by_filename('vgg16_cifar10_[0.100_87.880_7.643]')
# simulate_by_filename('vgg16_cifar10_[0.100_86.840_6.528]')
# simulate_by_filename('vgg16_cifar10_[0.100_84.440_5.808]')

simulate_by_filename('vgg16_cifar10_para_train')
#simulate_by_filename('vgg16_cifar10_pt_scheduled')
