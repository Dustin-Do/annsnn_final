import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import models
import argparse
import json
import modules
from utils import *
import numpy as np
import torch.nn.functional as F

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith('__') and callable(models.__dict__[name]))

# ---------------------------------Controling center of TRAINING--------------------------------------------------------
# Create ArgumentParser object, which holds all the info necessary to parse the command line into Python data types.
parser = argparse.ArgumentParser(description='Trainer')

# Fill an ArgumentParser with information about program arguments, which tell the ArgumentParser how to take the strings
# on the command line and turn them into objects. This information is stored and used when parse_args() is called
# Note:
#   'type': The type to which the command-line argument should be converted.
#   'default': The value produced if the argument is absent from the command line or absent from the namespace object.
#   'choices': A container of the allowable values for the argument.
#   'help': A brief description of what the argument does.
#   'nargs': The nargs keyword argument associates a different number of command-line arguments with a single action.
#            nargs='+': all command-line args present are gathered into a list.
#   'action': The basic type of action to be taken when this argument is encountered at the command line.
parser.add_argument('--dataset', type=str, default='cifar10', choices=list(num_classes.keys()), help='Dataset name.')
parser.add_argument('--no_data_aug', default=False, action='store_true', help='Disable data augmentation.')
parser.add_argument('--model', default='preactresnet18', choices=model_names, help='Model architecture.')

parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'], help='Choice of optimizer.')
parser.add_argument('--epoch', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.1, help='Global learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=1e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--dropout', type=float, default=0, help='Dropout applied to the model.')

parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# 'cuda:0,1,...,10' is useful only when we have multiple GPUs to choose. If not it uses cuda:0 as default
parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7', 'cuda:8', 'cuda:9', 'cuda:10'], help='Device.')

parser.add_argument('--sharescale', default=False, action='store_true', help='Disable scale share.')
parser.add_argument('--resume', '-r', default=False, help='resume from checkpoint')
parser.add_argument('--prefix', type=str, default='', help='model dir prefix')

# '.parser.parse_args()' inspects the command line, convert each argument to the appropriate type and then invoke the
# appropriate action. This means a simple Namespace object will be built up from attributes parsed out of the command line
args = parser.parse_args()
args.num_classes = num_classes[args.dataset]
# ----------------------------------------------------------------------------------------------------------------------


# ---------------------------------------Choose CUDA or CPU for device--------------------------------------------------
args.device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'
# 'assert' statements that you can use to set sanity checks during the development process. Assertions allow you to test
# the correctness of your code by checking if some specific conditions remain true, which can come in handy while you’re
# debugging code.
assert len(args.gammas) == len(args.schedule)
# ----------------------------------------------------------------------------------------------------------------------

conf = [args.model,args.dataset]
save_name = '_'.join(conf)
if len(args.prefix) == 0:
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = 'train_' + save_name + '_' + current_time
else:
    log_dir = 'train_' + save_name + '_' + args.prefix

args.log_dir = log_dir
args.save_name = save_name
print('log_dir: %s',args.log_dir)
print_args(args)

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
with open(os.path.join(log_dir, 'config.json'), 'w') as fw:
    json.dump(vars(args), fw)

# ----------------------------------Load dataset---------------------------------------------------
train_dataloader, test_dataloader = load_cv_data(data_aug=False,
                 batch_size=args.batch_size,
                 workers=0,
                 dataset=args.dataset,
                 data_target_dir=datapath[args.dataset]
                 )
# -------------------------------------------------------------------------------------------------

best_acc = 0.0
start_epoch = 0
sum_k = 0.0
cnt_k = 0.0
train_batch_cnt = 0
test_batch_cnt = 0

# Define model as 'preactresnet18'
model = models.__dict__[args.model](num_classes=args.num_classes, dropout=args.dropout)

# 'replace_maxpool2d_by_avgpool2d' & 'replace_relu_by_spikingnorm' are defined in 'modules.py'
model = modules.replace_maxpool2d_by_avgpool2d(model)
model = modules.replace_relu_by_spikingnorm(model,True)

if args.resume and os.path.isfile(args.resume):
    # Load checkpoint.
    print('==> Resuming from checkpoint...')
    checkpoint = torch.load(args.resume)
    # A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor. Note that only
    # layers with learnable parameters (convolutional layers, linear layers, etc.) and registered buffers (batchnorm’s
    # running_mean) have entries in the model’s state_dict.
    # '.load_state_dict' is used to load a state_dict
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    # Initialize the weights and biases of the layers
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)): # if 'm' is 'nn.Conv2d' (AND/OR)? 'nn.Linear'
            # Create weight explicitly by creating a random matrix based on Kaiming initialization
            # Read more:https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            if hasattr(m,'bias') and m.bias is not None: # if 'm' has 'bias' and 'm.bias' is not none
                nn.init.zeros_(m.bias) # set value of 'm.bias' to 0

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, val=1) # set value of 'm.weight' to 1
            nn.init.zeros_(m.bias) # set value of 'm.bias' to 0



# ------------------------Specify simulating configuration--------------------------------------------------------------
model.to(args.device)
args.device = torch.device(args.device)
if args.device.type == 'cuda':
    print(f"=> cuda memory allocated: {torch.cuda.memory_allocated(args.device.index)}")
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------Still don't understand--------------------------------------------------------------------
if args.sharescale:
    first_scale = None
    for m in model.modules():
        if isinstance(m,modules.SpikingNorm) and first_scale is None: # if 'm' is class 'SpikingNorm' & 'first_scale' is none
            first_scale = m.scale
        elif isinstance(m,modules.SpikingNorm) and first_scale is not None:
            setattr(m,'scale',first_scale) # set the value of 'scale' of 'm' equals to 'first_scale'
# ----------------------------------------------------------------------------------------------------------------------

ann_train_module = nn.ModuleList()
snn_train_module = nn.ModuleList()


def divide_trainable_modules(model):
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
loss_function1 = nn.CrossEntropyLoss()
# '.CrossEntropyLoss()' is defined in Pytorch, used to the cross entropy loss between input and target


# ----------------------------Choose optimizing method----------------------------
if args.optimizer == 'sgd':
    optimizer1 = optim.SGD(ann_train_module.parameters(),
                               momentum=args.momentum,
                               lr=args.lr,
                               weight_decay=args.decay)
elif args.optimizer == 'adam':
    optimizer1 = optim.Adam(ann_train_module.parameters(),
                           lr=args.lr,
                           weight_decay=args.decay)
elif args.optimizer == 'adamw':
    optimizer1 = optim.AdamW(ann_train_module.parameters(),
                           lr=args.lr,
                           weight_decay=args.decay)
# --------------------------------------------------------------------------------
writer = SummaryWriter(log_dir)


def adjust_learning_rate(optimizer, epoch):
    global args
    lr = args.lr
    for (gamma, step) in zip(args.gammas, args.schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


sum_k = 0
cnt_k = 0
last_k = 0
test_batch_cnt = 0
train_batch_cnt = 0


def layerwise_k(a, max=1.0):
    return torch.sum(a / max) / (torch.pow(torch.norm(a / max, 2), 2) + 1e-5)


def hook(module, input, output):
    global sum_k,cnt_k
    sum_k += layerwise_k(output)
    cnt_k += 1
    return

# 1.same as 'ann_train' function in 'tutorial.py'
# 2.'ann_train' use variable 'train_dataloader' to train the model
# 3.outputs:
#       'ann_train_loss': loss between 'ann_outputs' and 'targets'
#       'ann_correct': nb of same elements between 'ann_predicted' and 'targets'
def ann_train(epoch, args):
    print('Start ann_train')
    global sum_k,cnt_k,train_batch_cnt
    net = model.to(args.device)

    print('\n (ANN train) Epoch: %d Para Train' % epoch)
    net.train()
    ann_train_loss = 0
    ann_correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader)):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        ann_outputs = net(inputs)
        ann_loss = loss_function1(ann_outputs, targets)

        ann_train_loss += (ann_loss.item())
        _, ann_predicted = ann_outputs.max(1)

        tot = targets.size(0)
        total += tot
        ac = ann_predicted.eq(targets).sum().item()
        ann_correct += ac

        optimizer1.zero_grad()
        ann_loss.backward()
        # torch.nn.utils.clip_grad_norm_(ann_train_module.parameters(), 50)
        optimizer1.step()
        if np.isnan(ann_loss.item()) or np.isinf(ann_loss.item()):
            print('encounter ann_loss', ann_loss)
            return False

        writer.add_scalar('Train/Acc', ac / tot, train_batch_cnt)
        writer.add_scalar('Train/Loss', ann_loss.item(), train_batch_cnt)
        train_batch_cnt += 1
    print('Para Train Epoch %d Loss:%.3f Acc:%.3f' % (epoch,
                                                      ann_train_loss,
                                                      ann_correct / total))
    writer.add_scalar('Train/EpochAcc', ann_correct / total, epoch)
    return


# Used to run the simulation on the testing dataset
# Output:
#       1. 'ann_test_loss': loss between ann_outputs and targets (NOTE: this one is
#                           different with 'ann_train_loss' in 'ann_train.py')
#       2. 'ann_correct': nb of same elements between 'ann_predicted' and 'targets' (NOTE: this one is
#                         same as 'ann_correct' in 'ann_train.py')
def val(epoch, args):
    global sum_k,cnt_k,test_batch_cnt,best_acc
    net = model.to(args.device)

    handles = []
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            handles.append(m.register_forward_hook(hook))

    net.eval()
    ann_test_loss = 0
    ann_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_dataloader)):
            sum_k = 0
            cnt_k = 0
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            ann_outputs = net(inputs)
            ann_loss = loss_function1(ann_outputs, targets)

            if np.isnan(ann_loss.item()) or np.isinf(ann_loss.item()):
                print('encounter ann_loss', ann_loss)
                return False

            # The detach() method constructs a new view on a tensor which is
            # declared not to need gradients, i.e., it is to be excluded from
            # further tracking of operations, and therefore the subgraph involving
            # this view is not recorded.
            predict_outputs = ann_outputs.detach()

            ann_test_loss += (ann_loss.item())

            _, ann_predicted = predict_outputs.max(1) # '.max(1)' return max elements of rows in 'ann_predicted' and their positions

            tot = targets.size(0)  # 'tot' = nb of rows in maxtrix 'targets'
            total += tot
            ac = ann_predicted.eq(targets).sum().item()
            ann_correct += ac

            # 'layerwise_k':greedy layer-wise pretraining that
            # allowed very deep neural networks to be successfully trained
            # 'layerwise_k' is defined above
            last_k = layerwise_k(F.relu(ann_outputs), torch.max(ann_outputs))

            # The SummaryWriter class ('writer') is your main entry to log data for consumption and visualization by TensorBoard
            writer.add_scalar('Test/Acc', ac / tot, test_batch_cnt)
            writer.add_scalar('Test/Loss', ann_test_loss, test_batch_cnt)
            writer.add_scalar('Test/AvgK', (sum_k / cnt_k).item(), test_batch_cnt)
            writer.add_scalar('Test/LastK', last_k, test_batch_cnt)
            test_batch_cnt += 1


        print('Test Epoch %d Loss:%.3f Acc:%.3f AvgK:%.3f LastK:%.3f' % (epoch,
                                                             ann_test_loss,
                                                             ann_correct / total,
                                                             sum_k / cnt_k, last_k))
    writer.add_scalar('Test/EpochAcc', ann_correct / total, epoch)

    # ----------------------Save checkpoint-----------------------------------------------------------------------------
    acc = 100.*ann_correct/total
    if acc > best_acc:
        print('Saving checkpoint (val(ANN))...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.log_dir):
            os.mkdir(args.log_dir)
        torch.save(state, args.log_dir + '/%s.pth'%(args.save_name))
        best_acc = acc

    avg_k = ((sum_k + last_k) / (cnt_k + 1)).item()
    if (epoch + 1) % 10 == 0:
        print('Schedule saving checkpoint (val(ANN))...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'avg_k': avg_k
        }
        torch.save(state, args.log_dir + '/%s_pt_scheduled.pth' % (args.save_name))
    for handle in handles:
        handle.remove()
    # ------------------------------------------------------------------------------------------------------------------

for epoch in range(start_epoch, start_epoch + args.epoch):
    adjust_learning_rate(optimizer1, epoch)
    if args.resume and epoch==start_epoch:
        val(epoch, args)
    ret = ann_train(epoch, args)
    if ret==False:
        exit(-1)
    val(epoch, args)
    print("\nThres:")
    for n, m in model.named_modules():
        if isinstance(m, modules.SpikingNorm):
            print('thres', m.calc_v_th().data, 'scale', m.calc_scale().data)
