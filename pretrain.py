"""
######################## single-dataset train lenet example ########################
This example is a single-dataset training tutorial. If it is a multi-dataset, please refer to the multi-dataset training 
tutorial train_for_multidataset.py. This example cannot be used for multi-datasets!

######################## Instructions for using the training environment ########################
The image of the debugging environment and the image of the training environment are two different images, 
and the working local directories are different. In the training task, you need to pay attention to the following points.
1、(1)The structure of the dataset uploaded for single dataset training in this example
 MNISTData.zip
  ├── test
  └── train


2、Single dataset training requires predefined functions
(1)Copy single dataset from obs to training image
function ObsToEnv(obs_data_url, data_dir)

(2)Copy the output to obs
function EnvToObs(train_dir, obs_train_url)

(3)Download the input from Qizhi And Init 
function DownloadFromQizhi(obs_data_url, data_dir)

(4)Upload the output to Qizhi 
function UploadToQizhi(train_dir, obs_train_url)

(5)Copy ckpt file from obs to training image.
function ObsUrlToEnv(obs_ckpt_url, ckpt_url)

3、3 parameters need to be defined
--data_url is the dataset you selected on the Qizhi platform

--data_url,--train_url,--device_target,These 3 parameters must be defined first in a single dataset task, 
otherwise an error will be reported.    
There is no need to add these parameters to the running parameters of the Qizhi platform, 
because they are predefined in the background, you only need to define them in your code.                 

4、How the dataset is used
A single dataset uses data_url as the input, and data_dir (ie:'/cache/data') as the calling method
of the dataset in the image.
For details, please refer to the following sample code.

5、How to load the pretrain model checkpoint file
The checkpoint file is loaded by the ckpt_url parameter

In addition, if you want to get the model file after each training, you can call the UploadOutput.
"""

import os
import argparse
import moxing as mox
from config import mnist_cfg as cfg
from dataset import create_dataset
from dataset_distributed import create_dataset_parallel
from lenet import LeNet5
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
import mindspore.ops as ops
import time
from upload import UploadOutput

### Copy single dataset from obs to training image###
def ObsToEnv(obs_data_url, data_dir):
    try:     
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
    #Set a cache file to determine whether the data has been copied to obs. 
    #If this file exists during multi-card training, there is no need to copy the dataset multiple times.
    f = open("/cache/download_input.txt", 'w')    
    f.close()
    try:
        if os.path.exists("/cache/download_input.txt"):
            print("download_input succeed")
    except Exception as e:
        print("download_input failed")
    return 
### Copy ckpt file from obs to training image###
### To operate on folders, use mox.file.copy_parallel. If copying a file. 
### Please use mox.file.copy to operate the file, this operation is to operate the file
def ObsUrlToEnv(obs_ckpt_url, ckpt_url):
    try:
        mox.file.copy(obs_ckpt_url, ckpt_url)
        print("Successfully Download {} to {}".format(obs_ckpt_url,ckpt_url))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_ckpt_url, ckpt_url) + str(e)) 
    return  
### Copy the output to obs###
def EnvToObs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,obs_train_url) + str(e))
    return      
def DownloadFromQizhi(obs_data_url, data_dir):
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        ObsToEnv(obs_data_url,data_dir)
        context.set_context(mode=context.GRAPH_MODE,device_target=args.device_target)
    if device_num > 1:
        # set device_id and init for multi-card training
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=int(os.getenv('ASCEND_DEVICE_ID')))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num = device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, parameter_broadcast=True)
        init()
        #Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
        local_rank=int(os.getenv('RANK_ID'))
        if local_rank%8==0:
            ObsToEnv(obs_data_url,data_dir)
        #If the cache file does not exist, it means that the copy data has not been completed,
        #and Wait for 0th card to finish copying data
        while not os.path.exists("/cache/download_input.txt"):
            time.sleep(1)  
    return
def UploadToQizhi(train_dir, obs_train_url):
    device_num = int(os.getenv('RANK_SIZE'))
    local_rank=int(os.getenv('RANK_ID'))
    if device_num == 1:
        EnvToObs(train_dir, obs_train_url)
    if device_num > 1:
        if local_rank%8==0:
            EnvToObs(train_dir, obs_train_url)
    return

### --data_url,--train_url,--device_target,These 3 parameters must be defined first in a single dataset, 
### otherwise an error will be reported.
###There is no need to add these parameters to the running parameters of the Qizhi platform, 
###because they are predefined in the background, you only need to define them in your code.
parser = argparse.ArgumentParser(description='MindSpore Lenet Example')
parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default= '/cache/data/')

parser.add_argument('--train_url',
                    help='output folder to save/load',
                    default= '/cache/output/')
parser.add_argument('--ckpt_url',
                help='model to save/load',
                default=  '/cache/checkpoint.ckpt')

parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'CPU'],
    help='device where the code will be implemented (default: Ascend),if to use the CPU on the Qizhi platform:device_target=CPU')

parser.add_argument('--epoch_size',
                    type=int,
                    default=5,
                    help='Training epochs.')

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    data_dir = '/cache/data'  
    train_dir = '/cache/output'
    ckpt_url = '/cache/checkpoint.ckpt'
    try: 
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
    except Exception as e:
        print("path already exists")
    ###Initialize and copy data to training image
    ###Copy ckpt file from obs to training image
    ObsUrlToEnv(args.ckpt_url, ckpt_url)
    ###Copy data from obs to training image
    DownloadFromQizhi(args.data_url, data_dir)
    ###The dataset path is used here:data_dir +"/train"   
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        ds_train = create_dataset(os.path.join(data_dir, "train"),  cfg.batch_size)
    if device_num > 1:
        ds_train = create_dataset_parallel(os.path.join(data_dir, "train"),  cfg.batch_size)
    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    network = LeNet5(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    ###The ckpt path is used here:ckpt_url
    load_param_into_net(network, load_checkpoint(ckpt_url))

    if args.device_target != "Ascend":
        model = Model(network,
                      net_loss,
                      net_opt,
                      metrics={"accuracy": Accuracy()})
    else:
        model = Model(network,
                      net_loss,
                      net_opt,
                      metrics={"accuracy": Accuracy()},
                      amp_level="O2")

    config_ck = CheckpointConfig(
        save_checkpoint_steps=cfg.save_checkpoint_steps,
        keep_checkpoint_max=cfg.keep_checkpoint_max)
    #Note that this method saves the model file on each card. You need to specify the save path on each card.
    # In this example, get_rank() is added to distinguish different paths.
    if device_num == 1:
        outputDirectory = train_dir + "/"
    if device_num > 1:
        outputDirectory = train_dir + "/" + str(get_rank()) + "/"
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet",
                                directory=outputDirectory,
                                config=config_ck)
    print("============== Starting Training ==============")
    epoch_size = cfg['epoch_size']
    if (args.epoch_size):
        epoch_size = args.epoch_size
        print('epoch_size is: ', epoch_size)
    #Custom callback, upload output after each epoch
    uploadOutput = UploadOutput(train_dir,args.train_url)
    model.train(epoch_size,
                ds_train,
                callbacks=[time_cb, ckpoint_cb,
                           LossMonitor(), uploadOutput])

    ###Copy the trained output data from the local running environment back to obs,
    ###and download it in the training task corresponding to the Qizhi platform
    #This step is not required if UploadOutput is called
    UploadToQizhi(train_dir,args.train_url)