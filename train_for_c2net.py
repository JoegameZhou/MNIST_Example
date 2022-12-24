"""
######################## Attention!  ########################
智算网络需要在代码里使用mox拷贝数据集并解压，请参考函数C2netMultiObsToEnv;
不管是单数据集还是多数据集，在智算网络中都使用multi_data_url参数进行传递！
The intelligent computing network needs to use mox to copy the dataset and decompress it in the code, 
please refer to the function C2netMultiObsToEnv()

######################## multi-dataset train lenet example ########################
This example is a multi-dataset training tutorial. If it is a single dataset, please refer to the single dataset 
training tutorial train.py. This example cannot be used for a single dataset!
"""
"""
######################## Instructions for using the training environment ########################
1、(1)The structure of the dataset uploaded for multi-dataset training in this example
 MNISTData.zip
  ├── test
  └── train 
 
 checkpoint_lenet-1_1875.zip
  ├── checkpoint_lenet-1_1875.ckpt

  (2)The dataset structure in the training image for multiple datasets in this example
  workroot
   ├── MNISTData
   |     ├── test
   |     └── train 
   └── checkpoint_lenet-1_1875
         ├── checkpoint_lenet-1_1875.ckpt

2、Multi-dataset training requires predefined functions
(1)Copy multi-dataset from obs to training image and unzip
function C2netMultiObsToEnv(multi_data_url, data_dir)

(2)Copy the output to obs
function EnvToObs(train_dir, obs_train_url)

(2)Download the input from Qizhi And Init 
function DownloadFromQizhi(multi_data_url, data_dir)

(2)Upload the output to Qizhi 
function UploadToQizhi(train_dir, obs_train_url)

3、4 parameters need to be defined
--multi_data_url is the multi-dataset you selected on the Qizhi platform

--multi_data_url,--train_url,--device_target,These 3 parameters must be defined first in a multi-dataset task, 
otherwise an error will be reported.     
There is no need to add these parameters to the running parameters of the Qizhi platform, 
because they are predefined in the background, you only need to define them in your code                

4、How the dataset is used
Multi-datasets use multi_data_url as input, data_dir + dataset name + file or folder name in the dataset as the 
calling path of the dataset in the training image.
For example, the calling path of the train folder in the MNIST_Data dataset in this example is 
data_dir + "/MNIST_Data" +"/train"

For details, please refer to the following sample code.
"""

import os
import argparse

import moxing as mox
from config import mnist_cfg as cfg
from dataset import create_dataset
from dataset_distributed import create_dataset_parallel
from lenet import LeNet5
import json
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore import load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
import time

### Copy multiple datasets from obs to training image and unzip###  
def C2netMultiObsToEnv(multi_data_url, data_dir):
    #--multi_data_url is json data, need to do json parsing for multi_data_url
    multi_data_json = json.loads(multi_data_url)  
    for i in range(len(multi_data_json)):
        zipfile_path = data_dir + "/" + multi_data_json[i]["dataset_name"]
        try:
            mox.file.copy(multi_data_json[i]["dataset_url"], zipfile_path) 
            print("Successfully Download {} to {}".format(multi_data_json[i]["dataset_url"],zipfile_path))
            #get filename and unzip the dataset
            filename = os.path.splitext(multi_data_json[i]["dataset_name"])[0]
            filePath = data_dir + "/" + filename
            if not os.path.exists(filePath):
                os.makedirs(filePath)
            os.system("unzip {} -d {}".format(zipfile_path, filePath))

        except Exception as e:
            print('moxing download {} to {} failed: '.format(
                multi_data_json[i]["dataset_url"], zipfile_path) + str(e))
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
### Copy the output model to obs ###  
def EnvToObs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,
                                                    obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,
                                                    obs_train_url) + str(e))
    return                                                       
def DownloadFromQizhi(multi_data_url, data_dir):
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        C2netMultiObsToEnv(multi_data_url,data_dir)
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
            C2netMultiObsToEnv(multi_data_url,data_dir)
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

parser = argparse.ArgumentParser(description='MindSpore Lenet Example')
### --multi_data_url,--train_url,--device_target,These 3 parameters must be defined first in a multi-dataset,
### otherwise an error will be reported. 
### There is no need to add these parameters to the running parameters of the Qizhi platform, 
### because they are predefined in the background, you only need to define them in your code.

parser.add_argument('--multi_data_url',
                    help='path to multi dataset',
                    default= '/cache/data/')

parser.add_argument('--train_url',
                    help='model folder to save/load',
                    default= '/cache/output/')

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
    try: 
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
    except Exception as e:
        print("path already exists")
    ###Initialize and copy data to training image
    DownloadFromQizhi(args.multi_data_url, data_dir)
    ###The dataset path is used here:data_dir + "/MNIST_Data" +"/train"  
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        ds_train = create_dataset(os.path.join(data_dir + "/MNISTData", "train"),  cfg.batch_size)
    if device_num > 1:
        ds_train = create_dataset_parallel(os.path.join(data_dir + "/MNISTData", "train"),  cfg.batch_size)
    if ds_train.get_dataset_size() == 0:
        raise ValueError(
            "Please check dataset size > 0 and batch_size <= dataset size")
    network = LeNet5(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())

    if args.device_target != "Ascend":
        model = Model(network,net_loss,net_opt,metrics={"accuracy": Accuracy()})
    else:
        model = Model(network, net_loss,net_opt,metrics={"accuracy": Accuracy()},amp_level="O2")
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                keep_checkpoint_max=cfg.keep_checkpoint_max)
    #Note that this method saves the model file on each card. You need to specify the save path on each card.
    # In this example, get_rank() is added to distinguish different paths.
    if device_num == 1:
        outputDirectory = train_dir 
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
    # set callback functions
    callback =[time_cb,LossMonitor()]
    local_rank=int(os.getenv('RANK_ID'))
    # for data parallel, only save checkpoint on rank 0
    if local_rank==0 :
        callback.append(ckpoint_cb) 
    model.train(epoch_size,
                ds_train,
                callbacks=callback)