"""
######################## single-dataset inference lenet example ########################
This example is a single-dataset inference tutorial. 

######################## Instructions for using the inference environment ########################
1、Inference task requires predefined functions
(1)Copy single dataset from obs to inference image.
function ObsToEnv(obs_data_url, data_dir)

(2)Copy ckpt file from obs to inference image.
function ObsUrlToEnv(obs_ckpt_url, ckpt_url)

(3)Copy the output result to obs.
function EnvToObs(train_dir, obs_train_url)

3、4 parameters need to be defined.
--data_url is the dataset you selected on the Qizhi platform
--ckpt_url is the weight file you choose on the Qizhi platform

--data_url,--ckpt_url,--result_url,--device_target,These 4 parameters must be defined first in a single dataset, 
otherwise an error will be reported. 
There is no need to add these parameters to the running parameters of the Qizhi platform, 
because they are predefined in the background, you only need to define them in your code.                    

4、How the dataset is used
Inference task uses data_url as the input, and data_dir (ie: '/cache/data') as the calling method
of the dataset in the image.
For details, please refer to the following sample code.
"""

import os
import argparse
import moxing as mox
import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore import Tensor
import numpy as np
from glob import glob
from dataset import create_dataset
from config import mnist_cfg as cfg
from lenet import LeNet5

### Copy single dataset from obs to inference image ###
def ObsToEnv(obs_data_url, data_dir):
    try:     
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
    return 
### Copy ckpt file from obs to inference image###
### To operate on folders, use mox.file.copy_parallel. If copying a file. 
### Please use mox.file.copy to operate the file, this operation is to operate the file
def ObsUrlToEnv(obs_ckpt_url, ckpt_url):
    try:
        mox.file.copy(obs_ckpt_url, ckpt_url)
        print("Successfully Download {} to {}".format(obs_ckpt_url,ckpt_url))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_ckpt_url, ckpt_url) + str(e)) 
    return  
### Copy the output result to obs###
def EnvToObs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,obs_train_url) + str(e))
    return      

### --data_url,--ckpt_url,--result_url,--device_target,These 4 parameters must be defined first in a inference task, 
### otherwise an error will be reported.
### There is no need to add these parameters to the running parameters of the Qizhi platform, 
### because they are predefined in the background, you only need to define them in your code.
parser = argparse.ArgumentParser(description='MindSpore Lenet Example')
parser.add_argument('--data_url',
                type=str,
                default= '/cache/data/',
                help='path where the dataset is saved')      
parser.add_argument('--ckpt_url',
                help='model to save/load',
                default=  '/cache/checkpoint.ckpt')  
parser.add_argument('--result_url',
                help='result folder to save/load',
                default= '/cache/result/')   
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: Ascend)')                

if __name__ == "__main__":            
    args, unknown = parser.parse_known_args()

    ###Initialize the data and result directories in the inference image###
    data_dir = '/cache/data'  
    result_dir = '/cache/result'
    ckpt_url = '/cache/checkpoint.ckpt'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    ###Copy dataset from obs to inference image
    ObsToEnv(args.data_url, data_dir)

    ###Copy ckpt file from obs to inference image
    ObsUrlToEnv(args.ckpt_url, ckpt_url)

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    network = LeNet5(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    repeat_size = cfg.epoch_size
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")

    param_dict = load_checkpoint(os.path.join(ckpt_url))
    load_param_into_net(network, param_dict)
    ds_test = create_dataset(os.path.join(data_dir, "test"), batch_size=1).create_dict_iterator()
    data = next(ds_test)
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()
    print('Tensor:', Tensor(data['image']))
    output = model.predict(Tensor(data['image']))
    predicted = np.argmax(output.asnumpy(), axis=1)
    pred = np.argmax(output.asnumpy(), axis=1)
    print('predicted:', predicted)
    print('pred:', pred)

    print(f'Predicted: "{predicted[0]}", Actual: "{labels[0]}"')
    filename = 'result.txt'
    file_path = os.path.join(result_dir, filename)
    with open(file_path, 'a+') as file:
            file.write("    {}: {:.2f} \n".format("Predicted", predicted[0]))

    ###Copy result data from the local running environment back to obs,
    ###and download it in the inference task corresponding to the Qizhi platform
    EnvToObs(result_dir, args.result_url)