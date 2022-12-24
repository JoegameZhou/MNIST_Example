'''
imagnet-1k 数据集已通过磁盘挂载的方式挂载在训练镜像中，
用户可参考下列示例代码读取直接使用。

挂载路径为
.
└── cache/
    ├── ascend
    ├── outputs
    ├── user-job-dir
    └── sfs/
        └── data/
            └── imagenet/
                ├── train/
                │   └── n01440764/
                │       └── n01440764_11063.JPEG
                └── val/
                    └── n01440764/
                        └── ILSVRC2012_val_00011993.JPEG

mindspore.dataset.ImageFolderDataset
    - 读取imagenet-1k数据，同一文件夹下的数据为同一类class。
mindspore.dataset.vision.c_transforms
    - 数据加载和预处理。
mindspore.dataset.ImageFolderDataset
    - map：给定一组数据增强列表，按顺序将数据增强作用在数据集对象上。
    - batch：将数据集中连续 batch_size 条数据合并为一个批处理数据。
    - to_json：将数据处理管道序列化为JSON字符串，如果提供了文件名，则转储到文件中。

'''

import os
import argparse
import moxing as mox

import mindspore as ms
from mindspore.dataset import ImageFolderDataset
import mindspore.dataset.vision.c_transforms as transforms


### Copy the output to obs###
def EnvToObs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir, obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,
                                                       obs_train_url) + str(e))
    return


def UploadToQizhi(train_dir, obs_train_url):
    device_num = int(os.getenv('RANK_SIZE'))
    local_rank = int(os.getenv('RANK_ID'))
    if device_num == 1:
        EnvToObs(train_dir, obs_train_url)
    if device_num > 1:
        if local_rank % 8 == 0:
            EnvToObs(train_dir, obs_train_url)
    return

parser = argparse.ArgumentParser(description='Read big dataset ImageNet Example')
parser.add_argument('--train_url',
                    help='output folder to save/load',
                    default= '/cache/output/')

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    data_path = '/cache/sfs/data/imagenet/'
    modelart_output = '/cache/output'
    if not os.path.exists(modelart_output):
        os.makedirs(modelart_output)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    dataset_train = ImageFolderDataset(os.path.join(data_path, "train"),
                                       shuffle=True)
    trans_train = [
        transforms.RandomCropDecodeResize(size=224,
                                          scale=(0.08, 1.0),
                                          ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(prob=0.5),
        transforms.Normalize(mean=mean, std=std),
        transforms.HWC2CHW()
    ]

    dataset_train = dataset_train.map(operations=trans_train,
                                      input_columns=["image"])
    dataset_train = dataset_train.batch(batch_size=16, drop_remainder=True)

    data_info = dataset_train.to_json(filename= modelart_output + '/data_info.json')
    print(data_info)
    UploadToQizhi(modelart_output, args.train_url)