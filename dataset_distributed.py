
"""
Produce the dataset：
与单机不同的是，在数据集接口需要传入num_shards和shard_id参数，分别对应卡的数量和逻辑序号，建议通过HCCL接口获取：
get_rank：获取当前设备在集群中的ID。
get_group_size：获取集群数量。

"""

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_rank, get_group_size

def create_dataset_parallel(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1, shard_id=0, num_shards=8):
    """
    create dataset for train or test
    """

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081
    # get shard_id and num_shards.Get the ID of the current device in the cluster And Get the number of clusters.
    shard_id = get_rank() 
    num_shards = get_group_size()
    # define dataset
    mnist_ds = ds.MnistDataset(data_path, num_shards=num_shards, shard_id=shard_id)

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds
