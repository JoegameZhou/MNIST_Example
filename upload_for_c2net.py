from mindspore.train.callback import Callback
import os

class UploadOutput(Callback):
    def epoch_end(self,run_context):
        os.system("cd /cache/script_for_grampus/ &&./uploader_for_npu " + "/cache/output/")
