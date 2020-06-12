from fastai.vision import *
from fastai.metrics import accuracy, Recall
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 1033120000
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorboardX import SummaryWriter
from trains import Task
task = Task.init(project_name = 'Valid ID Classify',
                task_name = 'Stage 2 - Mobile Net - fastai - main training 2')
import warnings
warnings.filterwarnings("ignore")

data = ImageDataBunch.from_folder(path = 'data', train = 'Train', valid = 'Valid', size=224, bs=128, no_check = True
                                  ).normalize(([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

writer = SummaryWriter()

# save model class
class SaveBestModel(Recorder):
    def __init__(self, learn):
        super().__init__(learn)
        self.name = 0
        self.save_method = self.save_when_acc

    def save_when_acc(self, metrics):
        loss, accuracy, recall_0, precision_0, F1_0, recall_front, precision_front, F1_front, recall_back, precision_back, F1_back = float(metrics[0]), float(metrics[1]), float(metrics[2]), float(metrics[3]), float(metrics[4]), float(metrics[5]), float(metrics[6]), float(metrics[7]), float(metrics[8]), float(metrics[9]), float(metrics[10])
        writer.add_scalar('data_main/loss', loss, self.name)
        writer.add_scalar('data_main/accuracy', accuracy, self.name)
        writer.add_scalar('data_0/recall', recall_0, self.name)
        writer.add_scalar('data_0/precision', precision_0, self.name)
        writer.add_scalar('data_0/F1', F1_0, self.name)
        writer.add_scalar('data_front/recall', recall_front, self.name)
        writer.add_scalar('data_front/precision', precision_front, self.name)
        writer.add_scalar('data_front/F1', F1_front, self.name)
        writer.add_scalar('data_back/recall', recall_back, self.name)
        writer.add_scalar('data_back/precision', precision_back, self.name)
        writer.add_scalar('data_back/F1', F1_back, self.name)
        self.learn.save('fastai_v1_' + str(self.name))
        self.name += 1
            
    def on_epoch_end(self, last_metrics=MetricsList, **kwargs: Any):
        self.save_method(last_metrics)

learn = cnn_learner(data, models.mobilenet_v2, metrics= [accuracy, Recall(pos_label = 0), Precision(pos_label = 0), FBeta(pos_label = 0), 
                                                         Recall(pos_label = 2), Precision(pos_label = 2), FBeta(pos_label = 2),
                                                         Recall(pos_label = 1), Precision(pos_label = 1), FBeta(pos_label = 1)], callback_fns = SaveBestModel)

learn.load('fastai_v1_1')
learn.unfreeze()
learn.fit_one_cycle(20)