# Predicting with pretrained Models
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt  #Predicting with pretrained models

# Loading a pretrained model
net = model_zoo.yolo3_darknet53_voc('F:/models/yolo3_darknet53_voc', pretrained=True)
#net = model_zoo.yolo3_darknet53_coco('F:/models/yolo3_darknet53_coco', pretrained=True)

# Preprocess an image
file = utils.makedirs('F:/input')
file = 'F:/input/me.jpg'
#file = 'F:/input/lowres.mp4'
x, img = data.transforms.presets.yolo.load_test(file, short=512)
print('Shape of pre-processed video:', x.shape)

# Inference and Display
class_IDs, scores, bounding_boxs = net(x)
ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=net.classes)
plt.show()