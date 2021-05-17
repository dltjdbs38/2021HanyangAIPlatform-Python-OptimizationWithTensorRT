import torch
import urllib
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import tensorrt as trt
from torch2trt import tensorrt_converter
import torch2trt
import logging
import time

#model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)

TRT_LOGGER = trt.Logger() #log를 만들어 주기 위해서 Lab3에 있는 
trt_runtime = trt.Runtime(TRT_LOGGER) #log를 만들어 주기 위해 Lab3에 있는 코드 복붙



# Download an example image from the pytorch website

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename) #filename으로 불러오기

# sample execution (requires torchvision)

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


model = models.mobilenet_v2(pretrained=True)
model.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

#model을 tensorRT로 변환해줌. 그떄 쓰는 모듈(함수)가 torch2trt (모델,인풋,에다가 로그 주기 위해 log_level=~~)
model = torch2trt.torch2trt(model, input_batch, log_level = trt.Logger.Severity.VERBOSE)

start = time.time()
with torch.no_grad():
    output = model(input_batch)

sec = time.time() - start
print("Inference Time(second) : ", sec)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)


# Download ImageNet labels
#!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top1_prob, top1_catid = torch.topk(probabilities, 1) # list가 아니라 각각 나온다 5개였으면 리스트 5개씩 나오는데
# 그냥 가장 높은 확률인 카테고리 1개만 출력
print(categories[top1_catid], top1_prob.item())


