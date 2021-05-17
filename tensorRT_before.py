import torch
import urllib
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import tensorrt as trt
from torch2trt import tensorrt_converter
import logging

model = models.mobilenet_v2(pretrained=True) # 미리 학습된 models 속 mobilenet_v2라는 모델을 가져옴.
model.eval() # 이 model.eval()이 없으면 학습하는 줄 알고 weight를 업데이트 시킴. 그러므로 우리는
# 그냥 running후 결과값만 보여주면 되므로 model.eval을 한다.


# Download an example image from the pytorch website

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename) # 추론에 사용할 url에서 이미지를 불러옴. 'filename'으로 저장한다.

# sample execution (requires torchvision)

input_image = Image.open(filename) # 
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), # 이미지 형태인데 tensor형태로 추론에 활용될 수 있도록 변형해줌.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 전처리.
])
input_tensor = preprocess(input_image) #input_batch가 우리가 추론에 사용할 데이터.
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available(): #GPU가 만약 사용가능하다면, 저 이미지를
    input_batch = input_batch.to('cuda') #input_batch를 cuda(GPU)로 보낸다.
    model.to('cuda') #model 도 cuda(GPU)로 보낸다.

start = time.time() # 지금 현재 절대시간이 나온다. 9시 3분 33초
with torch.no_grad(): # 학습을 안 한다. 업데이트를 안 한다. = no_grad
    output = model(input_batch) # output= 최종 결과
sec = time.time() - start #sec=현재시간-start했을 때(9시 3분 33초) = 추론하는데 걸린 시간.
print("Before tensorrt, Inference Time(second):", sec)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0) # 확률이 제일 높은 걸 뽑아내는 것. softmax로

# Download ImageNet labels
#!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

# Read the categories
with open("imagenet_classes.txt", "r") as f: # label을 0~99의 숫자를 카테고리로 받아옴. 숫자-> 사물,개..종류로
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
# top5_prob, top5_catid = torch.topk(probabilities, 5) # 가장 확률이 높은 5개를 뽑아온다. topk는 torch라는 모듈
# #안에 있는 함수 이름.
top1_prob, top1_catid = torch.topk(probabilities, 1) # 가장 확률이 높은 확률(prob)(0~1), 카테고리(catid)(이건 0~99숫자) 1개만 출력
# for i in range(top5_prob.size(0)):
print(categories[top1_catid], top1_prob.item()) #categories 함수는 3->고양이 로 바꾸기 위해.


