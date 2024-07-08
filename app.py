from flask import Flask, jsonify, request
app = Flask(__name__)

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp 

import cv2
import gc 
import json
import base64
import numpy as np
from pathlib import Path


def pad_left(arr, n=64): 
    deficit_x = (n - arr.shape[1] % n) 
    deficit_y = (n - arr.shape[2] % n) 
    if not (arr.shape[1] % n): 
        deficit_x = 0 
    if not (arr.shape[2] % n): 
        deficit_y = 0 
    arr = np.pad(arr, ((0, 0), (deficit_x, 0), (deficit_y, 0)), mode='reflect') 
    return arr, deficit_x, deficit_y

class AppScope:
    machineLearningModel = None

class litSegModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_name='timm-efficientnet-b1',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                              encoder_weights="noisy-student",     # use `imagenet` pre-trained weights for encoder initialization
                              in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                              decoder_attention_type= None,                      # model output channels (number of classes in your dataset)
                              classes=4)
    def forward(self, x):
        return self.model(x)

class MachineLearningModel:
    model = None
    
    def __init__(self, model_path):
        #print("Initing")

        self.model = litSegModel()
        self.model = self.model.load_from_checkpoint(model_path + 'cloudwinner_rgb3.ckpt', map_location=torch.device('cpu')).eval()

    def returnImage(self, image):
        print("return image")
        return image
        
    def prediction_step(self, image, threshold=0.5):
        image_tile = torch.from_numpy(image).float().unsqueeze(0)
        
        output = self.model(image_tile).data.cpu().numpy()[0]
        output = np.argmax(output[[0,3,2,1]],axis=0)
        #print(output.shape)
        return output.astype(np.uint8)
        
    def predict_clouds(self, img, scale_percent=10):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        x_arr = img.copy()
    
        #x_arr = np.transpose(img, [1, 2, 0])
        x_arr = cv2.resize(x_arr, dim, interpolation = cv2.INTER_AREA)
        x_arr = np.transpose(x_arr, [2, 0, 1])
        im_test, def_x, def_y = pad_left(x_arr)
        result = self.prediction_step(im_test)
        result = result[def_x:, def_y:]
    
        result = cv2.resize(result, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)
    
        return result

@app.route('/healthcheck', methods=['GET'])
def healthcheck(*args, **kwargs):
    return 'OK highrescloudrgb'

@app.route('/api/process', methods=['POST'])
def process(*args, **kwargs):
    # Do whatever you need to do when calling the endpoint
    data = request.get_json()
    imageData = data.get('image')
    shapeData = data.get('shape')
    shape = json.loads(shapeData)
    image = np.fromstring(base64.b64decode(imageData),dtype=np.float32)
    image = image.reshape(shape)
    result = AppScope.machineLearningModel.predict_clouds(image)
    resultData = base64.b64encode(result)
    return jsonify({'result': resultData.decode()})

if __name__ == "__main__":
    AppScope.machineLearningModel = MachineLearningModel('/model/')
    app.run(host="0.0.0.0", debug=True, port=8070)
