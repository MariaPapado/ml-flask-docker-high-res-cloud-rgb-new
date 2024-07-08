import cv2
import json
import base64
import requests
import numpy as np
from datetime import datetime
from shapely import geometry
import rasterio
from pimsys.regions.RegionsDb import RegionsDb
import matplotlib.pyplot as plt

def normalise_bands(image, percentile_min=2, percentil_max=98):
    tmp = []
    for i in range(image.shape[0]):
        perc_2 = np.percentile(image[i, :, :], percentile_min)
        perc_98 = np.percentile(image[i, :, :], percentil_max)
        band = (image[i, :, :] - perc_2) / (perc_98 - perc_2)
        band[band < 0] = 0.
        band[band > 1] = 1.
        tmp.append(band)
    return np.array(tmp)

def load_tif_image(layer):
    # Define mapserver path
    mapserver_path = '/cephfs/mapserver/data'
    # Get path to after image
    layer_datetime = layer['capture_timestamp']
    path_tif = '/'.join([mapserver_path, layer_datetime.strftime('%Y%m%d'), layer['wms_layer_name']]) + '.tif'
    # Load image
    tif = rasterio.open(path_tif)
    img = tif.read()
    tif.close()
    # Normalize bands on min and max
    img = normalise_bands(img)
    # Normalize
    # img = exposure.equalize_adapthist(np.moveaxis(img, 0, -1), 100)
    # img = np.moveaxis(img, -1, 0)
    # Get tif bounds
    image_bounds = list(tif.bounds)
    image_poly = geometry.Polygon.from_bounds(image_bounds[0], image_bounds[1], image_bounds[2], image_bounds[3])
    return img, tif.transform, image_bounds, image_poly

width = 512 
height = 512

addr = 'http://10.140.0.19:8056'
test_url = addr + '/api/process'

coords = [-61.648133585982364, 10.189181664489293]

config_db = {
        "host": "sar-ccd-db.orbitaleye.nl",
        "port": 5433,
        "user": "postgres",
        "password": "sarccd-db",
        "database": "sarccd2"
    }

database = RegionsDb(config_db)
layer_1 = database.get_optical_images_containing_point_in_period(coords, [1666631662 - 1000, 1666631662 + 1000])[0]
database.close()

image_before, tif_transform, image_bounds, image_poly = load_tif_image(layer_1)

image_before = image_before.transpose((1,2,0))

print(image_before.shape)

image_before = (image_before).astype(np.float32)[:,:,:3]

image_before_d = base64.b64encode(np.ascontiguousarray(image_before))

response = requests.post(test_url, json={'image':image_before_d.decode(),'shape': json.dumps(list(image_before.shape))})


if response.ok:
    response_result = json.loads(response.text)
    response_result_data = base64.b64decode(response_result['result'])
    result = np.fromstring(response_result_data,dtype=np.uint8)
    
    
res = result.reshape(image_before.shape[:2])

#print(res.mean() == 0.07530838699986857)
plt.imshow(res)
plt.savefig('figure.png')
