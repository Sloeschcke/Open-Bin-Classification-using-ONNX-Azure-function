import logging
import azure.functions as func
import base64
import numpy as np
import cv2
import io
import onnxruntime as ort


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    dim_onnx = (800,600) # (width, height) of the model input
    img_base64 = req.params.get('img')
    if not img_base64:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            img_base64 = req_body.get('img')

    # get bytearray from request 
    img_byte_arr = req.get_body()
    # decode image as bytearray to get numpy array
    img = np.frombuffer(img_byte_arr, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, dim_onnx)
    

    if img_base64:
        img = decode_base64(img_base64, dim_onnx)
        model_path = 'FELObjecDetectionHttpTrigger\model_keen_frog_ysrb24zd.onnx'
        outputs = run_model(model_path, img)

        return func.HttpResponse(f"The output: {map_outputs(outputs,dim_onnx)}")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a img_base64 in the query string or in the request body for a personalized response.",
             status_code=200
        )

def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

# decode base64 image
def decode_base64(data, dim_onnx):
    img = base64.b64decode(data)
    img = np.fromstring(img, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, dim_onnx)
    img = img.transpose((2,0,1))
    img = img.reshape(1, 3, dim_onnx[0], dim_onnx[1])
    img = preprocess(img)
    img = img.reshape(1, 3, dim_onnx[1], dim_onnx[0])
    return img

# run model on image
def run_model(model_path, img):
    ort_sess = ort.InferenceSession(model_path)
    inputs = ort_sess.get_inputs()[0].name
    outputs = ort_sess.run(None, {inputs: img})
    return outputs

#load text file as list
def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            labels.append(line.strip())
    return labels


def get_box_dims(image_shape, box):
    box_keys = ['topX', 'topY', 'bottomX', 'bottomY']
    width, height = image_shape[0], image_shape[1]

    box_dims = dict(zip(box_keys, [coordinate.item() for coordinate in box]))

    box_dims['topX'] = box_dims['topX'] * 1.0 / width
    box_dims['bottomX'] = box_dims['bottomX'] * 1.0 / width
    box_dims['topY'] = box_dims['topY'] * 1.0 / height
    box_dims['bottomY'] = box_dims['bottomY'] * 1.0 / height

    return box_dims

# map mobilenet outputs to classes
def map_outputs(outputs, dim_onnx):
    """
    outputs: list of 3 arrays: boxes, classes, scores
    dim_onnx: tuple of (width, height) of the model input
    return output; dict with keys "filename", "boxes"
        - filename: string
        - boxes: list of dicts with keys "box", "label", "score"
    """
    labels = load_labels('./FELObjecDetectionHttpTrigger/FEL_classes.txt')
    # map classes to label strings
    boxes = outputs[0]
    classes = outputs[1]
    scores = outputs[2]
    pred_labels = [labels[i] for i in classes]
  
    output = {"filename": "test.jpg", "boxes": []}
    for i in range(len(boxes)):
        box_raw = boxes[i]
        box = get_box_dims(dim_onnx, box_raw)
        #box = {"topX": box["topX"], "topY": box["topY"], "bottomX": box["bottomX"], "bottomY": box["bottomY"]}

        output["boxes"].append({"box": box, "label": pred_labels[i], "score": scores[i]})

    return output

