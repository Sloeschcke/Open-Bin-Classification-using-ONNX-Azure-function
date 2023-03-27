import logging
import azure.functions as func
import numpy as np
import cv2
import onnxruntime as ort
import os 

''' 
    This function is a HTTP trigger function that takes an image as input and returns the output of the model.
    The model is a trained object detection model in ONNX format.
    The model is loaded in the init() function and is reused for each invocation of the function.
    The function is triggered by a HTTP POST request and expects the image to be passed as a byte array in the request body.
    The function returns the output as expected by the frontend:
         a dict with a value "boxes" that contains list of dicts with keys "box", "label", "score"
'''
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    dim_onnx = (800,600) # (width, height) of the model input
    base_path = "FELObjecDetectionHttpTrigger" 
    labels = load_labels(os.path.join(base_path, 'FEL_classes.txt')) # load labels
    model_path = os.path.join(base_path, 'model_keen_frog_ysrb24zd.onnx') # load model
    
    img_data = None
    content_type = req.headers.get('content-type')
    if content_type == 'image/jpeg' or content_type == 'image/png':
        img_data = req.get_body()
    elif 'content-type' not in req.headers:
        return func.HttpResponse(
            "Bad Request: content-type header is missing from the request",
            status_code=400)

    if img_data:    
        img = decode_byte_arr(img_data, dim_onnx)
        outputs = run_model(model_path, img)
        return func.HttpResponse(f"The output: {map_outputs(outputs,dim_onnx,labels)}")
    else:
        return func.HttpResponse(
                "This HTTP triggered function executed successfully. Pass a byte array image representation or json object containing 'img' in the query string or in the request body for a personalized response.",
                status_code=200)


def preprocess(img_data):
    ''' Preprocess the image data in the same way the original training data was preprocessed.
    Use Imagenet mean and standard deviation.
    :param img_data: The image data to be preprocessed.  (channel, height, width)
    :return: The preprocessed image data - (channel, height, width)
    '''
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

# decode image
def decode_byte_arr(img_byte_arr, dim_onnx):
    ''' Decode byte array of image and preprocess (normalize) it for model input
    img_byte_arr: byte array of image
    dim_onnx: tuple of (width, height) of the model input
    return img: numpy array of image
    '''
    img = np.frombuffer(img_byte_arr, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR) # BGR
    img = cv2.resize(img, dim_onnx) # HWC
    img = img.transpose((2,0,1)) # HWC to CHW
    img = img.reshape(1, 3, dim_onnx[0], dim_onnx[1]) # add batch dimension
    img = preprocess(img)
    img = img.reshape(1, 3, dim_onnx[1], dim_onnx[0]) # B C H W
    return img

# run model on image
def run_model(model_path, img):
    ''' Run ONNX model on image
    model_path: path to model
    img: numpy array of image
    return outputs: list of 3 arrays: boxes, classes, scores
    '''
    ort_sess = ort.InferenceSession(model_path)
    inputs = ort_sess.get_inputs()[0].name
    outputs = ort_sess.run(None, {inputs: img})
    return outputs

#load text file as list
def load_labels(path):
    ''' Loads labels file. Supports files with or without index numbers.
    :param path: path to labels file
    :return: a list of labels
    '''
    labels = []
    with open(path, 'r') as f:
        for line in f:
            labels.append(line.strip())
    return labels


def get_box_dims(image_shape, box):
    ''' Convert box coordinates from model output to normalized values
    image_shape: tuple of (width, height) of the model input
    box: array of 4 values: topX, topY, bottomX, bottomY
    return box_dims: dict with keys "topX", "topY", "bottomX", "bottomY"
    '''
    box_keys = ['topX', 'topY', 'bottomX', 'bottomY']
    width, height = image_shape[0], image_shape[1]

    box_dims = dict(zip(box_keys, [coordinate.item() for coordinate in box]))

    box_dims['topX'] = box_dims['topX'] * 1.0 / width
    box_dims['bottomX'] = box_dims['bottomX'] * 1.0 / width
    box_dims['topY'] = box_dims['topY'] * 1.0 / height
    box_dims['bottomY'] = box_dims['bottomY'] * 1.0 / height

    return box_dims

# map mobilenet outputs to classes
def map_outputs(outputs, dim_onnx, labels):
    ''' Map model outputs to classes and box dimensions (output format used by the frontend)
    outputs: list of 3 arrays: boxes, classes, scores
    dim_onnx: tuple of (width, height) of the model input
    return output; dict with keys "filename", "boxes"
        - filename: string
        - boxes: list of dicts with keys "box", "label", "score"
    '''
    # map classes to label strings
    boxes = outputs[0]
    classes = outputs[1]
    scores = outputs[2]
    pred_labels = [labels[i] for i in classes]
  
    output = {"filename": "test.jpg", "boxes": []}
    for i in range(len(boxes)):
        box_raw = boxes[i]
        box = get_box_dims(dim_onnx, box_raw)
        output["boxes"].append({"box": box, "label": pred_labels[i], "score": scores[i]})

    return output
