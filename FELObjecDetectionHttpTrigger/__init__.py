import logging
import azure.functions as func
import numpy as np
import cv2
import onnxruntime as ort
import os 



def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    dim_onnx = (800,600) # (width, height) of the model input
    base_path = "FELObjecDetectionHttpTrigger"
    labels = load_labels(os.path.join(base_path, 'FEL_classes.txt'))
    model_path = os.path.join(base_path, 'model_keen_frog_ysrb24zd.onnx')
    
    # check content type is "image/jpeg" or "image/png"
    content_type = req.headers.get('content-type')
    if content_type == 'image/jpeg' or content_type == 'image/png':
        img_data = req.get_body()

    if img_data:    
        img = decode_byte_arr(img_data, dim_onnx)
        outputs = run_model(model_path, img)
        return func.HttpResponse(f"The output: {map_outputs(outputs,dim_onnx,labels)}")
    else:
        return func.HttpResponse(
                "This HTTP triggered function executed successfully. Pass a byte array image representation or json object containing 'img' in the query string or in the request body for a personalized response.",
                status_code=200)


def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

# decode image
def decode_byte_arr(img_byte_arr, dim_onnx):
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
def map_outputs(outputs, dim_onnx, labels):
    """
    outputs: list of 3 arrays: boxes, classes, scores
    dim_onnx: tuple of (width, height) of the model input
    return output; dict with keys "filename", "boxes"
        - filename: string
        - boxes: list of dicts with keys "box", "label", "score"
    """
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

