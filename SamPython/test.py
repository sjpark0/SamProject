from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
        
def original(image, input_point, input_label):
    sam_checkpoint = 'weights/sam_vit_h_4b8939.pth'
    model_type = "vit_h"

    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    start = time.time()
    masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True,)
    end = time.time()
    print(end - start, "sec")

    print(masks.shape)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show() 
        
    #mask_generator = SamAutomaticMaskGenerator(sam)
    #masks = mask_generator.generate(image)
    #print(len(masks))
    #print(masks[0].keys())

     
    #return masks

def export_test(image, input_point, input_label):
    import onnxruntime
    onnx_model_path_preprocess = '../models/sam_onnx_preprocess.onnx'
    onnx_model_path = '../models/sam_onnx_example.onnx'
    
    #EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    EP_list = ['CUDAExecutionProvider']

    ort_session_preprocess = onnxruntime.InferenceSession(onnx_model_path_preprocess, provider=EP_list)
    ort_session = onnxruntime.InferenceSession(onnx_model_path, provider=EP_list)

    #print(onnxruntime.get_device())
    
    #sam_checkpoint = 'Weight/sam_vit_h_4b8939.pth'
    #model_type = "vit_h"
    #sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    #print(sam.image_encoder.img_size)
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()

    #predictor = SamPredictor(sam)
    #predictor.set_image(image)
    
    #image_embedding = predictor.get_image_embedding().cpu().numpy()
    #print("embedding shape : ", image_embedding.shape)
    
    from segment_anything.utils.transforms import ResizeLongestSide
    transform = ResizeLongestSide(1024)
    newimage = transform.apply_image(image)
    newimage = np.transpose(newimage, [2, 0, 1])
    newimage = newimage[np.newaxis,...]
    #print(newimage.shape)
    ort_inputs_preprocess = {
        "input" : newimage
    }
    image_embedding = ort_session_preprocess.run(None, ort_inputs_preprocess)
    #print(image_embedding)
    #print("embedding shape : ", image_embedding[0].shape)
    
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

    onnx_coord = transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    #onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    print("label : " , onnx_label)
    print("point : ", onnx_coord)
    
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    ort_inputs = {
        #"image_embeddings": image_embedding[0],
        "image_embeddings": image_embedding[0],
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }
    
    masks, scores, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > 0.0
    print(masks.shape)
    print(scores)
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show() 
    #for i, (mask, score) in enumerate(zip(masks[0,:], scores[0,:])):
    #    plt.imshow(image)
    #    show_mask(mask, plt.gca())
    #    show_points(input_point, input_label, plt.gca())
    #    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #    plt.axis('off')
    #    plt.show()
    #plt.show() 
    
#image = plt.imread('../Data/000.png')
input_point = np.array([[2132, 1144]])
input_label = np.array([1])
#image = (image * 255).astype('uint8')
image = cv2.imread("../Data/000.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#original(image, input_point, input_label)
export_test(image, input_point, input_label)

