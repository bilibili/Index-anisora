import torch
import numpy 
import cv2
import copy
def get_img_array(imgtensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """imgtensor: ([C,H,W],device=cuda)
    """
    denormimg = imgtensor.cpu().permute(1,2,0).mul_(torch.tensor(std)).add_(torch.tensor(mean))
    imgarray = denormimg.numpy()
    imgarray = imgarray * 255
    imgarray = imgarray.astype('uint8')
    imgarray = cv2.cvtColor(imgarray, cv2.COLOR_RGB2BGR)
    return imgarray

def draw_rec_in_img(img, target):
    tl = 3 # thickness line
    tf = max(tl-1,1) # font thickness
    color = [0,0,255] # color
    tempimg = copy.deepcopy(img)
    h, w = target['size']
    labels = target['labels'].cpu()
    xyxyboxes = target['xyxyboxes'].cpu()
    denorm_xyxyboxes = xyxyboxes * torch.tensor([w,h,w,h])
    for box,label in zip(denorm_xyxyboxes, labels):
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(tempimg,c1,c2,color,thickness=tl, lineType=cv2.LINE_AA)
        label = str(int(label))
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(tempimg, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(tempimg, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return tempimg

def draw_patch_in_img(img, tgt_patch, inputs_size):
    tl = 1 # thickness line
    tf = max(tl-1,1) # font thickness
    color = [0,255,0] # color
    point_size = 4
    point_color = (255, 0, 0) # BGR
    point_thickness = 4 # 可以为 0 、4、8
    tempimg = copy.deepcopy(img)
    h, w = inputs_size
    labels = tgt_patch['labels'].cpu()
    patch_indexs = tgt_patch['patch_index'].cpu()
    centers = tgt_patch['centers'].cpu()
    w_num = w//16
    for patch_index, label, center in zip(patch_indexs, labels, centers):
        point = (int(center[0]), int(center[1]))
        cv2.circle(tempimg, point, point_size, point_color, point_thickness)
        y_start_index = patch_index // w_num
        x_start_index = patch_index - y_start_index*w_num
        x_start = x_start_index * 16
        y_start = y_start_index * 16
        x_end = x_start + 16
        y_end = y_start + 16
        c1, c2 = (int(x_start), int(y_start)), (int(x_end), int(y_end))
        cv2.rectangle(tempimg,c1,c2,color,thickness=tl, lineType=cv2.LINE_AA)
        label = str(int(label))
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(tempimg, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(tempimg, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return tempimg