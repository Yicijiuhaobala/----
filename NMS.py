import torch
def iou(box1, box2):
    area1 = (box1[:, 2]-box1[:, 0])*(box1[:, 3]-box1[:, 1])
    area2 = (box2[:, 2]-box2[:, 0])*(box2[:, 3]-box2[:, 1])
    box1_num = box1.size(0)
    box2_num = box2.size(0)
    box1_box2_length = box1.size(1)
    box1_dela = box1.unsqueeze(1).expand(box1_num,box2_num,box1_box2_length)
    box2_dela = box2.unsqueeze(0).expand(box1_num,box2_num,box1_box2_length)
    area_cross_left_top = torch.maximum(box1_dela[:,:,:2],box2_dela[:,:,:2])
    area_cross_right_bottom = torch.minimum(box1_dela[:,:,2:],box2_dela[:,:,2:])
    area_cross_w = area_cross_right_bottom[:,:,0]-area_cross_left_top[:,:,0]
    area_cross_h = area_cross_right_bottom[:,:,1]-area_cross_left_top[:,:,1]
    area_cross = area_cross_h*area_cross_w

    area1_dela = area1.unsqueeze(1).expand(box1_num,box2_num)
    area2_dela = area2.unsqueeze(0).expand(box1_num,box2_num)
    iou = area_cross/(area1_dela+area2_dela-area_cross)
    return box1_dela,box2_dela,area_cross_left_top,area_cross_right_bottom, area_cross_w, area_cross_h,area_cross,iou

def nms(boxes, scores, threshold):
    area = (boxes[:,2]-boxes[:,0])*(boxes[:3]-boxes[:,1])
    _ , scores_sort_index_tensor = torch.sort(scores,descending=True)
    keep = []
    while scores_sort_index_tensor.numel>0 :
        if scores_sort_index_tensor.numel==1:
            index = scores_sort_index_tensor
            keep.append(index)
        else:
            index = scores_sort_index_tensor[0]
            keep.append(index)
        x1 = torch.clamp(boxes[:,0][scores_sort_index_tensor[1:]],min=boxes[:,0][index]) 
        y1 = torch.clamp(boxes[:,1][scores_sort_index_tensor[1:]],min=boxes[:,1][index])
        x2 = torch.clamp(boxes[:,2][scores_sort_index_tensor[1:]],max=boxes[:,2][index])
        y2 = torch.clamp(boxes[:,3][scores_sort_index_tensor[1:]],max=boxes[:,3][index])
        area_cross = (x2-x1)*(y2-y1)
        iou = area_cross/(area[index]+area[1:]-area_cross)
        iou_index = (iou<=threshold).nonzeros.squeeze(1)
        scores_sort_index_tensor = scores_sort_index_tensor[iou_index+1]

    

box1 = torch.arange(start=0,end=16).reshape((4,4))
box2 = torch.arange(start=20, end=40).reshape((5,4))
print(box1)
print(box2)
a,b,c,d,e,f,g,h= nms(box1,box2)
print(a, b, c, d, e, f,g,h)
