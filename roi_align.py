import torch
import numpy as np
from torchvision.ops import nms, roi_align

fp = torch.tensor(list(range(6 * 5))).float()
fp = fp.view(1, 3, 2, 5)
feat = fp.data.cpu().numpy()
np.save("./data/align_feat.npy", feat)

boxes = torch.tensor([[0, 0.25, 0.45, 1.73, 3.42]]).float()
rois = boxes.data.cpu().numpy()
np.save("./data/align_rois.npy", rois)
pooled_features = roi_align(fp, boxes, [4, 4])
print(pooled_features)
