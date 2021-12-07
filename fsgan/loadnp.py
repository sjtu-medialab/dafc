import numpy as np
predictions=np.load('/home/dafc/HRNet-Facial-Landmark-Detection/output/WFLW/face_alignment_wflw_hrnet_w18/predictions.npy')
print(predictions[0,0,0])