# cd eval
import torch
import numpy as np
import pickle


def cal_re(pred_landmarks, gt_landmarks, scale_rate):
    pred_landmarks = pred_landmarks.cpu().detach().numpy()
    gt_landmarks = gt_landmarks.cpu().detach().numpy()
    scale_rate_y = scale_rate[0]
    scale_rate_y = scale_rate_y.item() if type(scale_rate_y) == torch.Tensor else scale_rate_y
    scale_rate_x = scale_rate[1]
    scale_rate_x = scale_rate_x.item() if type(scale_rate_x) == torch.Tensor else scale_rate_x
    c = pred_landmarks.shape[0]
    diff = np.zeros([c, 2], dtype=float)  # y, x
    for i in range(c):
        diff[i][0] = abs(pred_landmarks[i][0] - gt_landmarks[i][0]) * scale_rate_y
        diff[i][1] = abs(pred_landmarks[i][1] - gt_landmarks[i][1]) * scale_rate_x
    Radial_Error = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
    Radial_Error = Radial_Error.mean()
    return Radial_Error

if __name__ == '__main__':
    file_path = '../final_runs/hand_63/pred/5577.pkl'
    file2_path = '../final_runs/hand_63/pseudo_labels/5577.pkl'

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    with open(file2_path, 'rb') as f:
        data2 = pickle.load(f)
        
    res = np.stack((data[0][0], data[0][1]),axis=1)
    res = torch.from_numpy(res).float()

    gt = data[1]
    landmark_tensor = torch.tensor([[pair[0], pair[1]] for pair in gt]).float()

    re = cal_re(res, landmark_tensor, (1, 1))
    print(re)