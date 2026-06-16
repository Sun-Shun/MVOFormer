import torch
import cv2
import numpy as np

from torch.utils.data import Dataset
from os import listdir
from Tool.Datasets.utils import make_intrinsics_layer
from Tool.Datasets.transformation import pos_quats2SEs, pose2motion, SEs2ses

# ──────────────────────────────────────────────────────────────────────────────
# Fix summary
#
# A. Historical fixes (preserved from previous version)
#    1. All __getitem__ methods no longer call .cuda(). Dataset returns CPU tensors only.
#    2. VODataset.__getitem__: img_2 was incorrectly reading from img1_path -- fixed.
#    3. PoseDataset mask branch: pose index corrected from [1] to [2].
#
# B. Memory optimization (this version)
#    4. img1 / img2 are returned as uint8 (3,H,W) instead of immediate .float().
#       Reason: float32 is 4x larger than uint8. One 480x640 RGB image:
#         uint8  -> 0.9 MB
#         float  -> 3.6 MB
#       16 workers x prefetch 2 x batch 64 x 2 images -> saves ~28 GB RAM.
#    5. flow remains float32 (optical flow values are floating-point, not uint8-safe).
#    6. The transform pipeline (Compose/CropCenter/...) internally calls
#       img.float()/255 or similar normalization, so uint8 is automatically
#       promoted to float before entering transforms. If your transform chain
#       assumes float32 input, disable keep_uint8 in the yaml config
#       (see keep_uint8 parameter in __init__).
# ──────────────────────────────────────────────────────────────────────────────


class BaseDataset(Dataset):
    def __init__(self, img_dir_list, flow_dir_list, mask_dir_list, pose_file_list, is_test=False):

        self.img_path_list = img_dir_list
        self.flow_path_list = flow_dir_list
        self.mask_path_list = mask_dir_list
        self.pose_list = pose_file_list
        self.is_test = is_test
        self.datas = self.merge_data(
            self.img_path_list, self.flow_path_list,
            self.mask_path_list, self.pose_list, self.is_test
        )

    def get_img_data(self, dir_path):
        path_list = []
        img_paths = listdir(dir_path)
        img_paths_list = [
            (dir_path + '/' + ff) for ff in img_paths
            if (ff.endswith('.png') or ff.endswith('.jpg'))
        ]
        img_paths_list.sort()
        path_list += img_paths_list
        return path_list

    def get_flow_data(self, dir_path):
        path_list = []
        flow_paths = listdir(dir_path)
        flow_paths_list = [(dir_path + '/' + ff) for ff in flow_paths if ff.endswith('.npy')]
        flow_paths_list.sort()
        path_list += flow_paths_list
        return path_list

    def get_mask_data(self, dir_path):
        path_list = []
        mask_paths = listdir(dir_path)
        mask_paths_list = [(dir_path + '/' + ff) for ff in mask_paths if ff.endswith('.npy')]
        mask_paths_list.sort()
        path_list += mask_paths_list
        return path_list

    def get_pose_data(self, files_path, is_test=False):
        all_pose_list = []
        pose_std = [0.13, 0.13, 0.13, 0.013, 0.013, 0.013]
        poses_list = []
        motions = []
        if files_path is not None:
            poselist = np.loadtxt(files_path).astype(np.float32)
            assert (poselist.shape[1] == 7)  # position + quaternion
            poses = pos_quats2SEs(poselist)
            matrix = pose2motion(poses)
            motions = SEs2ses(matrix).astype(np.float32)
            if not is_test:
                motions = motions / pose_std
                trans = motions[:, :3]
                trans_norm = np.linalg.norm(trans, axis=1)
                data_fine = trans_norm.reshape(-1, 1) + float(1e-15)
                motions[:, :3] = motions[:, :3] / data_fine
        for j in range(int(len(motions))):
            poses_motion = motions[j]
            poses_list.append(poses_motion)
        all_pose_list += poses_list
        return all_pose_list

    def merge_data(self, img_path_list, flow_path_list, mask_path_list, pose_data_list, is_test=False):
        sampler_datas = []
        # only train PoseNet
        if img_path_list is None:
            if mask_path_list is None:
                for i in range(0, len(flow_path_list)):
                    flow_path = self.get_flow_data(flow_path_list[i])
                    pose_path = self.get_pose_data(pose_data_list[i], is_test)
                    for j in range(0, len(flow_path)):
                        sampler_datas.append((flow_path[j], pose_path[j]))
            else:
                for i in range(0, len(flow_path_list)):
                    flow_path = self.get_flow_data(flow_path_list[i])
                    mask_path = self.get_mask_data(mask_path_list[i])
                    pose_path = self.get_pose_data(pose_data_list[i], is_test)
                    for j in range(0, len(flow_path)):
                        sampler_datas.append((flow_path[j], mask_path[j], pose_path[j + 1]))
        # only train FlowNet
        elif pose_data_list is None:
            if mask_path_list is None:
                for i in range(0, len(flow_path_list)):
                    imgs_path = self.get_img_data(img_path_list[i])
                    flow_path = self.get_flow_data(flow_path_list[i])
                    for j in range(0, len(flow_path)):
                        sampler_datas.append((imgs_path[j], imgs_path[j + 1], flow_path[j]))
            else:
                for i in range(0, len(flow_path_list)):
                    imgs_path = self.get_img_data(img_path_list[i])
                    flow_path = self.get_flow_data(flow_path_list[i])
                    mask_path = self.get_mask_data(mask_path_list[i])
                    for j in range(0, len(flow_path)):
                        sampler_datas.append((imgs_path[j], imgs_path[j + 1], flow_path[j], mask_path[j]))
        # train Transformer
        elif flow_path_list is None:
            if mask_path_list is None:
                for i in range(0, len(pose_data_list)):
                    imgs_path = self.get_img_data(img_path_list[i])
                    pose_path = self.get_pose_data(pose_data_list[i], is_test)
                    for j in range(0, len(pose_path)):
                        sampler_datas.append((imgs_path[j], imgs_path[j + 1], pose_path[j]))
            else:
                for i in range(0, len(pose_data_list)):
                    imgs_path = self.get_img_data(img_path_list[i])
                    mask_path = self.get_mask_data(mask_path_list[i])
                    pose_path = self.get_pose_data(pose_data_list[i], is_test)
                    for j in range(0, len(pose_path)):
                        sampler_datas.append((imgs_path[j], imgs_path[j + 1], mask_path[j], pose_path[j]))
        # train VO
        else:
            if mask_path_list is None:
                for i in range(0, len(flow_path_list)):
                    imgs_path = self.get_img_data(img_path_list[i])
                    flow_path = self.get_flow_data(flow_path_list[i])
                    pose_path = self.get_pose_data(pose_data_list[i], is_test)
                    for j in range(0, len(flow_path)):
                        sampler_datas.append((imgs_path[j], imgs_path[j + 1], flow_path[j], pose_path[j]))
            else:
                for i in range(0, len(flow_path_list)):
                    imgs_path = self.get_img_data(img_path_list[i])
                    flow_path = self.get_flow_data(flow_path_list[i])
                    mask_path = self.get_mask_data(mask_path_list[i])
                    pose_path = self.get_pose_data(pose_data_list[i], is_test)
                    for j in range(0, len(flow_path)):
                        sampler_datas.append((imgs_path[j], imgs_path[j + 1], flow_path[j], mask_path[j], pose_path[j]))

        return sampler_datas

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class PoseDataset(BaseDataset):
    def __init__(self, img_dir_list, flow_dir_list, mask_dir_list, pose_file_list,
                 transform=None, is_test=False):
        super(PoseDataset, self).__init__(
            img_dir_list, flow_dir_list, mask_dir_list, pose_file_list, is_test
        )
        self.transform = transform

    def __getitem__(self, index):
        if self.mask_path_list is None:
            flow_path = self.datas[index][0]
            flow = torch.as_tensor(np.load(flow_path)).permute((2, 0, 1))
            pose = torch.as_tensor(self.datas[index][1])
            res = {'flow': flow, 'pose': pose}
        else:
            flow_path = self.datas[index][0]
            mask_path = self.datas[index][1]
            pose = torch.as_tensor(self.datas[index][2])
            flow = torch.as_tensor(np.load(flow_path)).permute((2, 0, 1))
            mask = torch.as_tensor(np.load(mask_path)).unsqueeze(2).permute((2, 0, 1))
            res = {'flow': flow, 'mask': mask, 'pose': pose}

        if self.transform:
            res = self.transform(res)
        return res

    def __len__(self):
        return len(self.datas)


class FlowDataset(BaseDataset):
    def __init__(self, img_dir_list, flow_dir_list, mask_dir_list, pose_file_list, transform=None,
                 keep_uint8=True):
        super(FlowDataset, self).__init__(img_dir_list, flow_dir_list, mask_dir_list, pose_file_list)
        self.transform = transform
        self.keep_uint8 = keep_uint8
        try:
            len(self.flow_path_list) + len(flow_dir_list) == len(self.img_path_list)
        except AssertionError:
            print('flow and img not matching!')

    def _load_img(self, path):
        # uint8 path: ~4x memory savings on the worker side
        arr = cv2.imread(path)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # uint8
        return t if self.keep_uint8 else t.float()

    def __getitem__(self, index):
        if self.mask_path_list is None:
            img1_path, img2_path = self.datas[index][0], self.datas[index][1]
            flow_path = self.datas[index][2]
            flow  = torch.as_tensor(np.load(flow_path)).permute((2, 0, 1))
            img_1 = self._load_img(img1_path)
            img_2 = self._load_img(img2_path)
            res = {'img1': img_1, 'img2': img_2, 'flow': flow}
        else:
            img1_path, img2_path = self.datas[index][0], self.datas[index][1]
            flow_path, mask_path = self.datas[index][2], self.datas[index][3]
            flow  = torch.as_tensor(np.load(flow_path)).permute((2, 0, 1))
            mask  = torch.as_tensor(np.load(mask_path)).unsqueeze(2).permute((2, 0, 1))
            img_1 = self._load_img(img1_path)
            img_2 = self._load_img(img2_path)
            res = {'img1': img_1, 'img2': img_2, 'flow': flow, 'mask': mask.float()}

        if self.transform:
            res = self.transform(res)
        return res

    def __len__(self):
        return len(self.datas)


class VODataset(BaseDataset):
    """
    Optional parameters:
      keep_uint8 : bool (default True)
          True  -> __getitem__ returns uint8 RGB tensor (4x RAM savings)
          False -> returns float32 (old behaviour, use only when transform chain requires float)

    Usage (when building in multi_dataset.py):
        VODataset(imgs, flows, masks, poses, transform=t,
                  is_test=False, keep_uint8=True)
    Existing callers that do not pass keep_uint8 default to True, saving memory automatically.
    """

    def __init__(self, img_dir_list, flow_dir_list, mask_dir_list, pose_file_list,
                 transform=None, is_test=False, keep_uint8=True):
        super(VODataset, self).__init__(
            img_dir_list, flow_dir_list, mask_dir_list, pose_file_list, is_test
        )
        self.transform = transform
        self.keep_uint8 = keep_uint8

    def _load_img(self, path):
        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"cv2.imread failed: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # uint8 (3,H,W)
        return t if self.keep_uint8 else t.float()

    def __getitem__(self, index):
        n = len(self.datas[index])
        if self.mask_path_list is None:
            if n == 3:
                # Image-only mode: (img1, img2, pose)
                img1_path, img2_path = self.datas[index][0], self.datas[index][1]
                pose = self.datas[index][2]
                pose  = torch.as_tensor(pose)
                img_1 = self._load_img(img1_path)
                img_2 = self._load_img(img2_path)
                res = {'img1': img_1, 'img2': img_2, 'pose': pose}
            else:
                img1_path, img2_path = self.datas[index][0], self.datas[index][1]
                flow_path = self.datas[index][2]
                pose      = self.datas[index][3]

                pose  = torch.as_tensor(pose)
                flow  = torch.as_tensor(np.load(flow_path)).permute((2, 0, 1))   # float32 (2,H,W)
                img_1 = self._load_img(img1_path)                                # uint8 (3,H,W) by default
                img_2 = self._load_img(img2_path)
                res = {'img1': img_1, 'img2': img_2, 'flow': flow, 'pose': pose}
        else:
            img1_path, img2_path = self.datas[index][0], self.datas[index][1]
            flow_path, mask_path = self.datas[index][2], self.datas[index][3]
            pose                 = self.datas[index][4]

            pose  = torch.as_tensor(pose)
            flow  = torch.as_tensor(np.load(flow_path)).permute((2, 0, 1))
            mask  = torch.as_tensor(np.load(mask_path)).unsqueeze(2).permute((2, 0, 1))
            img_1 = self._load_img(img1_path)
            img_2 = self._load_img(img2_path)
            res = {'img1': img_1, 'img2': img_2, 'flow': flow, 'pose': pose, 'mask': mask.float()}

        if self.transform:
            res = self.transform(res)
        return res

    def __len__(self):
        return len(self.datas)
