from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from pathlib import Path
import torch
# import argparse
# import os
import cv2
import numpy as np

# from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from custom_nodes.comfyui_meshhamer.config import DETECTRON2_INIT_CHECKPOINT, DEVICE

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

from vitpose_model import ViTPoseModel


def norm_depth(rend_depth):
    # 1.0 - (0.8 * (depth - minval) / (maxval - minval))
    non_zero_indices = np.nonzero(rend_depth)
    non_zero_elements = rend_depth[non_zero_indices]
    minval = non_zero_elements.min()
    maxval = non_zero_elements.max()
    depth_norm = 1.0 - (0.8 * (non_zero_elements - minval) / (maxval - minval))
    rend_depth[non_zero_indices] = depth_norm
    return rend_depth


class MeshHamerMediapipe():
    def __init__(self, checkpoint, body_detector='vitdet', rescale_factor=2, hand_detect_thr=0.25):
        """
        初始化模型
        """
        self.device = DEVICE
        self.rescale_factor = rescale_factor
        self.model, self.model_cfg = load_hamer(checkpoint, map_location='cpu')
        self.model.eval().to(self.device)
        # 定义detector
        if body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            import hamer
            cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = DETECTRON2_INIT_CHECKPOINT
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = hand_detect_thr
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
            self.detector.model.to(self.device)
        elif body_detector == 'regnety':
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py',
                                                  trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        # keypoint detector
        self.cpm = ViTPoseModel(self.device)

        # Setup the renderer
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)

    def run_inference(self, image, boxes, right):
        """
        运行模型
        """
        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(self.model_cfg, image, boxes, right, rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2 * batch['right'] - 1)
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                               scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

        return all_verts, all_right, all_cam_t

    def get_human_pose(self, image):
        # Detect humans in image
        det_out = self.detector(image)
        img = image.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        # Detect human keypoints for each person
        vitposes_out = self.cpm.predict_pose(img, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)], )
        return vitposes_out

    def get_depth(self, input_image, mask_bbox_padding, left_confidence=0.5, right_confidence=0.5):
        """
        外部接口，获取深度图
        """
        img_cv2 = input_image
        vitposes_out = self.get_human_pose(img_cv2)

        bboxes = []
        is_right = []
        info = {'hand_name': [], 'hand_keypoints': [], 'bounding_box': []}
        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]
            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > left_confidence
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)
                info['hand_name'].append('left')
                info['hand_keypoints'].append(left_hand_keyp)
            keyp = right_hand_keyp
            valid = keyp[:, 2] > right_confidence
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)
                info['hand_name'].append('right')
                info['hand_keypoints'].append(right_hand_keyp)

        if len(bboxes) == 0:
            raise Exception('No hands detected in the image')

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        all_verts, all_right, all_cam_t = self.run_inference(img_cv2, boxes, right)

        img_size = np.array(img_cv2.shape[:2:][::-1])
        scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()

        # boudding_boxes = []
        mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)

        if len(all_verts) > 0:
            ori_depths = []
            norm_depths = []
            # Render front view
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            for i in range(len(all_verts)):
                cam_view, cam_depth_ori = self.renderer.render_rgba_multiple([all_verts[i]], cam_t=[all_cam_t[i]],
                                                                             render_res=img_size,
                                                                             is_right=[all_right[i]], return_depth=True,
                                                                             **misc_args)
                ori_depths.append(cam_depth_ori)
                cam_depth = norm_depth(cam_depth_ori.copy())
                norm_depths.append(cam_depth)
                non_zero_indices = np.nonzero(cam_depth_ori)
                bouding_pos = [np.min(non_zero_indices[1]), np.max(non_zero_indices[1]), np.min(non_zero_indices[0]),
                               np.max(non_zero_indices[0])]
                mask[np.min(non_zero_indices[0]) - mask_bbox_padding:np.max(non_zero_indices[0]) + mask_bbox_padding,
                np.min(non_zero_indices[1]) - mask_bbox_padding:np.max(non_zero_indices[1]) + mask_bbox_padding] = 255

                info['bounding_box'].append(bouding_pos)

            depth_ori = np.array(ori_depths)
            depth_norm = np.array(norm_depths)
            temp_mask = depth_ori != 0
            depth_ori_masked = np.where(temp_mask, depth_ori, np.inf)
            min_channel_indices = np.argmin(depth_ori_masked, axis=0)
            rows, cols = np.meshgrid(np.arange(depth_norm.shape[1]), np.arange(depth_norm.shape[2]), indexing='ij')
            result = depth_norm[min_channel_indices, rows, cols]
            result = (result * 255).astype(np.uint8)
            return result, mask, info
        else:
            raise Exception('No hands detected in the image')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("start")
    pipeline = MeshHamerMediapipe()
    depth, mask, bouding_boxx = pipeline.get_depth(cv2.imread("/home/liufeng/Data/Code/hamer/example_data/test1.jpg"),
                                                   30)
    plt.imshow(mask, cmap='gray')
    plt.imshow(depth)
    plt.show()
    print("end")
