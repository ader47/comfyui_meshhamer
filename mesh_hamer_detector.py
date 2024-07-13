import cv2
import numpy as np
from PIL import Image
import random, torch
from custom_nodes.comfyui_meshhamer.mesh_hamer.pipline import MeshHamerMediapipe
from custom_nodes.comfyui_controlnet_aux.src.controlnet_aux.util import resize_image_with_pad,common_input_validate,HWC3

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class MeshHamerDetector:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def from_pretrained(cls, checkpoint, body_detector='vitdet',hand_detect_thr=0.25):
        pipeline = MeshHamerMediapipe(checkpoint=checkpoint, body_detector=body_detector,hand_detect_thr=hand_detect_thr)
        return cls(pipeline)

    # def to(self, device):
    #     self.pipeline._model.to(device)
    #     self.pipeline.mano_model.to(device)
    #     self.pipeline.mano_model.layer.to(device)
    #     return self

    def __call__(self, input_image=None, mask_bbox_padding=30, detect_resolution=512, output_type=None,
                 upscale_method="INTER_CUBIC", seed=88,left_confidence=0.6,right_confidence=0.6, **kwargs):
        # input_image np.array [HWC]
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        set_seed(seed, 0)
        depth_map, mask, info = self.pipeline.get_depth(input_image, mask_bbox_padding,left_confidence=left_confidence,right_confidence=right_confidence)
        # mask, depth_map np.array  [HW]
        # todo info: 2D key points, hand name(left,right) ,bounding box
        if depth_map is None:
            depth_map = np.zeros_like(input_image)
            mask = np.zeros_like(input_image)

        # The hand is small
        depth_map, mask = HWC3(depth_map), HWC3(mask)
        depth_map, remove_pad = resize_image_with_pad(depth_map, detect_resolution, upscale_method)
        depth_map = remove_pad(depth_map)
        if output_type == "pil":
            depth_map = Image.fromarray(depth_map)
            mask = Image.fromarray(mask)

        return depth_map, mask, info
