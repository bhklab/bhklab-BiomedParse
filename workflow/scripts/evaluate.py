# TAKEN FROM: https://github.com/bhklab/medsam2-inference/blob/main/evaluate.py

import numpy as np
import monai.metrics as met
import torch

class Evaluator:
    def __init__(
        self, 
        metrics: list[str] = [
            "volume_dice", 
            "jaccard", 
            "hausdorff", 
            "surface_dice", 
            "panoptic_quality",
            "added_path_length", 
            "false_negative_volume", 
            "false_negative_path_length"
        ]
    ):
        self.metrics = metrics

    def __call__(self, preds: np.ndarray, targets: np.ndarray, spacing: tuple[float, float, float]) -> dict:
        results = {}
        
        for metric in self.metrics:
            match metric:
                case "volume_dice":
                    results[metric] = self._volume_dice(preds, targets)
                case "jaccard":
                    results[metric] = self._jaccard(preds, targets)
                case "hausdorff":
                    results[metric] = self._hausdorff(preds, targets, spacing)
                case "surface_dice":
                    results[metric] = self._surface_dice(preds, targets, spacing)
                case "panoptic_quality":
                    results[metric] = self._panoptic_quality(preds, targets)
                case "added_path_length":
                    results[metric] = self._apl(preds, targets)
                case "false_negative_volume":
                    results[metric] = self._fnv(preds, targets)
                case "false_negative_path_length":
                    results[metric] = self._fnpl(preds, targets)
                case _:
                    raise ValueError(f"Metric {metric} not supported")
        
        return results   

    def _volume_dice(self, preds: np.ndarray, targets: np.ndarray) -> float:
        dc = (np.sum(preds[targets == 1]) * 2.0) / (np.sum(preds) + np.sum(targets))
        return dc
    
    def _jaccard(self, preds: np.ndarray, targets: np.ndarray) -> float:
        j = np.sum(preds[targets == 1]) / (np.sum(preds) + np.sum(targets) - np.sum(preds[targets == 1]))
        return j

    def _hausdorff(self, preds: np.ndarray, targets: np.ndarray, spacing: tuple[float, float, float]) -> float:
        preds = torch.from_numpy(preds).unsqueeze(0).unsqueeze(0)
        targets = torch.from_numpy(targets).unsqueeze(0).unsqueeze(0)
        h = met.compute_hausdorff_distance(preds, targets, percentile=95, include_background=False, spacing=spacing)
        return h[0].item()

    def _surface_dice(self, preds: np.ndarray, targets: np.ndarray, spacing: tuple[float, float, float]) -> float:
        preds = torch.from_numpy(preds).unsqueeze(0).unsqueeze(0)
        targets = torch.from_numpy(targets).unsqueeze(0).unsqueeze(0)
        s = met.compute_average_surface_distance(preds, targets, include_background=False, spacing=spacing)
        return s[0].item()

    def _panoptic_quality(self, preds: np.ndarray, targets: np.ndarray) -> float:
        preds = torch.from_numpy(preds).unsqueeze(0).unsqueeze(0)
        targets = torch.from_numpy(targets).unsqueeze(0).unsqueeze(0)
        pq = met.compute_panoptic_quality(preds, targets, output_confusion_matrix=True)
        return pq[0].item()

    def _apl(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return AddedPathLength(preds, targets)
    
    def _fnv(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return FalseNegativeVolume(preds, targets)
    
    def _fnpl(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return FalseNegativePathLength(preds, targets)

"""
TAKEN FROM: https://github.com/kkiser1/Autosegmentation-Spatial-Similarity-Metrics/

Each function takes "auto" and "gt" arguments, which are respectively the autosegmentation and ground truth 
segmentation represented as three-dimensional NumPy arrays. The array dimensions should be the dimensions 
of the original image, and each array element should be 0 if its corresponding image pixel is not part of 
the segmentation or 1 if it is.
"""

def FalseNegativeVolume(auto, gt):
    '''
    Returns the false negative volume, in pixels
    
    Steps:
    1. Find pixels where the mask is present in gt but not in auto (wherever gt is 1 but auto is 0)
    2. Convert comparison from bool to int
    3. Compute # pixels
    '''
    
    fnv = (gt > auto).astype(int).sum()
    return fnv


def AddedPathLength(auto, gt):
    '''
    Returns the added path length, in pixels
    
    Steps:
    1. Find pixels at the edge of the mask for both auto and gt
    2. Count # pixels on the edge of gt that are not in the edge of auto
    '''
    
    # Check if auto and gt have same dimensions. If not, then raise a ValueError
    if auto.shape != gt.shape:
        raise ValueError('Shape of auto and gt must be identical!')

    # edge_auto has the pixels which are at the edge of the automated segmentation result
    edge_auto = getEdgeOfMask(auto)
    # edge_gt has the pixels which are at the edge of the ground truth segmentation
    edge_gt = getEdgeOfMask(gt)
    
    # Count # pixels on the edge of gt that are on not in the edge of auto
    apl = (edge_gt > edge_auto).astype(int).sum()
    
    return apl 


def FalseNegativePathLength(auto, gt):
    '''
    Returns the false negative path length, in pixels
    
    Steps:
    1. Find pixels at the edge of the mask for gt
    2. Count # pixels on the edge of gt that are not in the auto mask volume
    '''
    
    # Check if auto and gt have same dimensions. If not, then raise a ValueError
    if auto.shape != gt.shape:
        raise ValueError('Shape of auto and gt must be identical!')
    
    # edge_gt has the pixels which are at the edge of the ground truth segmentation
    edge_gt = getEdgeOfMask(gt)
    
    # Count # pixels where the edges in grount truth == 1 and auto == 0
    fnpl = (edge_gt > auto).astype(int).sum() 
    
    return fnpl

def getEdgeOfMask(mask):
    '''
    Computes and returns edge of a segmentation mask
    '''
    # edge has the pixels which are at the edge of the mask
    edge = np.zeros_like(mask)
    
    # mask_pixels has the pixels which are inside the mask of the automated segmentation result
    mask_pixels = np.where(mask > 0)

    for idx in range(0,mask_pixels[0].size):

        x = mask_pixels[0][idx]
        y = mask_pixels[1][idx]
        z = mask_pixels[2][idx]

        # Count # pixels in 3x3 neighborhood that are in the mask
        # If sum < 27, then (x, y, z) is on the edge of the mask
        if mask[x-1:x+2, y-1:y+2, z-1:z+2].sum() < 27:
            edge[x,y,z] = 1
            
    return edge