'''BiomedParse Inference Script for 2026 CIHR Spring'''

import pandas as pd 
import SimpleITK as sitk
import torch
import numpy as np
import gc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.measure import regionprops, label
from pathlib import Path
from prompt_testing import apply_windowing, rescale_imgs_255, biomedparse_preproc, initialize_model, do_inference, load_prompt_skeletons, calc_metrics, pos_neg_true_visual, mid_slice_visual, plot_hist, plot_density

def aaura_site_windowing(disease_site: str): 
    '''   
    Decides what window level and width to use based on the anatomical disease site.

    Parameters
    ----------
    disease_site: str
        Where in the body the disease is located (e.g. abdomen, lung, etc.)

    Returns
    ----------
    window_level: int 
        The centering value of the window 
    window_width: int
        The width of the window
    '''
    match disease_site: 
        case 'abdomen': 
            window_level = 40
            window_width = 400
        case 'lung': 
            # This is the self-described BiomedParse preprocessing window for lung, notably different window level 
            # than traditional (-600) 
            window_level = -160
            window_width = 1500
        case 'headneck': 
            window_level = 40
            window_width = 400
        case _:
            raise ValueError(f"Invalid dataset name: {disease_site}. Please check spelling or add to this function with the correct window and level")
        
    return window_level, window_width

def run_one_inference(img_path: Path, 
                      tumour_site: str,
                      text_prompt: dict, 
                      model, 
                      device: torch.device, 
                      save_folder: str): 
    '''  
    Run inference on an individual image and mask pair and save the corresponding output

    Parameters
    ----------
    img_path: Path
        The path to the current CT image data. Also used to create the save path for images.
        NOTE: We are assuming only one inference per image, so there should be no overwritting. 
    tumour_site: str
        Where in the body the tumour is located (e.g. abdomen, lung, headneck, etc.). To be used to 
        determine appropriate windowing.
    text_prompt: dict
        A singular text prompt to use. Assumes it is in BiomedParse-compatible format.
        NOTE: This assumes that the prompts do not require further processing.
    model: 
        The BiomedParse model created after initialization with the checkpoint
    device: torch.device
        Device used in model initialization to be used for predictions
    save_folder: str
        Where to save the predicted images. Replaces the "images" part of the original imaging path.
    '''
    # Load in image 
    ct_img_raw = sitk.ReadImage(img_path) 
    ct_img_arr_raw = sitk.GetArrayFromImage(ct_img_raw)

    # Choose windowing to use based on disease site 
    win_lvl, win_width = aaura_site_windowing(disease_site = tumour_site)

    # Window image to BiomedParse windowing preferences
    ct_img_arr = apply_windowing(img_array = ct_img_arr_raw, 
                                 window_level = win_lvl, 
                                 window_width = win_width)
    
    # Rescale image to [0, 255]
    ct_img_arr_rescaled = rescale_imgs_255(img_array = ct_img_arr)

    # Prepare save paths 
    dataset = str(img_path).split("/")[3] # Assumes data structure of "data/procdata/<disease_site>/<dataset>/..."
    
    if dataset == 'CVPR_LesionLocator': # Do check for most appropriate file saving structure.
        disease_site = 'MultiSite'
    elif dataset == 'OCSCC_RADCURE':
        disease_site == 'HeadNeck'
    else:
        disease_site = tumour_site.capitalize()

    base_savepath = Path("data/results") / disease_site / "/".join(str(img_path).split("/")[-1:]).replace("images", save_folder)
    if not base_savepath.exists(): 
        base_savepath.mkdir(parents = True, exist_ok = True)
    
    img_savepath = base_savepath / 'predicted_whole.nii.gz'
    print(f"Running inference for {base_savepath} with prompt {text_prompt}.")

    # Perform preprocessing
    input_tensor, pad_width, padded_size, valid_axis, ids = biomedparse_preproc(image = ct_img_arr_rescaled, 
                                                                                text_prompt = text_prompt, 
                                                                                device = device)
    # Perform inference 
    pred_mask = do_inference(in_tensor = input_tensor, 
                             pad_width = pad_width, 
                             padded_size = padded_size, 
                             valid_axis = valid_axis, 
                             ids = ids,
                             model = model)

    # Get the predicted mask into an image and ensure same spacing, orientation and direction 
    pred_mask_img = sitk.GetImageFromArray(pred_mask)

    pred_mask_img.SetSpacing(ct_img_raw.GetSpacing())
    pred_mask_img.SetOrigin(ct_img_raw.GetOrigin())
    pred_mask_img.SetDirection(ct_img_raw.GetOrigin())

    sitk.WriteImage(pred_mask_img, img_savepath)

    # Clean to save memory
    gc.collect() 
    del ct_img_raw, ct_img_arr, ct_img_arr_raw, ct_img_arr_rescaled, input_tensor, pred_mask_img, pred_mask
    torch.cudda.empty_cache()

def metrics_visuals_whole(index_df: pd.DataFrame, 
                          img_path: Path, 
                          save_folder: str, 
                          prompt: str): 
    '''
    For a given image, evaluate the performance metrics and get visualizations based upon the combined 
    masks related to an image. 

    Parameters
    ----------
    index_df: pd.DataFrame
        The index dataframe containing all paths and necessary information for preprocessing
    img_path: Path
        The path to the current medical image 
    save_folder: str
        Where to save the predicted images. Replaces the "images" part of the original imaging path.
    prompt: str
        The text prompt used to generate the results

    Returns
    ----------
    whole_metrics: pd.DataFrame
        The performance of the whole segmentation to the combined mask of all lesions for a given image. 
    '''
    # Get dataset from image path
    dataset = str(img_path).split("/")[0] # Assumes data structure of "<disease_site>/<dataset>/..."

    # Subset index file by image path to isolate matched lesions 
    img_idx_subset = index_df[index_df['image_path'] == img_path].reset_index() 

    tumour_site = img_idx_subset['lesion_location'].iloc[0]
    if dataset == 'CVPR_LesionLocator': # Do check for most appropriate file saving structure.
        disease_site = 'MultiSite'
    elif dataset == 'OCSCC_RADCURE':
        disease_site == 'HeadNeck'
    else:
        disease_site = tumour_site.capitalize()
    
    # Combine all ground truth masks 
    if img_idx_subset.shape[0] == 1: 
        temp_path = Path('data/procdata') / disease_site / img_idx_subset['mask_path'].iloc[0]
        mask_img = sitk.ReadImage(temp_path) 
        comb_mask = sitk.GetArrayFromImage(mask_img)

        # Get largest slice for visualization later 
        largest_slice = img_idx_subset['largest_slice_index'].iloc[0]
    
    else:
        for idx in range(img_idx_subset.shape[0]-1): # Only need range up to second last index b/c addition below
            if idx == 0: 
                # Load in both the first and second masks and combine them 
                mask_path_1 = Path('data/procdata' / disease_site / img_idx_subset['mask_path'].iloc[idx])
                mask_path_2 = Path('data/procdata' / disease_site / img_idx_subset['mask_path'].iloc[idx+1])

                mask_img_1 = sitk.ReadImage(mask_path_1) 
                mask_img_2 = sitk.ReadImage(mask_path_2)

                mask_arr_1 = sitk.GetArrayFromImage(mask_img_1) 
                mask_arr_2 = sitk.GetArrayFromImage(mask_img_2)

                comb_mask = mask_arr_1 | mask_arr_2

                # Get largest slice of all tumours
                slice1 = mask_arr_1[img_idx_subset['largest_slice_index'].iloc[idx]]
                slice2 = mask_arr_2[img_idx_subset['largest_slice_index'].iloc[idx+1]]

                # Get major axis lengths for both slices                 
                props1 = regionprops(slice1)[0]
                props2 = regionprops(slice2)[0]

                diam1 = props1.major_axis_length
                diam2 = props2.major_axis_length

                if diam1 > diam2: 
                    largest_slice = slice1
                    largest_diam = diam1
                elif diam1 < diam2: 
                    largest_slice = slice2
                    largest_diam = diam2
                else:
                    largest_slice = slice1 # If they're equal, just choose one of them
                    largest_diam = diam1
            else: 
                # Only need to load in the subsequent index and combine it with the previously combined masks 
                mask_path = Path('data/procdata' / disease_site / img_idx_subset['mask_path'].iloc[idx+1])
                mask_img = sitk.ReadImage(mask_path) 
                mask_arr = sitk.GetArrayFromImage(mask_img) 

                comb_mask = comb_mask | mask_arr 
                
                # Get the largest slice of the current mask 
                slice_n = mask_arr[img_idx_subset['largest_slice_index'].iloc[idx+1]]

                props_n = regionprops(slice_n)[0]
                diam_n = props_n.major_axis_length

                # Compare previous largest slice to the current tumour 
                if diam_n >= largest_diam: 
                    largest_slice = slice_n
                    largest_diam = diam_n                

    # Load in predicted mask 
    pred_mask_path = Path("data/results") / disease_site / "/".join(str(img_path).split("/")[-1:]).replace("images", save_folder) / 'predicted_whole.nii.gz'
    pred_mask_img = sitk.ReadImage(pred_mask_path)
    pred_mask = sitk.GetArrayFromImage(pred_mask_img) 

    # Load in image and get spacing
    img = sitk.ReadImage(img_path) 
    img_arr = sitk.GetArrayFromImage(img)
    img_spacing = img.GetSpacing()

    # Window image for visualization 
    win_lvl, win_width = aaura_site_windowing(disease_site = disease_site)
    img_arr_win = apply_windowing(img_array = img_arr, 
                                  window_level = win_lvl, 
                                  window_width = win_width)
    # Calculate metrics 
    whole_results_df = calc_metrics(pred_mask = pred_mask, 
                                    gt_mask = comb_mask, 
                                    spacing = img_spacing, 
                                    filename = pred_mask_path, 
                                    text_prompt = prompt)
    
    # Only make visualizations if the mask is non-empty
    visual_basepath = str(pred_mask_path).removesuffix('.nii.gz')
    if np.count_nonzero(pred_mask) > 0:
        # This iteration I only want histogram count and the largest tumour slice, the others are set up more for individual evaluation
        # Pixel count histogram # 
        pix_hist_savepath = Path(visual_basepath + "_pixhist.png")

        plot_hist(gt_mask = comb_mask, 
                  pred_mask = pred_mask, 
                  text_prompt = prompt, 
                  full_savepath = pix_hist_savepath)
        
        # Largest slice visualization #
        slice_savepath = Path(visual_basepath + "_largestslice.png")
        mid_slice_visual(image = img_arr_win, 
                         mask_preds = pred_mask, 
                         gt_masks = comb_mask, 
                         text_prompts = prompt,
                         mid_slice = largest_slice, 
                         full_savepath = slice_savepath)
        
    else: 
        print(f'Predicted segmentation is empty for patient {img_path}')

    return whole_results_df

def list_nonzero_seg_slices(seg: np.ndarray): 
    '''  
    From a given 3D segmentation array, list the slices that have nonzero values (mask) in them.

    Parameters
    ----------
    seg: np.ndarray
        A 3D array containing a mask (ground truth, predicted, etc.)
    
    Returns
    ----------
    nonzero_slices: list 
        Contains all of the slice numbers where there are nonzero values
    '''
    nonzero_slices = []
    for slice_idx in range(seg.shape[0]): 
        if np.count_nonzero(seg[slice_idx]) > 0: 
            nonzero_slices.append(slice_idx)
    return nonzero_slices

def bbox_from_seg(seg_2d: np.ndarray): 
    '''  
    Get a bounding box from a given segmentation (binary array). Assumes that only one tumour is within this segmentation.

    Parameters
    ----------
    seg2d: np.ndarray
        The segmentation slice to get a bounding box from 

    Returns
    ----------
    bbox: list 
        A list of coordinates for the bounding box in the order of [xmin, ymin, xmax, ymax]
    '''
    rows = np.any(seg_2d, axis=1)
    cols = np.any(seg_2d, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    bbox = [xmin, ymin, xmax, ymax]
    
    return bbox

def calc_2d_IoU(bbox_gt: np.ndarray,
                bbox_pred: np.ndarray):
    '''  
    Calculate the 2D intersection-over-union (IoU) between two bounding boxes. 

    Parameters
    ----------
    bbox_gt: np.ndarray
        Ground truth 2D bounding box in the order [xmin, ymin, xmax, ymax]
    bbox_pred: np.ndarray
        Predicted 2D bounding box in the order [xmin, ymin, xmax, ymax]

    Returns 
    ----------
    iou: float 
        The IoU of the two bounding boxes
    '''
    # Get coordinates and area of intersecting rectangle
    xmin_inter = max(bbox_gt[0], bbox_pred[0])
    ymin_inter = max(bbox_gt[1], bbox_pred[1])
    xmax_inter = min(bbox_gt[2], bbox_pred[2])
    ymax_inter = min(bbox_gt[3], bbox_pred[3])

    inter_width = max(0, xmax_inter-xmin_inter)
    inter_height = max(0, ymax_inter-ymin_inter)
    inter_area = inter_width * inter_height 

    # Get area of the each bounding box 
    gt_area = (bbox_gt[2]-bbox_gt[0]) * (bbox_gt[3]-bbox_gt[1])
    pred_area = (bbox_pred[2]-bbox_pred[0]) * (bbox_pred[3]-bbox_pred[1])

    # Calculate union area 
    union_area = gt_area + pred_area - inter_area 

    # Calculate IoU 
    if union_area == 0: 
        return 0.0 # This should never happen, but implementing just in case
    
    iou = inter_area / union_area 

    return iou

def iou_visual(image: np.ndarray, 
               pred: np.ndarray, 
               gt: np.ndarray, 
               iou: float,
               largest_slice: int, 
               full_savepath: Path): 
    '''  
    Creates and saves a visual of the ground truth and predicted bounding boxes on the largest tumour area slice with 
    the IoU measurement in the title. 

    Parameters
    ----------
    image: np.ndarray
        The current imaging being used 
    pred: np.ndarray 
        The predicted segmentation mask 
    gt: np.ndarray 
        The ground truth segmentation mask 
    iou: float
        The IoU metric for the current tumour 
    largest_slice: int
        The index of the slice with the largest tumour area 
    full_savepath: Path
        Where to save the visualization to
    '''
    # Get bounding boxes with width and height for both ground truth and predicted segmentations
    gt_bbox = bbox_from_seg(gt[largest_slice]) 
    pred_bbox = bbox_from_seg(pred[largest_slice]) 

    gt_width = gt_bbox[2] - gt_bbox[0] # width from xmax-xmin 
    gt_height = gt_bbox[3] - gt_bbox[1] # height from ymax-ymin

    pred_width = pred_bbox[2] - pred_bbox[0] # width from xmax-xmin 
    pred_height = pred_bbox[3] - pred_bbox[1] # height from ymax-ymin

    # Prep plot patches for bounding boxes
    gt_patch = mpatches.Rectangle((gt_bbox[0], gt_bbox[1]), gt_width, gt_height, 
                                  linewidth=1, 
                                  edgecolor='c', 
                                  facecolor='none', 
                                  label='Ground Truth') 
    pred_patch = mpatches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_width, pred_height, 
                                    linewidth=1, 
                                    edgecolor='m', 
                                    facecolor='none', 
                                    label='Prediction') 
    
    # Plot visualization
    fig, ax = plt.subplots(figsize = (8,8))

    ax.imshow(image, cmap="gray") # Assumes image is already windowed to correct level and width for visualization
    ax.add_patch(gt_patch) 
    ax.add_patch(pred_patch)
    ax.axis("off")
    ax.set_title(f"IoU = {iou:.2f}")

    fig.legend(handles=[gt_patch, pred_patch], loc='lower center', ncol=2)

    # Save visualization 
    fig.savefig(full_savepath, bbox_inches = 'tight')

def indiv_metric_visuals(index_df: pd.DataFrame, 
                         img_path: Path, 
                         save_folder: str, 
                         prompt: str, 
                         thresh_inc: int = 10): 
    '''  
    For all predicted masks, separate all connected components, match them to there appropriate ground truth segmentations
    (where applicable), calculate individual metrics, and produce visualizations for each correctly predicted tumour
    segmentation. 

    Parameters
    ----------
    index_df: pd.DataFrame
        The index dataframe containing all paths and necessary information for preprocessing
    img_path: Path
        The path to the current medical image 
    save_folder: str
        Where to save the predicted images. Replaces the "images" part of the original imaging path.
    prompt: str
        The text prompt used to generate the results
    thresh_inc: int
        The number of voxels each individual predicted segmentation needs to have to be considered for matching. Default 10.
    Returns
    ----------
    indiv_metrics: pd.DataFrame
        The performance of the individual segmentations to the ground truth masks for a given image. 
    false_negatives: pd.DataFrame
        Contains the information from the index file of the tumours that did not get a predicted segmentation.
    false_positives: pd.DataFrame
        Contains information about the connected components in the predicted segmentation which did not have a successful match.
    '''
    # Get dataset from image path
    dataset = str(img_path).split("/")[0] # Assumes data structure of "<disease_site>/<dataset>/..."

    # Subset index file by image path to isolate matched lesions 
    img_idx_subset = index_df[index_df['image_path'] == img_path].reset_index() 

    tumour_site = img_idx_subset['lesion_location'].iloc[0]
    if dataset == 'CVPR_LesionLocator': # Do check for most appropriate file saving structure.
        disease_site = 'MultiSite'
    elif dataset == 'OCSCC_RADCURE':
        disease_site == 'HeadNeck'
    else:
        disease_site = tumour_site.capitalize()

    # Load in predicted segmentation 
    pred_mask_path = Path("data/results") / disease_site / "/".join(str(img_path).split("/")[-1:]).replace("images", save_folder) / 'predicted_whole.nii.gz'
    pred_mask_img = sitk.ReadImage(pred_mask_path)
    pred_mask = sitk.GetArrayFromImage(pred_mask_img) 

    # Load in image and get spacing
    img = sitk.ReadImage(img_path) 
    img_arr = sitk.GetArrayFromImage(img)
    img_spacing = img.GetSpacing()

    # Load in ground truth segmentations 
    gt_segs = dict()
    for idx in len(img_idx_subset.shape[0]): 
        # Load in the current mask 
        curr_mask_path = Path('data/procdata' / disease_site / img_idx_subset['mask_path'].iloc[idx])
        curr_gt_mask = sitk.ReadImage(curr_mask_path)
        curr_gt_arr = sitk.GetArrayFromImage(curr_gt_mask) 
        gt_segs[curr_mask_path] = curr_gt_arr 

    # Separate the connected components from the predicted mask if it isn't empty, get matches into dictionary, and 
    # record false positives and negatives 
    if np.count_nonzero(pred_mask) > 0: 
        indiv_comps = []
        gt_pred_matches = dict()
        conn_comps, num_comps = label(pred_mask, return_num = True)

        for comp_idx in range(1, num_comps + 1): # num_comps is the maximum index range, and need that last index included to get all comps 
            curr_comp = np.ma.masked_where(conn_comps != comp_idx, conn_comps).filled(0) # Masks out all other components
            if np.count_nonzero(curr_comp) < thresh_inc: 
                continue # Ignore this component as it has less voxels than the threhsold for inclusion

            # Get current component as a binary array 
            curr_comp_bin = curr_comp / comp_idx 
            indiv_comps.append(curr_comp_bin)

            # Check if there are any overlaps with the individual ground truth segmentations 
            match_found = False
            for key, val in gt_segs.items(): 
                curr_intersect = np.bitwise_and(val, curr_comp_bin)
                if np.count_nonzero(curr_intersect) > 0: 
                    # Indicates overlap between the two segmentations, so these have matched. Save as a pair for further analysis
                    gt_pred_matches[str(comp_idx)]['gts_path'] = key
                    gt_pred_matches[str(comp_idx)]['gts'] = val
                    gt_pred_matches[str(comp_idx)]['pred'] = curr_comp_bin
                    
                    # Replace current ground truth segmentation in the gt_segs array with 'match_found' to keep 
                    # track of false negatives 
                    gt_segs[key] = 'match_found'
                    match_found = True
            
            # If no match was found, record it as a false positive
            if not match_found: 
                false_pos = True
                nonzero_slices = list_nonzero_seg_slices(curr_comp_bin)
                voxels = np.count_nonzero(curr_comp_bin)
                temp_info = [img_path, comp_idx, nonzero_slices, voxels]
                temp_df = pd.DataFrame([temp_info], columns = ['image_path', 'conn_comp_idx', 'nonzero_slices', 'num_voxels'], index = 0)
                if 'false_positives' not in locals(): 
                    false_positives = temp_df
                else: 
                    false_positives = pd.concat([false_positives, temp_df], ignore_index = True).reset_index(drop = True) 
        
        # If there are no false positives, prep to return empty dataframe 
        if not false_pos: 
            false_positives = pd.DataFrame()

        # If a ground truth segmentation did not have a successful match, record as false negative 
        gt_no_match = [key for key, value in gt_segs.items() if value != 'match_found']
        for no_match in gt_no_match:
            temp_df = img_idx_subset[img_idx_subset['mask_path'] == no_match]
            if 'false_negatives' not in locals(): 
                false_negatives = temp_df
            else: 
                false_negatives = pd.concat([false_negatives, temp_df], ignore_index = True).reset_index(drop = True) 
        else: 
            false_negatives = pd.DataFrame() # No false negatives, so prep to return empty dataframe 
        
    else: 
        # If the prediction is empty, then consider all individual tumours as false negatives.
        false_negatives = img_idx_subset
        false_positives = pd.DataFrame() 
        results_df = pd.DataFrame()

        return results_df, false_negatives, false_positives

    # If there were any matches, record metrics and visualizations for each
    if gt_pred_matches: 
        # Window image for visualization 
        win_lvl, win_width = aaura_site_windowing(disease_site = disease_site)
        img_arr_win = apply_windowing(img_array = img_arr, 
                                    window_level = win_lvl, 
                                    window_width = win_width)
        
        for key, value in gt_pred_matches.items(): 
            curr_results_df = calc_metrics(pred_mask = value['pred'], 
                                      gt_mask = value['gts'], 
                                      spacing = img_spacing, 
                                      filename = value['gts_path'])
            # Get largest slice for the current ground truth segmentation 
            curr_seg_row = img_idx_subset[img_idx_subset['mask_path'] == value['gts_path']].reset_index(drop = True)
            curr_largest_slice = curr_seg_row['largest_slice_index']

            # Get bounding boxes for IoU calc
            gt_bbox = bbox_from_seg(value['gts'][curr_largest_slice])
            pred_bbox = bbox_from_seg(value['pred'][curr_largest_slice])

            # Calculate IoU 
            curr_iou = calc_2d_IoU(bbox_gt = gt_bbox, 
                                   bbox_pred = pred_bbox) 
            results_df['2D_IoU'] = curr_iou
            results_df['conn_comp'] = key # This is the connected component index from the matched predicted mask in case it needs reference later. Connected component calculations should be reproducable. 
            results_df['prompt'] = prompt

            # Get visualizations
            visual_basepath = str(pred_mask_path).removesuffix('.nii.gz')

            # True Positive, False Positive, False Negative plot # 
            pos_neg_savepath = Path(visual_basepath + "_posneg_" + str(key) + ".png")

            pos_neg_true_visual(image = img_arr_win, 
                                mask_preds = value['pred'], 
                                gt_masks = value['gts'], 
                                full_savepath = pos_neg_savepath)
            
            # IoU visualization
            iou_savepath = Path(visual_basepath + "_iou_" + str(key) + ".png") 

            iou_visual(image = img_arr_win, 
                       pred = value['pred'],
                       gt = value['gts'], 
                       iou = curr_iou, 
                       largest_slice = curr_largest_slice, 
                       full_savepath = iou_savepath)

            # Save current results into results dataframe 
            if 'results_df' not in locals(): 
                results_df = curr_results_df
            else: 
                results_df = pd.concat([results_df, curr_results_df], ignore_index = True).reset_index(drop = True) 
        
        return results_df, false_negatives, false_positives
            
    else: 
        results_df = pd.DataFrame()

        return results_df, false_negatives, false_positives
    
def run_all_inference(aaura_idx: Path,
                      checkpoint_path: Path, 
                      prompt_path: Path,
                      save_folder: str,
                      n_jobs: int
                      ): 
    '''  
    Run the prompt testing across a specific dataset based on
    the data management plan structure. 

    Parameters
    ----------
    aaura_idx: Path
        The csv that has all relevant information for CT images and their corresponding masks
    checkpoint_path: Path
        Where the checkpoint file is located for model initialization.
    prompt_path: Path
        Where the .yaml config file is for the text prompts.
    save_folder: str
        Where to save the outputs to. Just needs to be one folder name, not a whole path. 
    n_jobs: int
        How many jobs to run in parallel during the prompt testing.
    '''
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load in index csv
    index_df = pd.read_csv(aaura_idx) 

    # Initialize model and load prompt skeletons
    model, device = initialize_model(ckpt_path = checkpoint_path)
    prompt_skels = load_prompt_skeletons(prompt_config_path = prompt_path) # NOTE: Current iteration of script assumes only one text prompt in config with a specific key

    # For the current prompt, predict segmentations 
    text_prompt = prompt_skels['TEXT_PROMPTS']['AAURA_PROMPT']

    # Run predictions in parallel 
    img_paths = index_df['image_path'].unique()
    Parallel(n_jobs = n_jobs)(delayed(run_one_inference)(img_path = curr_path, 
                                                         tumour_site = index_df[index_df['image_path'] == curr_path].reset_index(drop = True)['lesion_location'].iloc[0], 
                                                         text_prompt = text_prompt, 
                                                         model = model, 
                                                         device = device, 
                                                         save_folder = save_folder)
                                                         for curr_path in (tqdm(img_paths, 
                                                                           desc = "Running inference using BiomedParse", 
                                                                           total = len(img_paths))))
    
    # Get results (both whole and individual) from the predictions 
    # Get savepath 
    dataset = str(img_paths[0]).split("/")[0]

    tumour_site = index_df['lesion_location'].iloc[0]

    if dataset == 'CVPR_LesionLocator': # Do check for most appropriate file saving structure.
        disease_site = 'MultiSite'
    elif dataset == 'OCSCC_RADCURE':
        disease_site == 'HeadNeck'
    else:
        disease_site = tumour_site.capitalize()

    # Define savepath for metrics
    # NOTE: This path should exist form the inference section, if it doesn't, something has gone very wrong
    out_path = Path("data/results") / disease_site / "/".join(str(img_paths[0]).split("/")[:3]).replace("images", save_folder)
    if not out_path.exists(): 
        raise FileNotFoundError(f"The specified save path does not exist yet. Please run inference first or check the inputted path: {out_path}")

    whole_metrics = Parallel(n_jobs = n_jobs)(delayed(metrics_visuals_whole)(index_df = index_df, 
                                                                            img_path = curr_path, 
                                                                            save_folder = save_folder, 
                                                                            prompt = text_prompt["1"])
                                                                            for curr_path in (tqdm(img_paths, 
                                                                                            desc = "Running metrics on combined masks", 
                                                                                            total = len(img_paths))))

    for metric in whole_metrics: 
        if not metric.empty(): 
            if 'all_whole_metrics' not in locals(): 
                all_whole_metrics = metric 
            else: 
                all_whole_metrics = pd.concat([all_whole_metrics, metric], ignore_index = True).reset_index(drop = True) 
    
    all_whole_metrics.to_csv(Path(out_path / 'combined_mask_metric_eval.csv'), index = False)
    del all_whole_metrics

    indiv_metrics, false_negatives, false_positives = zip(*Parallel(n_jobs = n_jobs)(delayed(indiv_metric_visuals)(index_df = index_df, 
                                                                                                                    img_path = curr_path, 
                                                                                                                    save_folder = save_folder, 
                                                                                                                    prompt = text_prompt["1"])
                                                                                                                    for curr_path in (tqdm(img_paths, 
                                                                                                                                    desc = "Running metrics on individual masks", 
                                                                                                                                    total = len(img_paths)))))

    for metric in indiv_metrics: 
        if not metric.empty(): 
            if 'all_indiv_metrics' not in locals(): 
                all_indiv_metrics = metric 
            else: 
                all_indiv_metrics = pd.concat([all_indiv_metrics, metric], ignore_index = True).reset_index(drop = True) 
    
    all_indiv_metrics.to_csv(Path(out_path / 'indiv_mask_metric_eval.csv'), index = False)
    del all_indiv_metrics 

    for false_neg in false_negatives: 
        if not false_neg.empty(): 
            if 'all_false_neg' not in locals(): 
                all_false_neg = false_neg 
            else: 
                all_false_neg = pd.concat([all_false_neg, false_neg], ignore_index = True).reset_index(drop = True)
    
    all_false_neg.to_csv(Path(out_path / 'indiv_false_negatives.csv'), index = False)
    del all_false_neg

    for false_pos in false_positives: 
        if not false_pos.empty(): 
            if 'all_false_pos' not in locals(): 
                all_false_pos = false_pos
            else: 
                all_false_pos = pd.concat([all_false_pos, false_pos], ignore_index = True).reset_index(drop = True) 

    all_false_pos.to_csv(Path(out_path / 'indiv_false_positives.csv'), index = False) 
    del all_false_pos 