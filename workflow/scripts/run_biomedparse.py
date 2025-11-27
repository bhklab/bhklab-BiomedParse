import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
import hydra
import click
import math

from evaluate import Evaluator
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from utils import process_input, process_output
from inference import postprocess, merge_multiclass_masks
from pathlib import Path
from joblib import Parallel, delayed
from skimage.measure import regionprops, label
from tqdm import tqdm

from skimage import segmentation
from skimage.measure import label

def initialize_model(ckpt_path: Path):
    '''
    Initialize BiomedParse using the most recent checkpoint.

    Parameters
    ----------
    ckpt_path: Path
        Where the checkpoint file is located. Should have been downloaded from HuggingFace and put into a known location.
    
    Returns
    ----------
    model: src.model.biomedparse_3D.BiomedParseModel 
        The model object initialized using the given checkpoint
    ''' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    GlobalHydra.instance().clear()
    hydra.initialize(config_path="configs/model", job_name="example_prediction")
    cfg = compose(config_name="biomedparse_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")
    print(ckpt_path)
    model.load_pretrained(ckpt_path)
    model = model.to(device).eval()

    return model, device

def npzs_from_existing(npz_file: Path, 
                       text_prompt: str, 
                       out_folder: Path):
    '''
    From the npz file created for MedSAM2_RECIST, create new npz files that hold the relevant data 
    for BiomedParse and save them into a separate location. Assumes all preprocessing wanted that
    is not handled by BiomedParse is already done (e.g. windowing) 

    Parameters 
    ----------
    npz_file: Path 
        Contains the input npz information used for MedSAM2_RECIST (keys: imgs, gts, recist, 
        spacing, direction, origin)
    text_prompt: str 
        The text prompt to use for the particular disease site (e.g. Presence of kidney lesion detected in 
        abdominal CT imaging)
    out_folder: Path 
        Where the new npz files for input to BiomedParse will be saved to

    Returns 
    ----------
    image_arr: np.ndarray()
    gts_arr: np.ndarray() 
    text_arr: np.ndarray() 
    spacing: np.ndarray() 
    ''' 
    # Load in npz data 
    medsam_re_npz = np.load(npz_file) 
    image_arr = medsam_re_npz['imgs']
    gts_arr = medsam_re_npz['gts']
    spacing_arr = medsam_re_npz['spacing']
    dir_arr = medsam_re_npz['direction'] 
    origin_arr = medsam_re_npz['origin']
    text_arr = np.array({'1': text_prompt, 'instance_label': 0})

    # Get filename from medsam npz path 
    out_filename = str(npz_file).split('/')[-1]
    save_path = out_folder / out_filename

    # Make sure out folder exists before saving 
    if not out_folder.exists(): 
        out_folder.mkdir(parents = True, exist_ok = True)

    # Create new npz with the potentially relevant keys 
    np.savez_compressed(save_path, 
                        imgs = image_arr, 
                        gts = gts_arr, 
                        text_prompts = text_arr, 
                        spacing = spacing_arr, 
                        direction = dir_arr, 
                        origin = origin_arr)
    
    # Return info only important to the future prediction 
    return image_arr, gts_arr, text_arr, spacing_arr

def locate_centre_slice(mask_3d):
    """
    Locates the center slice of a 3D mask.

    Args:
    - mask_3d (ndarray): A 3D binary mask.

    Returns:
    - (int): The index of the center slice.
    """
    
    # find the slice in the center
    lesion_labels = label(mask_3d)
    centre_slc = regionprops(lesion_labels)[0].centroid[0]

    # number of voxels per slice
    vox_per_slc = np.array([np.sum(mask_3d[slc,:,:]) for slc in range(mask_3d.shape[0])])
    max_vox_slc = np.where(vox_per_slc==np.max(vox_per_slc))[0]
    if len(max_vox_slc) > 1: 
        max_vox_slc = max_vox_slc[int(np.floor(len(max_vox_slc)/2))]

    return int(np.floor((centre_slc+max_vox_slc)/2))

def mid_slice_visual(image, 
                     mask_preds, 
                     gt_masks, 
                     text_prompts: dict, 
                     mid_slice: int, 
                     full_savepath: Path): 
    '''
    Adjusted visualization from the inference_example_3D.ipynb example notebook that is in the BiomedParse repo. Saves a figure 
    showing the middle slice of the original image, the ground truth mask overlayed, and the predicted mask overlayed along 
    with the text prompt used to create the mask as the legend. 

    Parameters
    ----------
    image: 
        The array containing the original image data (same shape as mask_preds and gt_masks)
    mask_preds: 
        The array containing the predicted mask values (same shape as the image and gt_masks) 
    gt_masks: 
        The array containing the ground truth mask values (same shape as the image and mask_preds) 
    text_prompts: dict 
        Contains all of the prompts used to create the predicted masks (for now only one text prompt in dict) 
    mid_slice: int 
        The middle slice of the segmentation 
    full_savepath: Path
        Should contain where to save the path and what to call the file outputted
    '''
    slice_id = mid_slice
    slice_image = image[slice_id]
    slice_mask = mask_preds[slice_id]
    slice_gt   = gt_masks[slice_id]

    # 1) Compute the mapping
    unique_ids = np.unique(np.concatenate((slice_mask, slice_gt)))
    id_map = {orig_id: new_i for new_i, orig_id in enumerate(unique_ids)}

    slice_mask_mapped = np.vectorize(id_map.get)(slice_mask)
    slice_gt_mapped   = np.vectorize(id_map.get)(slice_gt)

    slice_mask_mapped = np.ma.masked_where(slice_mask_mapped == 0, slice_mask_mapped)
    slice_gt_mapped = np.ma.masked_where(slice_gt_mapped == 0, slice_gt_mapped)

    # 2) Which IDs to show (drop background=0)
    mask_ids = unique_ids[1:]

    # 3) Labels for legend
    legends = [text_prompts[str(i)] for i in mask_ids if str(i) in text_prompts]
    mask_ids = [i for i in mask_ids if str(i) in text_prompts]

    # 4) Build a *discrete* colormap of size len(unique_ids)
    #    so that cmap(k) gives exactly the k-th color.
    cmap = plt.get_cmap('tab20', len(unique_ids)-1)

    # 5) Create handles using integer lookup into the discrete cmap
    handles = [
        mpatches.Patch(color=cmap(id_map[i]-1), label=txt)
        for i, txt in zip(mask_ids, legends)
    ]

    # 6) Plot
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    axes[0].imshow(slice_image, cmap="gray")
    axes[0].set_title("Original Image Slice")
    axes[0].axis("off")

    axes[1].imshow(slice_image, cmap='gray')
    axes[1].imshow(slice_gt_mapped, cmap=cmap, interpolation='nearest', alpha = 0.6)
    axes[1].set_title("Ground Truth Masks")
    axes[1].axis("off")

    axes[2].imshow(slice_image, cmap='gray')
    axes[2].imshow(slice_mask_mapped, cmap=cmap, interpolation='nearest', alpha = 0.6)
    axes[2].set_title("Predicted Masks")
    axes[2].axis("off")

    # 7) Shared legend below, single column
    fig.legend(handles, legends, loc='lower center', bbox_to_anchor=(0.29, -0.15), ncol=1, frameon=False, fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0)
    
    fig.savefig(full_savepath, bbox_inches = 'tight')

def find_first_last_slice(mask): 
    '''
    Based on a 3D mask array, get the first and last slice within the array that has masked values. 

    Parameters
    ----------
    mask: 
        3D mask array 
    
    Returns 
    ----------
    first_slice: int 
        The index where the first slice of the mask is 
    last_slice: int 
        The index where the last slice of the mask is 
    '''
    axes = tuple([i for i in range(mask.ndim) if i != 0])

    slices = mask.any(axis = axes) 

    nonzero_indices = np.where(slices)[0]

    first_slice = np.amin(nonzero_indices)
    last_slice = np.amax(nonzero_indices) 

    return first_slice, last_slice

def pos_neg_true_visual(image, 
                        mask_preds, 
                        gt_masks, 
                        full_savepath: Path): 
    '''
    Visualization of the selected slices based on the ground truth, showing the true positive, false positive, and false
    negative areas within these slices.

    Parameters
    ----------
    image: 
        The array containing the original image data (same shape as mask_preds and gt_masks)
    mask_preds: 
        The array containing the predicted mask values (same shape as the image and gt_masks) 
    gt_masks: 
        The array containing the ground truth mask values (same shape as the image and mask_preds) 
    full_savepath: Path
        Should contain where to save the path and what to call the file outputted
    '''
    # Make the predicted mask a different number to represent a different colour 
    mask_alt = mask_preds * 2 

    # Add masks together so that false negative is 1, false positive is 2, and true positive is 3 
    comb_masks = mask_alt + gt_masks
    comb_masks = np.ma.masked_where(comb_masks == 0, comb_masks)

    # Find the first and last slices that have mask in them 
    gt_min, gt_max = find_first_last_slice(gt_masks)

    num_nonzero_slices = gt_max - gt_min + 1 # need to add one to get true number. e.g. slices 0 - 5 have non zero (inclusive), true answer is 6 slices, but subtraction only will yield 5
    # Check to see if there are more than 5 slices within the ground truth mask and adjust the subplot information accordingly 
    if num_nonzero_slices < 5: 
        subplot_slices = num_nonzero_slices
        slices_to_plot = range(gt_min, gt_max + 1)
    else: 
        subplot_slices = 5
        slices_to_plot = [gt_min, gt_min + math.floor(num_nonzero_slices/4), gt_min + math.floor(num_nonzero_slices/2), gt_min + math.floor(num_nonzero_slices * 3 / 4), gt_max]
    
    print(slices_to_plot)
    fig, axes = plt.subplots(1, subplot_slices, figsize = (15, 3)) 

    # Create colour map for mask 
    colours = ['red', 'green', 'blue']
    boundaries = [1, 2, 3, 4]
    cmap = mcolors.ListedColormap(colours)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    # Make legend info 
    legend_elem = [mpatches.Patch(color = 'red', label = 'False Negative'), 
                mpatches.Patch(color = 'green', label = 'False Positive'), 
                mpatches.Patch(color = 'blue', label = 'True Positive')]
    counter = 0
    for i in slices_to_plot: 
        axes[counter].imshow(image[i], cmap = 'gray') 
        axes[counter].imshow(comb_masks[i], cmap = cmap, norm = norm, interpolation = 'nearest', alpha = 0.6)
        axes[counter].axis("off")
        counter += 1

        plt.tight_layout()

    fig.legend(handles = legend_elem, loc = 'lower right', bbox_to_anchor=(0.67, -0.15), ncol=3, frameon=False, fontsize=11)
    fig.savefig(full_savepath, bbox_inches = 'tight')

def calc_metrics(pred_mask: np.ndarray, 
                 gt_mask: np.ndarray, 
                 spacing: np.ndarray, 
                 filename: str): 
    '''
    Calculate performance metrics based on the predicted and ground truth masks and save into a dataframe. 

    Parameters
    ----------
    pred_mask: np.ndarray
        The mask that was predicted by the model. 
    gt_mask: np.ndarray
        The ground truth segmentation array. 
    spacing: np.ndarray
        The spacing associated with the ground truth mask. 
    filename: str 
        The name of the npz file that is being evaluated.
    
    Returns
    ----------
    metric_df: pd.DataFrame
        Contains the evaluation performance. 
    '''
    #Initialize evaluator 
    metric_eval = Evaluator() 
    metric_dict = metric_eval(preds = pred_mask, 
                              targets = gt_mask, 
                              spacing = spacing 
                              )
    
    metric_df = pd.DataFrame(metric_dict, index = [0])

    #Add columns for the range of segmentation values (both ground truth and predicted)
    first_gts, last_gts = find_first_last_slice(gt_mask)
    first_pred, last_pred = find_first_last_slice(pred_mask) 

    gts_range = [first_gts, last_gts] 
    pred_range = [first_pred, last_pred] 

    metric_df['GTSliceRange'] = [gts_range]
    metric_df['PredSliceRange'] = [pred_range]
    metric_df['filename'] = filename # To ensure we can map the results back to the segmentations 

    return metric_df

def create_from_prev_and_predict(npz_file: Path, 
                                 text_prompt: str, 
                                 device: torch.device,
                                 model,
                                 npz_folder: Path, 
                                 results_folder: Path, 
                                 visualizations: bool = True):
    '''
    From the existing files created for the MedSAM2-RECIST testing, create new npz files containing all relevant information and 
    output the results. Optional visualizations are available. 

    Parameters 
    ----------
    npz_file: Path 
        The path to a single npz file from the MedSAM2-RECIST iteration 
    text_prompt: str
        The text prompt to be used for the segmentation prediction
    device: torch.device 
        Device used in model initialization to be used for predictions
    model: 
        The BiomedParse model created after initialization with the checkpoint
    npz_folder: Path
        Where to save the created npz 
    results_folder: Path 
        Where to save the predicted mask information and visualizations (if applicable)
    visualizations: bool 
        Whether or not to save any visualizations about the prediction. Automatically set to true.
    '''
    # Create the current npz files form the existing MedSAM2-RECIST npz files and get necessary information for segmentation out
    image, gt_masks, prompt, spacing = npzs_from_existing(npz_file = npz_file, 
                                          text_prompt = text_prompt, 
                                          out_folder = npz_folder)
    
    filename = str(npz_file).split("/")[-1]
    # Get text prompt info into a dictionary again 
    text_prompts = prompt.item()

    # Perform preprocessing 
    ids = [int(_) for _ in text_prompts.keys() if _ != "instance_label"]
    ids.sort()
    text = "[SEP]".join([text_prompts[str(i)] for i in ids])

    imgs, pad_width, padded_size, valid_axis = process_input(image, 512)

    imgs = imgs.to(device).int()

    input_tensor = {
        "image": imgs.unsqueeze(0),  # Add batch dimension
        "text": [text],
    }

    # Predict segmentation 
    with torch.no_grad():
        output = model(input_tensor, mode="eval", slice_batch_size=4)

        mask_preds = output["predictions"]["pred_gmasks"]
        mask_preds = F.interpolate(mask_preds, size=(512, 512), mode="bicubic", align_corners=False, antialias=True)

        mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"])
        mask_preds = merge_multiclass_masks(mask_preds, ids)
        mask_preds = process_output(mask_preds, pad_width, padded_size, valid_axis) # Make sure the predicted mask aligns with the inputs and gts

        save_filename = str(npz_file).split("/")[-1]
        print(f"Mask prediction for {save_filename} is complete.")
        print("Processed mask shape:", mask_preds.shape)
    
    # Save relevant information out into npz files 
    save_path = results_folder / save_filename 

    # Make sure results folder exists before saving
    if not results_folder.exists(): 
        results_folder.mkdir(parents = True, exist_ok = True) 

    np.savez_compressed(save_path, 
                        imgs = image, 
                        gts = gt_masks, 
                        preds = mask_preds 
                        )
    # Calculate metrics 
    metrics_df = calc_metrics(pred_mask = mask_preds, 
                              gt_mask = gt_masks, 
                              spacing = spacing, 
                              filename = filename)
    
    if visualizations: 
        # Create folder to put the visualizations 
        visual_folder = results_folder / Path('visualization') 

        if not visual_folder.exists(): 
            visual_folder.mkdir(parents = True, exist_ok = True) 
        
        # Save a visualization of the middle slice of the ground truth segmentation and the predicted segmentation at that slice
        mid_slice = locate_centre_slice(gt_masks) #FUTURE NOTE: If we decide to do multiple segmentations in one array, this will need to change

        midview_folder = visual_folder / 'mid_seg_slice_view'

        if not midview_folder.exists(): 
            midview_folder.mkdir(parents = True, exist_ok = True) 
        
        midview_save = midview_folder / Path(save_filename.split('.')[0] + '_midseg.png') 

        mid_slice_visual(image = image, 
                         mask_preds = mask_preds, 
                         gt_masks = gt_masks, 
                         text_prompts = text_prompts, 
                         mid_slice = mid_slice, 
                         full_savepath = midview_save)

        # Save a visualization of the 5 most central slices and the true positive, false positive, and false negative regions
        pos_neg_folder = visual_folder / 'pos_neg_true_view'

        if not pos_neg_folder.exists(): 
            pos_neg_folder.mkdir(parents = True, exist_ok = True) 

        pos_neg_save = pos_neg_folder / Path(save_filename.split('.')[0] + '_posneg.png') 

        pos_neg_true_visual(image = image, 
                            mask_preds = mask_preds, 
                            gt_masks = gt_masks, 
                            full_savepath = pos_neg_save)
        
    return metrics_df

@click.command()
@click.option('--checkpoint')
@click.option('--prev_npz_input_folder')
@click.option('--new_npz_input_folder') 
@click.option('--results_folder') 
@click.option('--text_prompt') 
@click.option('--n_jobs')
def run_create_pred_eval(checkpoint: Path, 
                         prev_npz_input_folder: Path, 
                         new_npz_input_folder: Path, 
                         results_folder: Path, 
                         text_prompt: str,
                         n_jobs: int): 
    model, device = initialize_model(ckpt_path = checkpoint)
    print(text_prompt)
    metrics_results = Parallel(n_jobs = n_jobs)(delayed(create_from_prev_and_predict)(npz_file = file, 
                                                                                      text_prompt = text_prompt, 
                                                                                      device = device,
                                                                                      model = model,
                                                                                      npz_folder = Path(new_npz_input_folder), 
                                                                                      results_folder = Path(results_folder) 
                                                                                      ) for file in tqdm(Path(prev_npz_input_folder).iterdir(), total = len(list(Path(prev_npz_input_folder).iterdir()))) if str(file).endswith('.npz'))
    
    for result in metrics_results: 
        if 'all_metrics_df' not in locals(): 
            all_metrics_df = result 
        else: 
            all_metrics_df = pd.concat([all_metrics_df, result], ignore_index = True).reset_index(drop = True)
    
    all_metrics_df.to_csv(results_folder / 'metric_eval.csv', index = False)

if __name__ == "__main__": 
    run_create_pred_eval()

        


