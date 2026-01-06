import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import hydra
import seaborn as sns 
import yaml
import SimpleITK as sitk
import click

from skimage.measure import regionprops 
from skimage.draw import line
from pathlib import Path
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from joblib import Parallel, delayed
from inference import postprocess, merge_multiclass_masks
from utils import process_input, process_output
from tqdm import tqdm

from run_biomedparse import initialize_model, locate_centre_slice, mid_slice_visual, pos_neg_true_visual, calc_metrics

### Helper Functions ###

## Initialization and Loading ## 
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
    device: torch.device 
        What the model will be using (cuda or cpu)
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

def load_prompt_skeletons(prompt_config_path: Path): 
    '''  
    From a .yaml configuration file, load in the prompt skeletons
    as a dictionary. 

    Parameters
    ----------
    prompt_config_path: Path
        Path to the .yaml configuration file
    
    Returns 
    ----------
    prompt_skel_dict: dict 
        A nested dictionary containing all of the different prompts
        listed in the configuration file. Should be in the form 
        TEXT_PROMPTS
        ├── PROMPT1
        │   └── PROMPT1_INFO
        ├── PROMPT2
        │   └── PROMPT2_INFO
        ├── ...
        └── PROMPTn
            └── PROMPTn_info
        where the prompt info is stored in a dictionary that is 
        BiomedParse-compatible. 
    '''
    with open(prompt_config_path, 'r') as file:
        prompt_skel_dict = yaml.load(file, Loader=yaml.SafeLoader)

    return prompt_skel_dict

## Imaging Preprocessing ## 
def apply_windowing(img_array: np.ndarray,
                    window_level: int, 
                    window_width: int
                    ) -> np.ndarray:
    '''
    Window an image based on a window width (width of range of values to use) and a window level (where to center a window level). Otherwise known as clipping or clamping in image processing.
    
    Parameters
    ----------
    img_array: np.ndarray, 
        The image to be windowed 
    window_level: int
        Where to center the range defined in window_width
    window_width: int 
        How wide the of a range to include, centered on the level.  

    Returns 
    ----------
    windowed_img: np.ndarray
        The processed image with values clamped at the upper and lower value
    '''
    #Calculate upper and lower clamp values
    upper_val = window_level + window_width / 2 
    lower_val = window_level - window_width / 2 

    #Window image
    windowed_img = np.clip(img_array, lower_val, upper_val)

    return windowed_img 

## Generating Coordinates ## 
def pad_bbox(box:np.array,
             mask:np.array, 
             padding:int,
             spacing:np.array = None
             ) -> np.array:
    # Get full image dimensions to keep padding within image size
    # D, H, W (z, y, x)
    mask_shape = mask.shape

    if spacing is not None: # Use the actual image spacing to calculate the padding
        # Check that spacing can be applied to this mask's bounding box dimensions
        if len(spacing) < len(mask_shape):
            spacing = spacing[0, len(mask_shape)]
        elif len(spacing) > len(mask_shape):
            message = "Spacing for padding has more dimensions than the mask image."
            raise ValueError(message)

        # calculate the number of voxels to pad based on the actual image spacing
        padding = np.round(padding / spacing)
    else:
        # Convert padding into an array with the same length as image dimensions
        # This matches the behaviour of the spacing option
        padding = padding * np.ones(len(mask_shape))
    
    pad_x_min = max(0, box[0] - padding[0])
    pad_y_min = max(0, box[1] - padding[1])
    # Handling 2D bounding box
    if len(box) == 4:
        mask_H, mask_W = mask_shape[0], mask_shape[1]
        pad_x_max = min(mask_W, box[2] + padding[0])
        pad_y_max = min(mask_H, box[3] + padding[1])

        padded_box = np.array([pad_x_min, pad_y_min,
                               pad_x_max, pad_y_max])

    # Handling 3D bounding box
    if len(box) == 6:
        mask_D, mask_H, mask_W = mask_shape[0], mask_shape[1], mask_shape[2]
        pad_z_min = max(0, box[2] - padding[2])
        pad_x_max = min(mask_W, box[3] + padding[0])
        pad_y_max = min(mask_H, box[4] + padding[1]) 
        pad_z_max = min(mask_D, box[5] + padding[2])
        
        padded_box = np.array([pad_x_min, pad_y_min, pad_z_min,
                               pad_x_max, pad_y_max, pad_z_max])

    return padded_box.astype(int)

def mask2D_to_bbox(gt2D:np.array, 
                   mask_path:Path, 
                   padding:int | None = None,
                   spacing:np.array = None
                   ) -> np.array:
    try:
        ## Old code 
        # y_indices, x_indices = np.where(gt2D > 0)
        # x_min, x_max = np.min(x_indices), np.max(x_indices)
        # y_min, y_max = np.min(y_indices), np.max(y_indices)
        # boxes = np.array([x_min, y_min, x_max, y_max])

        props = regionprops(gt2D)[0]
        y_cent, x_cent = props.centroid
        orientation = props.orientation
        semi_maj_axis_len = props.major_axis_length / 2

        x_start = x_cent - np.sin(orientation) * semi_maj_axis_len
        y_start = y_cent - np.cos(orientation) * semi_maj_axis_len

        x_end = x_cent + np.sin(orientation) * semi_maj_axis_len
        y_end = y_cent + np.cos(orientation) * semi_maj_axis_len

        boxes = np.array([x_start, y_start, x_end, y_end])

        if padding:
            boxes = pad_bbox(box = boxes,
                             mask = gt2D,
                             padding = padding,
                             spacing = spacing)
        
        return boxes.astype(int)
    
    except Exception as e:
        raise Exception(f'error {e} with file {mask_path} and sum of gts is {gt2D.sum()}')

def mask3D_to_bbox(gt3D:np.array, 
                   mask_path:Path, 
                   padding:int | None = None,
                   spacing:np.array = None
                   ) -> np.array:
    try:
         # Find slices in the mask that have label voxels to get z boundaries
        z_indices, _, _ = np.where(gt3D > 0)
        z_min, z_max = np.min(z_indices), np.max(z_indices)
    except Exception as e:
        raise Exception(f'error {e} with file {mask_path} and sum of gts is {gt3D.sum()}')
   
    # Find the centre slice of the mask
    z_mid = np.median(z_indices).astype(int)
    # Select out this slice from the array
    gt_mid = gt3D[z_mid]

    # Get the x and y mask boundaries using the 2D function
    box_2d = mask2D_to_bbox(gt_mid, mask_path)
    x_min, y_min, x_max, y_max = box_2d
    boxes3D = np.array([x_min, y_min, z_min, x_max, y_max, z_max])

    if padding:
        # Apply padding to bounding box if requested
        boxes3D = pad_bbox(box = boxes3D,
                           mask = gt3D,
                           padding = padding,
                           spacing = spacing)
    
    return boxes3D.astype(int)

def get_line_from_recist(recist_coords: np.array, 
                         slice_number: int, 
                         img_size: np.array):
    '''
    From the RECIST measurement coordinates, generate a line connecting both coordinates on the correct slice and return an np.ndarray the same shape as the image.
    Output to be compatible with the ['recist'] array of the .npz files needed for MedSAM2-RECIST.

    Parameters
    ----------
    recist_coords: array
        A list of coordinates in [x1, y1, x2, y2] format that defines the RECIST measurement 
    slice_number: int
        The slice that the measurement was taken on
    img_size: np.array
        The x, y, and z size of the image in [z_space, x_space, y_space] format
    
    Returns
    ----------
    recist_arr: np.ndarray
        A binary array of the same shape as the image with the pixels of the line = 1
    '''
    #Generate an array in the same size as the image filled with all zeros 
    recist_arr = np.zeros((img_size[0], img_size[1], img_size[2]), dtype = int)
    
    #Round the coordinate values to their nearest integers 
    coords_round = np.rint(recist_coords).astype(int)

    #Draw line using coordinates 
    rr, cc = line(coords_round[0], coords_round[1], coords_round[2], coords_round[3])

    #Put line into the correct slice in the RECIST array of all zeros 
    recist_arr[slice_number][cc, rr] = 1

    return recist_arr

def find_first_last_slice(mask: np.ndarray): 
    '''
    Based on a 3D mask array, get the first and last slice within the array that has masked values. 
    Assumes (z, x, y) coordinate order.

    Parameters
    ----------
    mask: np.ndarray
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

def get_recist_line_coords(recist_arr: np.ndarray): 
    '''
    From a 3D array containing the RECIST line, compute the coordinates of the line's endpoints
    and the line's segmentation. 

    Parameters
    ----------
    recist_arr: np.ndarray
        The 3D array containing the RECIST line. Assumes (z, x, y) coordinate order. 

    Returns
    ----------
    first_coord: np.array
        The location of the first endpoint of the RECIST line. In [x1, y1] format
    last_coord: np.array
        The location of the last endpoint of the RECIST line. In [x2, y2] format
    slice_num: int 
        The slice number where the RECIST line is located.          
    '''
    slice_num, _ = find_first_last_slice(recist_arr) #Should only have one slice with non-zero elements --> only need first slice
    
    non_zero_idxs = np.argwhere(recist_arr[slice_num])

    if non_zero_idxs.size < 1: 
        print("No annotation found, please check previous preprocessing steps.")
    else:
        first_coord = np.array(non_zero_idxs[0])
        last_coord = np.array(non_zero_idxs[-1])

        print(f"First Point: {first_coord}")
        print(f"Last Point: {last_coord}")
        print(f"Slice Number: {slice_num}")

        return first_coord, last_coord, slice_num

def get_recist_midpoint(coord1: np.array,
                        coord2: np.array): 
    '''  
    Get the midpoint coordinates from a RECIST line. 

    Parameters
    ----------
    coord1: np.array
        One of the endpoints of the RECIST line in [x1, y1] format
    coord2: np.array 
        The other endpoint of the RECIST line in [x2, y2] format

    Returns 
    ---------
    midpoint: np.array
        The midpoint coordinates in [x, y] format
    '''
    midpoint = [int((coord1[0] + coord2[0]) / 2), int((coord1[1] + coord2[1]) / 2)]

    return midpoint

def format_coord_to_str(recist_coord1: np.array, 
                        recist_coord2: np.array, 
                        slice_num: int): 
    '''  
    From the two endpoint coordinates and slice number found, format info into string format 
    following the text prompt form examples in the .yaml file. 

    Parameters
    ----------
    recist_coord1: np.array
        The first RECIST line endpoint's coordinates in [x1, y1] format 
    recist_coord2: np.array
        The second RECIST line endpoint's coordinates in [x2, y2] format 
    slice_num: int 
        The slice number describing where the RECIST line was drawn in the image. 

    Returns 
    ----------
    coord_point: str 
        A string of the form "x = <X>, y = <Y>, z = <Z>", where the <> are replaced with
        the appropriate coordinate information
    line1_coords: str
        A string of the form "x1 = <X1>, y1 = <Y1>, z1 = <Z1>, x2 = <X2>, y2 = <Y2>, z2 = <Z2>", 
        where the <> are replaced with the appropriate coordinate information
    line2_coords: str
        An alternative form to describe a line "(<Z1>, <X1>, <Y1>), (<Z2>, <X2>, <Y2>)".
        Format follows the two strings above. 
    '''
    # Get midpoint coordinates to be used for coord_point 
    midpoint = get_recist_midpoint(coord1 = recist_coord1, 
                                   coord2 = recist_coord2)
    
    # Create prompt strings 
    coord_point = f"x = {midpoint[0]}, y = {midpoint[1]}, z = {slice_num}"
    line1_coords = f"x1 = {recist_coord1[0]}, y1 = {recist_coord1[1]}, z1 = {slice_num}, x2 = {recist_coord2[0]}, y2 = {recist_coord2[1]}, z2 = {slice_num}"
    line2_coords = f"({slice_num}, {recist_coord1[0]}, {recist_coord1[1]}), ({slice_num}, {recist_coord2[0]}, {recist_coord2[1]})"

    return coord_point, line1_coords, line2_coords

def create_coord_prompts(prompt_skel_dict: dict, 
                         coord_point: str, 
                         line1_coords: str, 
                         line2_coords: str): 
    '''  
    Using the coordinate point strings, replace the dummy variables in the prompt 
    skeleton dictionary with the appropriate string and return the new dictionary.

    Parameters
    ----------
    prompt_skel_dict: dict
        A dictionary containing all of the skeleton text prompts. Originally loaded in 
        from .yaml file
    coord_point: str 
        A string of the form "x = <X>, y = <Y>, z = <Z>", where the <> are replaced with
        the appropriate coordinate information
    line1_coords: str
        A string of the form "x1 = <X1>, y1 = <Y1>, z1 = <Z1>, x2 = <X2>, y2 = <Y2>, z2 = <Z2>", 
        where the <> are replaced with the appropriate coordinate information
    line2_coords: str
        An alternative form to describe a line "(<Z1>, <X1>, <Y1>), (<Z2>, <X2>, <Y2>)".
        Format follows the two strings above.    

    Returns 
    ----------
    coord_prompt_dict: dict
        A dictionary with the same structure as prompt_skel_dict, but the dummy variables
        are replaced with the appropriate coordinate strings        
    '''
    prompts = prompt_skel_dict["TEXT_PROMPTS"]
    for key, _ in prompts.items(): 
        skel_prompt = prompts[key]['1']
        if 'COORD_POINT' in skel_prompt: 
            coord_prompt = skel_prompt.replace('COORD_POINT', coord_point)
        elif 'LINE1_COORDS' in skel_prompt: 
            coord_prompt = skel_prompt.replace('LINE1_COORDS', line1_coords)
        elif 'LINE2_COORDS' in skel_prompt: 
            coord_prompt = skel_prompt.replace('LINE2_COORDS', line2_coords) 
        
        prompts[key]['1'] = coord_prompt

    coord_prompt_dict = {"TEXT_PROMPTS": prompts}

    return coord_prompt_dict

## Metrics and Visualization ##
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

def get_hist_data(seg: np.ndarray): 
    '''  
    Get the nonzero pixel counts for each slice into dictionary form. Counts to be used for histogram plot.

    Parameters
    ----------
    seg: np.ndarray 
        A 3D array containing a mask (ground truth, predicted, etc.) 
    
    Returns 
    ----------
    pix_slice_dict: dict
        A dictionary with the slice number as the keys and the nonzero pixel count as the corresponding values
    '''
    pix_slice_dict = dict() 
    for slice_idx in range(seg.shape[0]): 
        pix_count = np.count_nonzero(seg[slice_idx]) 
        pix_slice_dict[slice_idx] = pix_count

    return pix_slice_dict

def get_hist_data_df(seg): 
    '''  
    Get the nonzero pixel counts for each slice into dataframe form. To be used for the density plot to be 
    compatible with seaborn. 

    Parameters
    ----------
    seg: np.ndarray
        A 3D array containing a mask (ground truth, predicted, etc.)

    Returns
    ----------
    pix_slice_df: pd.DataFrame
        A dataframe containing the slice number and the corresponding count in their respective columns
    '''
    pix_slice_dict = {
        'slice_num': [],
        'pix_count': []
    }
    for slice_idx in range(seg.shape[0]): 
        pix_count = np.count_nonzero(seg[slice_idx]) 
        pix_slice_dict['slice_num'].append(slice_idx) 
        pix_slice_dict['pix_count'].append(pix_count) 

    pix_slice_df = pd.DataFrame(pix_slice_dict)

    return pix_slice_df

def plot_hist(gt_mask: np.ndarray, 
              pred_mask: np.ndarray,
              text_prompt: str, 
              full_savepath: Path): 
    '''  
    Plot a histogram of the number of mask pixels in each of the slices. To give a quick
    view of where the model is segmenting vs. where the ground truth mask is. For
    a visual check to see how well the coordinates localize the segmentation to a 
    specific point. 

    Parameters
    ----------
    gt_mask: np.ndarray
        A 3D array containing the ground truth mask segmentation 
    pred_mask: np.ndarray 
        A 3D array containing the predicted mask segmentation 
    text_prompt: str
        The text prompt used to generate the predicted mask 
    full_savepath: Path
        A path containing both the location for saving and the 
        name of the file to be saved.
    '''
    # Get histogram data of number of pixels in each slice 
    pred_hist_data = get_hist_data(pred_mask)
    gt_hist_data = get_hist_data(gt_mask)

    # Get count data and slice data in a form that is compatible with histogram
    pred_val, pred_weight = zip(*[(key, val) for key, val in pred_hist_data.items()])
    gt_val, gt_weight = zip(*[(key, val) for key, val in gt_hist_data.items()])

    # Create figure 
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    ax[0].hist(pred_val, weights = pred_weight, bins = gt_mask.shape[0]-1) 
    ax[0].set_ylim(0, max(max(gt_hist_data.values()), max(pred_hist_data.values())))
    ax[0].set_title('Predicted Mask')
    ax[0].set_ylabel('Pixel Count')
    ax[0].set_xlabel('Slice Number')
    ax[1].hist(gt_val, weights = gt_weight, bins = gt_mask.shape[0]-1)
    ax[1].set_ylim(0, max(max(gt_hist_data.values()), max(pred_hist_data.values())))
    ax[1].set_title('Ground Truth Mask')
    ax[1].set_xlabel('Slice Number')
    plt.figtext(0.5, -0.05, "Prompt: " + text_prompt, ha='center', va='top')

    # Save figure 
    fig.savefig(full_savepath, bbox_inches = 'tight')

def plot_density(gt_mask: np.ndarray, 
              pred_mask: np.ndarray,
              text_prompt: str, 
              full_savepath: Path): 
    '''  
    Make a density plot to showcase where most of the segmented pixels
    are located. Similar to the histogram, but this shows the density
    information of the ground truth and the predicted mask overlayed 
    on the same plot.

    Parameters
    ----------
    gt_mask: np.ndarray
        A 3D array containing the ground truth mask segmentation 
    pred_mask: np.ndarray 
        A 3D array containing the predicted mask segmentation 
    text_prompt: str
        The text prompt used to generate the predicted mask 
    full_savepath: Path
        A path containing both the location for saving and the 
        name of the file to be saved.
    '''
    # Get data into a dataframe to be compatible with seaborn 
    gt_data = get_hist_data_df(gt_mask)
    pred_data = get_hist_data_df(pred_mask)

    # Add labels to each dataframe to identify which are ground truth 
    # and which are predicted
    gt_data["Mask Type"] = "Ground Truth"
    pred_data["Mask Type"] = "Predicted"

    # Combine dataframes into one for plotting
    all_data = pd.concat([gt_data, pred_data], axis = 0).reset_index(drop = True)

    # Create figure 
    plot = sns.displot(data = all_data, 
                x = "slice_num", 
                weights = "pix_count", 
                hue = "Mask Type",
                kind = "kde", 
                fill = True
                )
    plot.set(xlim=(0, gt_mask.shape[0]), xlabel = "Slice Number")

    plt.figtext(0.5, -0.05, "Prompt: " + text_prompt, ha='center', va='top')

    # Save figure
    plt.savefig(full_savepath, bbox_inches = 'tight')

## Functions for Run ## 
def biomedparse_preproc(image: np.ndarray, 
                        text_prompt: dict, 
                        device: torch.device): 
    '''  
    Performs the BiomedParse preprocessing found in their example notebook. Creates 
    input tensor to be used for prediction. 

    Parameters
    ----------
    img: np.ndarray 
        The image to be segmented. Perform any other preprocessing (e.g. windowing) 
        before this step. 
    text_prompt: dict
        The text prompt to be used for inference.
    device: torch.device
        The device used during model initialization to be used for inference. 
    
    Returns
    ----------
    input_tensor: dict 
        The input to be used for inference.
    pad_width: list
        A 2x3 list indicating how much to pad each side of the 3D imaging array 
    padded_size: int 
        The size of the cube that the image gets padded to 
    valid_axis: int 
        The axis to slice the image on
    ids: list 
        The keys of the text prompt dictionary (excluding the 'instance_label' key)
    '''
    # Perform preprocessing 
    ids = [int(_) for _ in text_prompt.keys() if _ != "instance_label"]
    ids.sort()
    text = "[SEP]".join([text_prompt[str(i)] for i in ids])

    imgs, pad_width, padded_size, valid_axis = process_input(image, 512)

    imgs = imgs.to(device).int()

    input_tensor = {
        "image": imgs.unsqueeze(0),  # Add batch dimension
        "text": [text],
    }

    return input_tensor, pad_width, padded_size, valid_axis, ids

def do_inference(in_tensor: dict, 
                 pad_width: list,
                 padded_size: int,
                 valid_axis: int, 
                 ids: list,
                 model): 
    ''' 
    For a given input dictionary, perform inference and save and return the 
    post-processed segmentation. 

    Parameters 
    ----------
    in_tensor: dict 
        The input tensor containing the 3D image volume and the text prompt 
        made from the BiomedParse preprocessing function. 
    pad_width: list
        A 2x3 list indicating how much to pad each side of the 3D imaging array 
    padded_size: int 
        The size of the cube that the image gets padded to 
    valid_axis: int 
        The axis to slice the image on
    ids: list 
        The keys of the text prompt dictionary (excluding the 'instance_label' key)
    model: 
        The BiomedParse model created after initialization with the checkpoint
        
    Returns
    ----------
    mask_preds: np.ndarray
        The predicted segmentation. 
    '''
    with torch.no_grad():
        output = model(in_tensor, mode="eval", slice_batch_size=4)

        mask_preds = output["predictions"]["pred_gmasks"]
        mask_preds = F.interpolate(mask_preds, size=(512, 512), mode="bicubic", align_corners=False, antialias=True)

        mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"])
        mask_preds = merge_multiclass_masks(mask_preds, ids)
        mask_preds = process_output(mask_preds, pad_width, padded_size, valid_axis) # Make sure the predicted mask aligns with the inputs and gts

    return mask_preds

def run_infer_result_plot(img: np.ndarray, 
                        seg: np.ndarray, 
                        spacing: np.ndarray,
                        text_prompt: dict, 
                        prompt_name: str,
                        model, 
                        device: torch.device,
                        pred_savepath: Path): 
    '''  
    For one image, segmentation and text prompt, have BiomedParse attempt to segment 
    the image. Assumes preprocessing has already been completed. 

    Parameters
    ----------
    img: np.ndarray
        The image to be segmented. Please preprocess image first before passing into this
        function. Ensure same shape as segmentation. 
    seg: np.ndarray
        The ground truth segmentation that pairs with a given image. 
    spacing: np.ndarray
        The spacing associated with the ground truth mask.
    text_prompt: dict 
        The text prompt used for inference. Assumes in a BiomedParse-compatible form. 
    prompt_name: str
        The name of the prompt used (for save name purposes).
    model: 
        The BiomedParse model created after initialization with the checkpoint
    device: torch.device
        Device used in model initialization to be used for predictions
    pred_savepath: Path 
        The path to save the predicted mask and the associated plot images to.
        Structure should have a folder with the RTSTRUCT name nested in a folder that 
        has the med-imagetools name for the patient and the save filename being the 
        same name as the NIFTI filename of the original mask.

    Returns
    ----------
    results_df: pd.DataFrame 
        Contains all segmentation performance results for the current run. 
    '''
    print(f"Running inference for {pred_savepath} with prompt {text_prompt}.")
    # Perform preprocessing 
    input_tensor, pad_width, padded_size, valid_axis, ids = biomedparse_preproc(image = img, 
                                                                                text_prompt = text_prompt, 
                                                                                device = device)
    
    # Perform inference 
    pred_mask = do_inference(in_tensor = input_tensor, 
                             pad_width = pad_width, 
                             padded_size = padded_size, 
                             valid_axis = valid_axis, 
                             ids = ids,
                             model = model)
    
    # Check if save path exists and if not, create it
    if not pred_savepath.exists(): 
        pred_savepath.mkdir(parents = True, exist_ok = True) 
    
    # Create predicted mask save name and save prediction 
    full_savepath = str(pred_savepath).removesuffix('.nii.gz')
    pred_mask_savepath = full_savepath + '_' + prompt_name + '_pred.nii.gz'

    sitk.WriteImage(image = sitk.GetImageFromArray(pred_mask), 
                    fileName = pred_mask_savepath)

    # Calculate metrics for this run 
    results_df = calc_metrics(pred_mask = pred_mask, 
                              gt_mask = seg,
                              spacing = spacing, 
                              filename = pred_mask_savepath)
    
    # Get plots and save to same folder 
    
    # Midslice plot #
    mid_slice = locate_centre_slice(mask_3d = seg) # Get center slice of ground truth
    mid_slice_savepath = Path(full_savepath + "_midslice.png") 
    
    mid_slice_visual(image = img, 
                     mask_preds = pred_mask, 
                     gt_masks = seg, 
                     text_prompts = text_prompt, 
                     mid_slice = mid_slice, 
                     full_savepath = mid_slice_savepath)

    # True Positive, False Positive, False Negative plot # 
    pos_neg_savepath = Path(full_savepath + "_posneg.png")

    pos_neg_true_visual(image = img, 
                        mask_preds = pred_mask, 
                        gt_masks = seg, 
                        full_savepath = pos_neg_savepath)
    
    # Pixel count histogram #
    pix_hist_savepath = Path(full_savepath + "_pixhist.png")
    
    plot_hist(gt_mask = seg, 
              pred_mask = pred_mask, 
              text_prompt = text_prompt['1'], 
              full_savepath = pix_hist_savepath)
    
    # Pixel-slice density plot #
    pix_dens_savepath = Path(full_savepath + "_pixdens.png")

    plot_density(gt_mask = seg, 
                 pred_mask = pred_mask, 
                 text_prompt = text_prompt['1'], 
                 full_savepath = pix_dens_savepath)

    return results_df

def choose_windowing(dataset: str): 
    '''  
    Determine which window level and width to use based on 
    the current dataset being used. 

    Parameters
    ----------
    dataset: str
        The full name of the dataset being used (e.g. TCIA_CPTAC-CCRCC)
    
    Returns 
    ----------
    window_level: int 
        The centering value of the window 
    window_width: int
        The width of the window
    '''
    match dataset: 
        case 'TCIA_CPTAC-CCRCC': 
            window_level = 50
            window_width = 400
        case 'TCIA_CPTAC-PDA':
            window_level = 50
            window_width = 400
        case 'TCIA_HEAD-NECK-RADIOMICS-HN1': 
            window_level = 50
            window_width = 400
        case 'TCIA_NSCLC-Radiogenomics': 
            window_level = -600
            window_width = 1500
        case 'TCIA_NSCLC-Radiomics': 
            window_level = -600
            window_width = 1500
        case _: 
            raise ValueError(f"Invalid dataset name: {dataset}. Please check spelling or add to this function with the correct window and level")

    return window_level, window_width

def run_one_prompt_test(img_path: Path, 
                        seg_path: Path, 
                        prompt_skels: dict, 
                        device: torch.device, 
                        model,
                        win_lvl: int, 
                        win_width: int,
                        n_jobs: int):
    '''  
    For a single image-segmentation pair, prepare the prompts with 
    the appropriate coordinate information, preprocess images, and 
    run the prompt testing in parallel.

    Parameters
    ----------
    img_path: Path
        The path to the current CT image data 
    seg_path: Path
        The path to the ground truth segmentation (either RTSTRUCT
        or SEG file)
    prompt_skels: dict
        The dictionary of prompt skeletons.
    model: 
        The BiomedParse model created after initialization with the checkpoint
    device: torch.device
        Device used in model initialization to be used for predictions
    window_lvl: int
        The centering value of the window for image preprocessing
    win_width: int
        The width of the window for image preprocessing
    n_jobs: int
        The number of jobs to run in parallel

    Returns
    ----------
    prompt_test_df: pd.DataFrame
        Contains the results of the prompt testing on one 
        sample.
    '''
    # Load in image and segmentation 
    ct_img_raw = sitk.ReadImage(img_path)
    ct_img_arr_raw = sitk.GetArrayFromImage(ct_img_raw)

    gt_seg = sitk.ReadImage(seg_path)
    gt_seg_arr = sitk.GetArrayFromImage(gt_seg)

    # Window image 
    ct_img_arr = apply_windowing(img_array = ct_img_arr_raw, 
                                 window_level = win_lvl, 
                                 window_width = win_width)
    
    # Prepare prompts 
    x_min, y_min, z_min, x_max, y_max, z_max = mask3D_to_bbox(gt3D = gt_seg_arr,
                                                              mask_path = seg_path)
    z_mid = (z_min + z_max) // 2 #Get mid slice of 3D bounding box

    coords = np.array([x_min, y_min, x_max, y_max])
    rerecist_arr = get_line_from_recist(recist_coords = coords, 
                                        slice_number = z_mid, 
                                        img_size = ct_img_arr.shape)

    coord1, coord2, mid_slice = get_recist_line_coords(recist_arr = rerecist_arr)

    mid_coord, line1_coord, line2_coord = format_coord_to_str(recist_coord1 = coord1, 
                                                              recist_coord2 = coord2, 
                                                              slice_num = mid_slice)
    prompt_dict = create_coord_prompts(prompt_skel_dict = prompt_skels, 
                                       coord_point = mid_coord, 
                                       line1_coords = line1_coord, 
                                       line2_coords = line2_coord)

    # Make save path 
    seg_filename = "/".join(str(seg_path).split("/")[-3:]) #Gets the patient ID, RTSTRUCT/SEG name, and mask file name
    dataset = str(seg_path).split("/")[-6]
    disease_site = str(seg_path).split("/")[-7]
    savepath = Path("data/results") / 'prompt_testing' / disease_site / dataset / Path(seg_filename)

    # Run prompt testing in parallel 
    samp_test_results = Parallel(n_jobs = n_jobs)(
        delayed(run_infer_result_plot)(img = ct_img_arr, 
                        seg = gt_seg_arr, 
                        spacing = gt_seg.GetSpacing(),
                        text_prompt = curr_prompt, 
                        prompt_name = prompt_name,
                        model =  model, 
                        device = device,
                        pred_savepath = savepath)
                        for prompt_name, curr_prompt in tqdm(
                            prompt_dict["TEXT_PROMPTS"].items(),
                            desc = "Running prompt testing using BiomedParse.",
                            total = len(prompt_dict["TEXT_PROMPTS"])
                        )
    )

    for result in samp_test_results: 
        if 'prompt_test_df' not in locals(): 
            prompt_test_df = result
        else: 
            prompt_test_df = pd.concat([prompt_test_df, result], ignore_index = True).reset_index(drop = True) 
    
    return prompt_test_df 

def run_prompt_test(dataset: str, 
                    disease_site: str, 
                    n_jobs: int, 
                    checkpoint_path: Path, 
                    skel_prompt_path: Path
                    ): 
    '''  
    Run the prompt testing across a specific dataset based on
    the data management plan structure. 

    Parameters
    ----------
    dataset: str
        The full name of the dataset to be used (e.g. TCIA_CPTAC-CCRCC). 
        Please ensure that the datasets used have a configured
        window and level to them in the choose_windowing function.
    disease_site: str
        Where the disease is located (e.g. Abdomen). Corresponds
        to the folder the dataset folder is in. 
    n_jobs: int
        How many jobs to run in parallel during the prompt testing.
    checkpoint_path: Path
        Where the checkpoint file is located for model initialization.
    skel_prompt_path: Path
        Where the .yaml config file is for the skeleton prompts.
    '''
    # Get current path for input data and output data
    dataset_short = dataset.split("_")[-1]
    curr_path = Path("data/procdata") / disease_site / dataset / 'images' / Path('mit_' + dataset_short)

    out_path = Path("data/results") / disease_site / dataset / 'prompt_testing'

    # Initialize model and load prompt skeletons
    model, device = initialize_model(ckpt_path = checkpoint_path)
    prompt_skels = load_prompt_skeletons(prompt_config_path = skel_prompt_path)

    # Choose windowing to use based on dataset 
    window_level, window_width = choose_windowing(dataset = dataset)

    for patient in curr_path.iterdir(): 
        for folder in patient.iterdir(): 
            if str(folder).str.contains('CT'): 
                for file in folder.iterdir(): # Assumes one file per folder, otherwise may not have matching image/segmentations
                    imaging_path = curr_path / patient / folder / file
            elif str(folder).str.contains('RTSTRUCT') or str(folder).str.contains('SEG'): 
                for file in folder.iterdir():
                    gt_seg_path = curr_path / patient / folder / file
            
            # Run the prompt test on current sample 
            result_df = run_one_prompt_test(img_path = imaging_path, 
                                            seg_path = gt_seg_path, 
                                            prompt_skels = prompt_skels, 
                                            model = model, 
                                            device = device, 
                                            win_lvl = window_level, 
                                            win_width = window_width, 
                                            n_jobs = n_jobs)
            if 'total_result_df' not in locals():
                total_result_df = result_df
            else:
                total_result_df = pd.concat([total_result_df, result_df], ignore_index = True).reset_index(drop = True) 
            
    total_result_df.to_csv(out_path / 'prompt_test_metrics.csv', index = False)

@click.command()
@click.option('--subset_yaml') 
@click.option('--prompt_skels')
@click.option('--checkpoint_path')
@click.option('--n_jobs')
def test_subset(subset_yaml: Path, 
                prompt_skels: Path,
                checkpoint_path: Path, 
                n_jobs: int): 
    '''  
    A predetermined test of a subset of patients from specific datasets. 

    Parameters
    ----------
    subset_yaml: Path 
        Contains the paths to the subset of patients to test. 
    prompt_skels: Path
        Contains the prompt skeletons to be used for testing. 
    checkpoint_path: Path
        Where the model checkpoint can be loaded from for initialization.
    n_jobs: int
        How many jobs to use for parallelization. 
    '''   
    out_path = Path('data/results') / 'biomedparse_prompt_testing'

    # Initialize model and load prompt skeletons
    model, device = initialize_model(ckpt_path = checkpoint_path)
    prompt_skels = load_prompt_skeletons(prompt_config_path = prompt_skels)

    # Load subset config file 
    with open(subset_yaml, 'r') as file:
        patient_subset = yaml.load(file, Loader=yaml.SafeLoader)

    for key, value in patient_subset["REL_PATHS"].items():
        window_level, window_width = choose_windowing(dataset = str(key)) 
        for path in value:
            # Iterate over files in current relative path and use key (dataset name) for windowing
            full_path = Path("data/procdata") / path
            for folder in full_path.iterdir(): 
                if str(folder).str.contains('CT'): 
                    # Assumes one file per folder and only one CT scan in the patient folder. 
                    for file in folder.iterdir():
                        imaging_path = full_path / folder / file 
                        break
            for folder in full_path.iterdir():
                if str(folder).str.contains('CT'): 
                    continue
                elif str(folder).str.contains('RTSTRUCT') or str(folder).str.contains('SEG'): 
                    for file in folder.iterdir(): # Assumes one file per folder 
                        gt_seg_path = full_path / folder / file
                
                # Run the prompt test on current sample 
                result_df = run_one_prompt_test(img_path = imaging_path, 
                                                seg_path = gt_seg_path, 
                                                prompt_skels = prompt_skels, 
                                                model = model, 
                                                device = device, 
                                                win_lvl = window_level, 
                                                win_width = window_width, 
                                                n_jobs = n_jobs)
                if 'total_result_df' not in locals():
                    total_result_df = result_df
                else:
                    total_result_df = pd.concat([total_result_df, result_df], ignore_index = True).reset_index(drop = True) 
            
    total_result_df.to_csv(out_path / 'prompt_test_metrics.csv', index = False)

if __name__ == "__main__": 
    test_subset()