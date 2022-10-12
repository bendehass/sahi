from sahi.slicing import slice_coco
from sahi.utils.file import load_json

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt



coco_dict, coco_path = slice_coco(
    coco_annotation_file_path="/media/ben/training/Downloads/1573_FC_CVAT_import/annotations/instances_FC.json",
    image_dir="/media/ben/training/Downloads/1573_FC_CVAT_import/images/",
    output_coco_annotation_file_name="sliced_coco.json",
    ignore_negative_samples=False,
    output_dir="/media/ben/training/Downloads/1573_FC_CVAT_import/sliced/",
    slice_height=10000,
    slice_width=5000,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.1,
    verbose=True
)

f, axarr = plt.subplots(4, 5, figsize=(13,13))
img_ind = 0
for ind1 in range(4):
    for ind2 in range(5):
        img = Image.open("/media/ben/training/Downloads/1573_FC_CVAT_import/sliced/" + coco_dict["images"][img_ind]["file_name"])
        axarr[ind1, ind2].imshow(img)
        img_ind += 1