from sahi.slicing import slice_coco
from sahi.utils.file import load_json

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import sys
data_dir = sys.argv[1]


coco_dict, coco_path = slice_coco(
    coco_annotation_file_path=f"{data_dir}/annotations/instances_FC.json",
    image_dir=f"{data_dir}/images/",
    output_coco_annotation_file_name="sliced_coco.json",
    ignore_negative_samples=False,
    output_dir=f"{data_dir}/sliced/",
    slice_height=10000,
    slice_width=5000,
    overlap_height_ratio=0.04,
    overlap_width_ratio=0.04,
    min_area_ratio=0.1,
    verbose=True
)

f, axarr = plt.subplots(4, 5, figsize=(13,13))
img_ind = 0
for ind1 in range(4):
    for ind2 in range(5):
        img = Image.open(f"{data_dir}/sliced/" + coco_dict["images"][img_ind]["file_name"])
        axarr[ind1, ind2].imshow(img)
        img_ind += 1