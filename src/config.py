input_width = 640
input_height = 480
image_shape = (input_height, input_width)

anchor_ratios = [0.5, 1, 2]
anchor_scales = [16 * 2, 16 * 4, 16 * 8, 16 * 16]

grid_nums = [(input_height // 16, input_width // 16)]

target_pos_iou_thres = 0.7
target_neg_iou_thres = 0.3

rpn_n_sample = 128
rpn_pos_ratio = 0.5

n_pos = rpn_pos_ratio * rpn_n_sample


nms_iou_thresh = 0.7
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_test_pre_nms = 6000
n_test_post_nms = 300
min_size = 16

roi_n_sample = 128
roi_pos_ratio = 0.25
roi_pos_iou_thresh = 0.5
roi_neg_iou_thresh = 0.3

n_classes = 4  # include background


dataset_dir = "dataset"
training_img_dir = "dataset_blood/BCCD"
test_img_dir = "dataset_blood/BCCD"
