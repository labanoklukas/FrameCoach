from mmpose.apis import MMPoseInferencer
import os
import time

print("Starting video pose estimation...")

video_path = "images/Black_Leader_NTT_Bad3_LeaderFront_Shoulder.mp4"
python_path = "configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py"
pth_path = "checkpoints/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth"

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, "predictions"), exist_ok=True)
os.makedirs(os.path.join(results_dir, "visualisations"), exist_ok=True)

# Initialise inferencer
inferencer = MMPoseInferencer(
    pose2d=python_path,
    pose2d_weights=pth_path,
    device="cuda:0"  
)

print("Processing video...")
start_time = time.time()

# Process frames
frame_count = 0
for result in inferencer(
    video_path, 
    out_dir=results_dir,
    pred_out_dir="results/predictions",
    vis_out_dir="results/visualisations"
):
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")

total_time = time.time() - start_time
print(f"Video processed in {total_time:.2f} seconds")
print(f"Processed {frame_count} frames")
print(f"Results saved to results")
print(f"Predictions: results/predictions")
print(f"Visualisations: results/visualisations")