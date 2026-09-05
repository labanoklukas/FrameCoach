import os
import cv2
import numpy as np
import json
import math

print("Creating ballroom dance visualisation:")

# ------- FILE PATHS -------
input_video = "images/Coloured_RTLO_Good2.mp4"
keypoints_file = "results/predictions/Coloured_RTLO_Good2.json"
output_video = "results/posture_analysis.mp4"

# ------- CHECK FILES EXIST -------
if not os.path.exists(input_video):
    print(f"ERROR: Input video file not found: {input_video}")
    exit(1)

if not os.path.exists(keypoints_file):
    print(f"ERROR: Keypoints JSON file not found: {keypoints_file}")
    exit(1)

# ------- LOAD KEYPOINTS DATA -------
try:
    with open(keypoints_file, 'r') as f:
        json_data = json.load(f)
    
    print(f"Loaded JSON data with {len(json_data)} frames")
    
    KEYPOINT_NAMES = {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 
        4: "right_ear", 5: "left_shoulder", 6: "right_shoulder",
        7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 
        10: "right_wrist", 11: "left_hip", 12: "right_hip",
    }
    
except Exception as e:
    print(f"ERROR loading JSON: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# ------- SET UP VIDEO -------
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video dimensions: {width}x{height}, {fps} FPS, {total_frames} total frames")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# ------- DEFINE SKELETON CONNECTIONS -------
connections = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 3), (6, 4),  # Ears to shoulders
    (5, 6),  # Shoulder line
    (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12),  # Torso
    (11, 12)  # Hips
]

# ------- DEFINE COLOURS -------
fixed_colours = {
    (0, 1): (0, 255, 0), (0, 2): (0, 255, 0), (1, 3): (0, 255, 0), (2, 4): (0, 255, 0),  # Head
    (5, 3): (255, 0, 0), (6, 4): (255, 0, 0),  # Ears to shoulders
    (5, 6): (255, 0, 0),  # Shoulders
    (5, 11): (255, 0, 0), (6, 12): (255, 0, 0),  # Torso
    (11, 12): (255, 0, 0)  # Hips
}

keypoint_colours = {
    0: (0, 0, 255), 1: (0, 255, 0), 2: (0, 255, 0), 3: (0, 255, 0), 4: (0, 255, 0),
    5: (255, 0, 0), 6: (255, 0, 0), 7: (0, 0, 255), 8: (0, 0, 255),
    9: (0, 255, 255), 10: (0, 255, 255), 11: (255, 0, 255), 12: (255, 0, 255)
}

# ------- FUNCTION TO DETECT ORIENTATION -------
def detect_pose_orientation(keypoints, scores):
    if scores[0] < 0.5:
        return "back"
    
    if scores[5] < 0.5 or scores[6] < 0.5:
        return "unknown"
    
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    
    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    nose_offset = nose[0] - shoulder_center_x
    
    if nose_offset < -20:
        return "right-side"
    elif nose_offset > 20:
        return "left-side"
    else:
        return "forward"


# ------- ANALYSE SHOULDER ALIGNMENT -------
def analyse_shoulder_alignment(left_shoulder, right_shoulder):
    vertical_diff = abs(left_shoulder[1] - right_shoulder[1])
    
    if vertical_diff <= 5:
        return 1.0
    elif vertical_diff <= 15:
        return 0.9 - ((vertical_diff - 5) / 10) * 0.3
    elif vertical_diff <= 30:
        return 0.6 - ((vertical_diff - 15) / 15) * 0.3
    else:
        return max(0.1, 0.3 - ((vertical_diff - 30) / 30) * 0.2)

# ------- ANALYSE ELBOW ALIGNMENT WITH SHOULDER -------
def analyse_elbow_shoulder_alignment(shoulder, elbow):
    elbow_y_diff = abs(elbow[1] - shoulder[1])
    return max(0.0, 1.0 - (elbow_y_diff / 50.0))

# ------- ANALYSE LEFT ARM WRIST POSITION -------
def analyse_left_wrist_position(shoulder, elbow, wrist, reference_point=None):
    wrist_above_elbow = elbow[1] - wrist[1]
    
    if reference_point is None:
        if wrist_above_elbow <= 0:
            return 0.3
        
        wrist_to_shoulder = wrist[1] - shoulder[1]
        
        if wrist_to_shoulder >= 0:
            return max(0.3, 0.7 - (wrist_to_shoulder / 50.0))
        else:
            return 0.9
    else:
        if wrist_above_elbow <= 0:
            return 0.3
        
        wrist_to_ref = wrist[1] - reference_point[1]
        
        if wrist_to_ref < -50:
            return max(0.3, 0.8 - abs(wrist_to_ref + 50) / 100.0)
        elif wrist_to_ref > 50:
            return max(0.3, 0.8 - (wrist_to_ref - 50) / 100.0)
        else:
            if wrist_to_ref >= 0 and wrist_to_ref <= 30:
                return 1.0
            else:
                return 0.9

# ------- ANALYSE RIGHT ARM WRIST POSITION -------
def analyse_right_wrist_position(elbow, wrist):
    vertical_diff = wrist[1] - elbow[1]
    
    if vertical_diff < 0:
        return 0.3
    elif vertical_diff == 0:
        return 0.7
    elif vertical_diff > 0 and vertical_diff <= 60:
        return 1.0
    elif vertical_diff > 60 and vertical_diff <= 120:
        return max(0.5, 1.0 - (vertical_diff - 60) / 120.0)
    else:
        return max(0.3, 0.5 - (vertical_diff - 120) / 150.0)

# ------- ANALYSE FOLLOWER RIGHT ARM POSITION -------
def analyse_follower_right_arm(shoulder, elbow, wrist, reference_point=None):
    wrist_above_elbow = elbow[1] - wrist[1]
    
    if reference_point is None:
        if wrist_above_elbow <= 0:
            return 0.3
        
        wrist_to_shoulder = wrist[1] - shoulder[1]
        
        if wrist_to_shoulder >= 0:
            return max(0.3, 0.7 - (wrist_to_shoulder / 50.0))
        else:
            return 0.9
    else:
        if wrist_above_elbow <= 0:
            return 0.3
        
        wrist_to_ref = wrist[1] - reference_point[1]
        
        if wrist_to_ref < -50:
            return max(0.3, 0.8 - abs(wrist_to_ref + 50) / 100.0)
        elif wrist_to_ref > 50:
            return max(0.3, 0.8 - (wrist_to_ref - 50) / 100.0)
        else:
            if wrist_to_ref >= 0 and wrist_to_ref <= 30:
                return 1.0
            else:
                return 0.9

# ------- ANALYSE FOLLOWER LEFT ARM POSITION -------
def analyse_follower_left_arm(shoulder, elbow, wrist):
    elbow_y_diff = abs(elbow[1] - shoulder[1])
    elbow_score = max(0.0, 1.0 - (elbow_y_diff / 30.0))
    
    wrist_y_diff = wrist[1] - elbow[1]
    wrist_x_diff = abs(wrist[0] - elbow[0])
    
    wrist_score = 1.0
    if wrist_y_diff > 20:  
        wrist_score = max(0.3, 1.0 - (wrist_y_diff - 20) / 50.0)
    
    wrist_x_score = 1.0
    ideal_x_dist = 50 
    if abs(wrist_x_diff - ideal_x_dist) > 30:
        wrist_x_score = max(0.3, 1.0 - (abs(wrist_x_diff - ideal_x_dist) - 30) / 50.0)
    
    return (elbow_score * 0.4) + (wrist_score * 0.3) + (wrist_x_score * 0.3)

# ------- FUNCTION TO DETECT LEADER -------
def detect_if_leader(left_wrist, right_wrist, left_elbow, right_elbow, scores):
    required_points = [7, 8, 9, 10]
    if not all(scores[i] > 0.5 for i in required_points):
        return False
    
    left_arm_higher = (left_wrist[1] < right_wrist[1] - 30) or (left_elbow[1] < right_elbow[1] - 20)
    left_wrist_above_elbow = left_wrist[1] < left_elbow[1] - 20
    
    return left_arm_higher and left_wrist_above_elbow

# ------- FUNCTION TO DETECT FOLLOWER -------
def detect_if_follower(left_wrist, right_wrist, left_elbow, right_elbow, scores):
    required_points = [7, 8, 9, 10]
    if not all(scores[i] > 0.5 for i in required_points):
        return False
    
    right_arm_higher = (right_wrist[1] < left_wrist[1] - 30) or (right_elbow[1] < left_elbow[1] - 20)
    right_wrist_above_elbow = right_wrist[1] < right_elbow[1] - 20
    
    return right_arm_higher and right_wrist_above_elbow

# ------- COLOUR FUNCTIONS -------
def get_colour_from_score(score):
    if score > 0.7:
        g = 255
        r = int(255 * (1.0 - score))
        b = 0
    elif score > 0.3:
        g = int(255 * (score - 0.3) / 0.4)
        r = 255
        b = 0
    else:
        g = 0
        r = 255
        b = 0
    
    return (b, g, r)

# ------- PROCESS VIDEO FRAMES -------
frame_idx = 0
detected_role = None
role_text = None

while True:
    ret, frame = cap.read()
    
    if not ret:
        print(f"End of video reached after {frame_idx} frames")
        break
    
    frame_data = None
    for frame_info in json_data:
        if frame_info.get('frame_id') == frame_idx:
            frame_data = frame_info
            break
    
    # Default overall score
    overall_frame_score = 0.5
    
    if frame_data and 'instances' in frame_data:
        for instance in frame_data['instances']:
            keypoints = instance.get('keypoints', [])
            scores = instance.get('keypoint_scores', [])
            
            if len(keypoints) < 13 or len(scores) < 13:
                continue
            
            orientation = detect_pose_orientation(keypoints, scores)
                
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_elbow = keypoints[7]
            right_elbow = keypoints[8]
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            left_ear = keypoints[3]
            right_ear = keypoints[4]
            
            left_ear_visible = scores[3] > 0.5
            right_ear_visible = scores[4] > 0.5
            left_eye = keypoints[1] if scores[1] > 0.5 else None
            right_eye = keypoints[2] if scores[2] > 0.5 else None
            
            left_arm_good = all(scores[i] > 0.5 for i in [5, 7, 9])
            right_arm_good = all(scores[i] > 0.5 for i in [6, 8, 10])
            shoulders_good = scores[5] > 0.5 and scores[6] > 0.5
            left_torso_good = scores[5] > 0.5 and scores[11] > 0.5
            right_torso_good = scores[6] > 0.5 and scores[12] > 0.5
            hips_good = scores[11] > 0.5 and scores[12] > 0.5
            
            left_elbow_score = 0.5
            right_elbow_score = 0.5
            left_wrist_score = 0.5
            right_wrist_score = 0.5
            shoulders_aligned_score = 0.5
            follower_left_arm_score = 0.5
            follower_right_arm_score = 0.5
            
            reference_point = None
            if left_ear_visible:
                reference_point = left_ear
            elif left_eye is not None:
                reference_point = left_eye
                
            right_reference_point = None
            if right_ear_visible:
                right_reference_point = right_ear
            elif right_eye is not None:
                right_reference_point = right_eye
            
            if left_arm_good:
                left_elbow_score = analyse_elbow_shoulder_alignment(left_shoulder, left_elbow)
                left_wrist_score = analyse_left_wrist_position(left_shoulder, left_elbow, left_wrist, reference_point)
                follower_left_arm_score = analyse_follower_left_arm(left_shoulder, left_elbow, left_wrist)
            
            if right_arm_good:
                right_elbow_score = analyse_elbow_shoulder_alignment(right_shoulder, right_elbow)
                right_wrist_score = analyse_right_wrist_position(right_elbow, right_wrist)
                follower_right_arm_score = analyse_follower_right_arm(right_shoulder, right_elbow, right_wrist, right_reference_point)
            
            if shoulders_good:
                shoulders_aligned_score = analyse_shoulder_alignment(left_shoulder, right_shoulder)
            
            # Calculate overall frame score
            score_components = []
            if shoulders_good:
                score_components.append(shoulders_aligned_score)
            
            is_leader_detected = detect_if_leader(left_wrist, right_wrist, left_elbow, right_elbow, scores)
            is_follower_detected = detect_if_follower(left_wrist, right_wrist, left_elbow, right_elbow, scores)
            
            if is_leader_detected:
                if left_arm_good:
                    score_components.append(left_elbow_score)
                    score_components.append(left_wrist_score)
                if right_arm_good:
                    score_components.append(right_elbow_score)
                    score_components.append(right_wrist_score)
            elif is_follower_detected:
                if left_arm_good:
                    score_components.append(follower_left_arm_score)
                if right_arm_good:
                    score_components.append(follower_right_arm_score)
            else:
                # If role is unclear, use average of all scores
                if left_arm_good:
                    score_components.append(left_elbow_score)
                    score_components.append(left_wrist_score)
                if right_arm_good:
                    score_components.append(right_elbow_score)
                    score_components.append(right_wrist_score)
            
            # Calculate overall score if we have components
            if score_components:
                overall_frame_score = sum(score_components) / len(score_components)
            
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw connections
            for connection in connections:
                idx1, idx2 = connection
                
                if idx1 < 5 or idx2 < 5:
                    continue
                
                if idx1 < len(keypoints) and idx2 < len(keypoints):
                    pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                    pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                    
                    if (idx1 < len(scores) and idx2 < len(scores) and 
                        scores[idx1] > 0.5 and scores[idx2] > 0.5):
                        
                        if connection == (5, 6) and shoulders_good:
                            colour = get_colour_from_score(shoulders_aligned_score)
                            thickness = 3
                            line_type = cv2.LINE_AA
                            cv2.line(frame, pt1, pt2, colour, thickness, line_type)
                        elif connection == (5, 11) and left_torso_good:
                            colour = (255, 0, 0)  
                            thickness = 3
                            line_type = cv2.LINE_AA
                            cv2.line(frame, pt1, pt2, colour, thickness, line_type)
                        elif connection == (6, 12) and right_torso_good:
                            colour = (255, 0, 0)  
                            thickness = 3
                            line_type = cv2.LINE_AA
                            cv2.line(frame, pt1, pt2, colour, thickness, line_type)
                        elif connection == (11, 12) and hips_good:
                            colour = fixed_colours.get(connection)
                            thickness = 2
                            line_type = cv2.LINE_AA
                            cv2.line(frame, pt1, pt2, colour, thickness, line_type)
                        elif connection == (5, 7) and left_arm_good:
                            is_leader_detected = detect_if_leader(left_wrist, right_wrist, left_elbow, right_elbow, scores)
                            is_follower_detected = detect_if_follower(left_wrist, right_wrist, left_elbow, right_elbow, scores)
                            
                            if is_follower_detected:
                                colour = get_colour_from_score(follower_left_arm_score)
                            else:
                                colour = get_colour_from_score(left_elbow_score)
                                
                            thickness = 2
                            line_type = cv2.LINE_AA
                            cv2.line(frame, pt1, pt2, colour, thickness, line_type)
                        elif connection == (7, 9) and left_arm_good:
                            is_leader_detected = detect_if_leader(left_wrist, right_wrist, left_elbow, right_elbow, scores)
                            is_follower_detected = detect_if_follower(left_wrist, right_wrist, left_elbow, right_elbow, scores)
                            
                            if is_follower_detected:
                                colour = get_colour_from_score(follower_left_arm_score)
                            else:
                                colour = get_colour_from_score(left_wrist_score)
                                
                            thickness = 2
                            line_type = cv2.LINE_AA
                            cv2.line(frame, pt1, pt2, colour, thickness, line_type)
                        elif connection == (6, 8) and right_arm_good:
                            is_leader_detected = detect_if_leader(left_wrist, right_wrist, left_elbow, right_elbow, scores)
                            is_follower_detected = detect_if_follower(left_wrist, right_wrist, left_elbow, right_elbow, scores)
                            
                            if is_follower_detected:
                                colour = get_colour_from_score(follower_right_arm_score)
                            else:
                                colour = get_colour_from_score(right_elbow_score)
                                
                            thickness = 2
                            line_type = cv2.LINE_AA
                            cv2.line(frame, pt1, pt2, colour, thickness, line_type)
                        elif connection == (8, 10) and right_arm_good:
                            is_leader_detected = detect_if_leader(left_wrist, right_wrist, left_elbow, right_elbow, scores)
                            is_follower_detected = detect_if_follower(left_wrist, right_wrist, left_elbow, right_elbow, scores)
                            
                            if is_follower_detected:
                                colour = get_colour_from_score(follower_right_arm_score)
                            else:
                                colour = get_colour_from_score(right_wrist_score)
                                
                            thickness = 2
                            line_type = cv2.LINE_AA
                            cv2.line(frame, pt1, pt2, colour, thickness, line_type)
                        else:
                            colour = fixed_colours.get(connection, (255, 255, 255))
                            thickness = 2
                            line_type = cv2.LINE_AA
                            cv2.line(frame, pt1, pt2, colour, thickness, line_type)
            
            # Draw keypoints
            for i, (point, score) in enumerate(zip(keypoints, scores)):
                if 5 <= i <= 12 and score > 0.5:
                    x = int(point[0])
                    y = int(point[1])
                    
                    colour = keypoint_colours.get(i, (0, 165, 255))
                    
                    if i == 5 or i == 6:
                        if shoulders_good:
                            colour = get_colour_from_score(shoulders_aligned_score)
                        else:
                            colour = keypoint_colours.get(i)
                    elif i == 7 and left_arm_good:
                        is_follower_detected = detect_if_follower(left_wrist, right_wrist, left_elbow, right_elbow, scores)
                        if is_follower_detected:
                            colour = get_colour_from_score(follower_left_arm_score)
                        else:
                            colour = get_colour_from_score(left_elbow_score)
                    elif i == 8 and right_arm_good:
                        is_follower_detected = detect_if_follower(left_wrist, right_wrist, left_elbow, right_elbow, scores)
                        if is_follower_detected:
                            colour = get_colour_from_score(follower_right_arm_score)
                        else:
                            colour = get_colour_from_score(right_elbow_score)
                    elif i == 9 and left_arm_good:
                        is_follower_detected = detect_if_follower(left_wrist, right_wrist, left_elbow, right_elbow, scores)
                        if is_follower_detected:
                            colour = get_colour_from_score(follower_left_arm_score)
                        else:
                            colour = get_colour_from_score(left_wrist_score)
                    elif i == 10 and right_arm_good:
                        is_follower_detected = detect_if_follower(left_wrist, right_wrist, left_elbow, right_elbow, scores)
                        if is_follower_detected:
                            colour = get_colour_from_score(follower_right_arm_score)
                        else:
                            colour = get_colour_from_score(right_wrist_score)
                   
                    cv2.circle(frame, (x, y), 4, colour, -1)
            
            # Check if leader or follower 
            current_is_leader = detect_if_leader(left_wrist, right_wrist, left_elbow, right_elbow, scores)
            current_is_follower = detect_if_follower(left_wrist, right_wrist, left_elbow, right_elbow, scores)

            # Update role only if a definitive detection is made
            if current_is_leader:
                if detected_role != "LEADER":  
                    detected_role = "LEADER"
                    role_text = "LEADER"
            elif current_is_follower:
                if detected_role != "FOLLOWER":  
                    detected_role = "FOLLOWER"
                    role_text = "FOLLOWER"
    
    score_text = f"Frame Score: {overall_frame_score:.2f}"
    score_color = get_colour_from_score(overall_frame_score)
    cv2.putText(frame, score_text, (20, height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)
                    
    if role_text:
        text_size = cv2.getTextSize(role_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = width - text_size[0] - 20
        text_y = height - 30
        cv2.putText(frame, role_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
                    
    out.write(frame)
    frame_idx += 1
   
    if frame_idx % 100 == 0:
        print(f"Processed {frame_idx} frames...")

# ------- CLEANUP -------
cap.release()
out.release()

print(f"Ballroom dance position analysis done.")
print(f"Total frames processed:{frame_idx}")
print(f"Output saved to:{output_video}")