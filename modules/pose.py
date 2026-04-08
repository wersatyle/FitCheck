import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_pose(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Could not read image")
        return None
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:
        results = pose.process(imgRGB)
    
    if results.pose_landmarks is None:
        print("No person detected")
        return None
    
    landmarks = results.pose_landmarks.landmark
    keypoints = {
    # Head/Neck
    "nose":             {"x": landmarks[0].x,  "y": landmarks[0].y},
    
    # Torso (for tops)
    "left_shoulder":    {"x": landmarks[11].x, "y": landmarks[11].y},
    "right_shoulder":   {"x": landmarks[12].x, "y": landmarks[12].y},
    "left_hip":         {"x": landmarks[23].x, "y": landmarks[23].y},
    "right_hip":        {"x": landmarks[24].x, "y": landmarks[24].y},
    
    # Arms (for sleeves)
    "left_elbow":       {"x": landmarks[13].x, "y": landmarks[13].y},
    "right_elbow":      {"x": landmarks[14].x, "y": landmarks[14].y},
    "left_wrist":       {"x": landmarks[15].x, "y": landmarks[15].y},
    "right_wrist":      {"x": landmarks[16].x, "y": landmarks[16].y},
    
    # Legs (for bottoms/pants)
    "left_knee":        {"x": landmarks[25].x, "y": landmarks[25].y},
    "right_knee":       {"x": landmarks[26].x, "y": landmarks[26].y},
    "left_ankle":       {"x": landmarks[27].x, "y": landmarks[27].y},
    "right_ankle":      {"x": landmarks[28].x, "y": landmarks[28].y},
}
    
    mp_drawing.draw_landmarks(
        img,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
    
    cv2.imwrite("pose_output.jpg", img)
    print("Saved pose_output.jpg")
    
    return keypoints

if __name__ == "__main__":
    result = detect_pose("test_image.jpg")
    print(result)