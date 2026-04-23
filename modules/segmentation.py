import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# LIP labels from SCHP (Self-Correction-Human-Parsing)
LIP_LABELS = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Glove": 3,
    "Sunglasses": 4,
    "Upper-clothes": 5,
    "Dress": 6,
    "Coat": 7,
    "Socks": 8,
    "Pants": 9,
    "Jumpsuits": 10,
    "Scarf": 11,
    "Skirt": 12,
    "Face": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Left-leg": 16,
    "Right-leg": 17,
    "Left-shoe": 18,
    "Right-shoe": 19,
}

# For top-wear try-on preparation, keep torso + sleeves area.
TORSO_CLASSES = [
    LIP_LABELS["Upper-clothes"],
    LIP_LABELS["Dress"],
    LIP_LABELS["Coat"],
    LIP_LABELS["Left-arm"],
    LIP_LABELS["Right-arm"],
]


def detect_shoulders(image_bgr: np.ndarray) -> dict:
    mp_pose = mp.solutions.pose
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as pose:
        results = pose.process(image_rgb)

    if results.pose_landmarks is None:
        return {}

    landmarks = results.pose_landmarks.landmark
    left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    return {
        "left_shoulder": {"x": int(left.x * w), "y": int(left.y * h)},
        "right_shoulder": {"x": int(right.x * w), "y": int(right.y * h)},
    }


def run_schp_inference(
    image_path: Path,
    schp_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    schp_python: Path,
) -> Path:
    schp_root = schp_root.resolve()
    checkpoint_path = checkpoint_path.resolve()
    schp_python = schp_python.resolve()
    output_dir = output_dir.resolve()

    input_dir = output_dir / "_input"
    parsing_dir = output_dir / "parsing"
    input_dir.mkdir(parents=True, exist_ok=True)
    parsing_dir.mkdir(parents=True, exist_ok=True)

    input_image_path = input_dir / image_path.name
    shutil.copy2(image_path, input_image_path)

    extractor = schp_root / "simple_extractor.py"
    command = [
        str(schp_python),
        str(extractor),
        "--dataset",
        "lip",
        "--model-restore",
        str(checkpoint_path),
        "--gpu",
        "0",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(parsing_dir),
    ]

    subprocess.run(command, cwd=str(schp_root), check=True)

    parsing_path = parsing_dir / f"{image_path.stem}.png"
    if not parsing_path.exists():
        raise FileNotFoundError(f"Parsing output not found at: {parsing_path}")
    return parsing_path


def extract_torso_mask(parsing_path: Path, output_dir: Path) -> Path:
    parsing_img = Image.open(parsing_path)
    parsing = np.array(parsing_img, dtype=np.uint8)
    torso_mask = np.isin(parsing, TORSO_CLASSES).astype(np.uint8) * 255

    mask_path = output_dir / "torso_mask.png"
    cv2.imwrite(str(mask_path), torso_mask)
    return mask_path


def apply_mask_on_person(person_bgr: np.ndarray, mask_path: Path, output_dir: Path) -> Path:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    person_only = cv2.bitwise_and(person_bgr, person_bgr, mask=mask)
    output_path = output_dir / "person_torso_only.png"
    cv2.imwrite(str(output_path), person_only)
    return output_path


def save_pose_preview(person_bgr: np.ndarray, shoulders: dict, output_dir: Path) -> Path:
    preview = person_bgr.copy()
    if shoulders:
        left = shoulders["left_shoulder"]
        right = shoulders["right_shoulder"]
        cv2.circle(preview, (left["x"], left["y"]), 12, (0, 255, 0), -1)
        cv2.circle(preview, (right["x"], right["y"]), 12, (0, 255, 0), -1)
        cv2.line(preview, (left["x"], left["y"]), (right["x"], right["y"]), (255, 0, 0), 4)

        cv2.putText(
            preview,
            f"L({left['x']},{left['y']})",
            (left["x"] + 10, max(left["y"] - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"R({right['x']},{right['y']})",
            (right["x"] + 10, max(right["y"] - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        shoulder_width = abs(left["x"] - right["x"])
        cv2.putText(
            preview,
            f"shoulder_px={shoulder_width}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            preview,
            "No pose landmarks detected",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        preview,
        f"run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        (20, preview.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    preview_path = output_dir / "pose_shoulders_preview.jpg"
    cv2.imwrite(str(preview_path), preview)
    return preview_path


def _load_cloth_rgba(cloth_image_path: Path) -> np.ndarray:
    if not cloth_image_path.exists():
        cloth_image_path.parent.mkdir(parents=True, exist_ok=True)
        cloth_rgba = np.zeros((420, 360, 4), dtype=np.uint8)

        # Body panel
        cv2.rectangle(cloth_rgba, (90, 80), (270, 390), (35, 120, 220, 255), -1)
        # Sleeves
        cv2.rectangle(cloth_rgba, (20, 90), (95, 220), (35, 120, 220, 255), -1)
        cv2.rectangle(cloth_rgba, (265, 90), (340, 220), (35, 120, 220, 255), -1)
        # Neck cutout
        cv2.circle(cloth_rgba, (180, 90), 32, (0, 0, 0, 0), -1)

        cv2.imwrite(str(cloth_image_path), cloth_rgba)

    cloth = cv2.imread(str(cloth_image_path), cv2.IMREAD_UNCHANGED)
    if cloth is None:
        raise FileNotFoundError(f"Could not read cloth image: {cloth_image_path}")

    if cloth.ndim == 2:
        cloth = cv2.cvtColor(cloth, cv2.COLOR_GRAY2BGRA)
    elif cloth.shape[2] == 3:
        b, g, r = cv2.split(cloth)
        alpha = np.full_like(b, 255)
        cloth = cv2.merge([b, g, r, alpha])

    return cloth


def _compute_torso_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def create_tryon_preview(
    person_bgr: np.ndarray,
    torso_mask_path: Path,
    shoulders: dict,
    cloth_image_path: Path,
    output_dir: Path,
) -> Path:
    if not shoulders:
        raise RuntimeError("No shoulder keypoints detected. Cannot align cloth.")

    torso_mask = cv2.imread(str(torso_mask_path), cv2.IMREAD_GRAYSCALE)
    if torso_mask is None:
        raise RuntimeError(f"Failed to read torso mask: {torso_mask_path}")

    bbox = _compute_torso_bbox(torso_mask)
    if bbox is None:
        raise RuntimeError("Torso region was empty. Cannot place cloth.")

    x_min, y_min, x_max, y_max = bbox
    torso_h = max(1, y_max - y_min + 1)

    left = shoulders["left_shoulder"]
    right = shoulders["right_shoulder"]
    shoulder_center_x = int((left["x"] + right["x"]) / 2)
    shoulder_center_y = int((left["y"] + right["y"]) / 2)
    shoulder_width = abs(left["x"] - right["x"])
    target_cloth_w = max(40, int(shoulder_width * 1.75))

    cloth_rgba = _load_cloth_rgba(cloth_image_path)
    cloth_h, cloth_w = cloth_rgba.shape[:2]
    scale = target_cloth_w / float(cloth_w)
    target_cloth_h = max(40, int(cloth_h * scale))

    # Keep cloth height in a practical range around torso size.
    target_cloth_h = min(max(target_cloth_h, int(torso_h * 0.75)), int(torso_h * 1.35))

    cloth_rgba = cv2.resize(cloth_rgba, (target_cloth_w, target_cloth_h), interpolation=cv2.INTER_AREA)

    canvas_h, canvas_w = person_bgr.shape[:2]
    x1 = shoulder_center_x - (target_cloth_w // 2)
    y1 = shoulder_center_y - int(target_cloth_h * 0.18)
    x2 = x1 + target_cloth_w
    y2 = y1 + target_cloth_h

    # Clip overlay region to image bounds.
    ox1 = max(0, x1)
    oy1 = max(0, y1)
    ox2 = min(canvas_w, x2)
    oy2 = min(canvas_h, y2)
    if ox1 >= ox2 or oy1 >= oy2:
        raise RuntimeError("Cloth placement went out of bounds. Try another image.")

    cx1 = ox1 - x1
    cy1 = oy1 - y1
    cx2 = cx1 + (ox2 - ox1)
    cy2 = cy1 + (oy2 - oy1)

    cloth_crop = cloth_rgba[cy1:cy2, cx1:cx2]
    cloth_rgb = cloth_crop[:, :, :3].astype(np.float32)
    cloth_alpha = (cloth_crop[:, :, 3].astype(np.float32) / 255.0)

    person_roi = person_bgr[oy1:oy2, ox1:ox2].astype(np.float32)
    torso_roi_mask = (torso_mask[oy1:oy2, ox1:ox2].astype(np.float32) / 255.0)

    # Blend cloth mostly within the segmented torso area.
    blend_alpha = np.clip(cloth_alpha * torso_roi_mask * 0.95, 0.0, 1.0)
    blend_alpha_3c = np.dstack([blend_alpha, blend_alpha, blend_alpha])

    blended_roi = (cloth_rgb * blend_alpha_3c) + (person_roi * (1.0 - blend_alpha_3c))

    result = person_bgr.copy()
    result[oy1:oy2, ox1:ox2] = blended_roi.astype(np.uint8)

    # Add small overlay guide for debugging placement.
    cv2.rectangle(result, (ox1, oy1), (ox2, oy2), (0, 255, 255), 1)

    output_path = output_dir / "tryon_result.png"
    cv2.imwrite(str(output_path), result)
    return output_path


def segment_person_with_schp(
    person_image: Path,
    schp_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    schp_python: Path,
    cloth_image: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    person_bgr = cv2.imread(str(person_image))
    if person_bgr is None:
        raise FileNotFoundError(f"Could not read image: {person_image}")

    parsing_path = run_schp_inference(person_image, schp_root, checkpoint_path, output_dir, schp_python)
    torso_mask_path = extract_torso_mask(parsing_path, output_dir)
    torso_only_path = apply_mask_on_person(person_bgr, torso_mask_path, output_dir)
    shoulders = detect_shoulders(person_bgr)
    pose_preview_path = save_pose_preview(person_bgr, shoulders, output_dir)
    tryon_path = create_tryon_preview(person_bgr, torso_mask_path, shoulders, cloth_image, output_dir)

    result = {
        "input_image": str(person_image.resolve()),
        "parsing_map": str(parsing_path.resolve()),
        "torso_mask": str(torso_mask_path.resolve()),
        "torso_cutout": str(torso_only_path.resolve()),
        "pose_preview": str(pose_preview_path.resolve()),
        "tryon_result": str(tryon_path.resolve()),
        "shoulders": shoulders,
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrained SCHP segmentation + MediaPipe shoulders")
    parser.add_argument("--image", default="test.jpg", help="Path to person image")
    parser.add_argument(
        "--schp-root",
        default="../Self-Correction-Human-Parsing",
        help="Path to Self-Correction-Human-Parsing repo",
    )
    parser.add_argument(
        "--checkpoint",
        default="../Self-Correction-Human-Parsing/checkpoints/exp-schp-201908261155-lip.pth",
        help="Path to pretrained SCHP LIP checkpoint (.pth)",
    )
    parser.add_argument(
        "--schp-python",
        default="../Self-Correction-Human-Parsing/venv/Scripts/python.exe",
        help="Python interpreter used to run SCHP simple_extractor.py",
    )
    parser.add_argument(
        "--cloth-image",
        default="cloth.png",
        help="Path to cloth image (prefer PNG with transparent background)",
    )
    parser.add_argument("--output-dir", default="output", help="Directory to store outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    person_image = Path(args.image)
    schp_root = Path(args.schp_root)
    checkpoint = Path(args.checkpoint)
    schp_python = Path(args.schp_python)
    cloth_image = Path(args.cloth_image)
    output_dir = Path(args.output_dir)

    result = segment_person_with_schp(person_image, schp_root, checkpoint, output_dir, schp_python, cloth_image)
    print("Segmentation completed:")
    for key, value in result.items():
        print(f"- {key}: {value}")