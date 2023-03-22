import matplotlib
import warnings
matplotlib.use('Agg')

import imageio
import tqdm
import numpy as np

from argparse import ArgumentParser
from util import crop_bbox_from_frames, bb_intersection_over_union, join, compute_aspect_preserved_bbox, one_box_inside_other
import os

import cv2
from skimage.transform import resize
from skimage.color import rgb2gray

import mmcv
from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from util import scheduler


def check_full_person(kps, args):
    head_present = np.sum((kps[:5] > args.kp_confidence_th))
    leg_present = np.sum((kps[-4:] > args.kp_confidence_th))
    return head_present and leg_present


def check_camera_motion(current_frame, previous_frame):
    flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.quantile(mag, [0.25, 0.5, 0.75], overwrite_input=True)


def store(video_path, trajectories, args, chunks_data, fps, frame):
    for i, (tube_bbox, bbox, start, end) in enumerate(trajectories):
        final_bbox = compute_aspect_preserved_bbox(bbox, args.increase)
        video_id = os.path.basename(video_path).split('.')[0]
        name = (video_id + "#" + str(start).zfill(6) + "#" + str(end).zfill(6) + ".mp4")
        # partition = 'test' if video_id in test_videos else 'train'
        chunks_data.append({'bbox': '-'.join(map(str, final_bbox)), 'start': start, 'end': end, 'fps': fps, 'partition': 'train',
                            'video_id': video_id, 'height': frame.shape[0], 'width': frame.shape[1]})

def get_bbox_from_kps(kps, bbox_det):
    left, top = np.min(kps[:13], axis=0)
    right, bot = np.max(kps[:13], axis=0)
    top = min(top, bbox_det[1])
    return np.array([left, top, right, bot])


def process_video(video_path, det_model, pose_model, dataset, dataset_info, args):
    video = mmcv.VideoReader(video_path)
    fps = video.fps
    trajectories = []
    previous_frame = None
    chunks_data = []
    try:
        for i, frame in enumerate(video):
            if args.minimal_video_size > min(frame.shape[0], frame.shape[1]):
                return chunks_data
            if i % args.sample_rate != 0:
                continue

            mmdet_results = inference_detector(det_model, frame)
            person_results = process_mmdet_results(mmdet_results, 1)

            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                frame,
                person_results,
                bbox_thr=0.3,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None)

            if pose_results:
                keypoints = np.array([pose_result['keypoints'][:, :2] for pose_result in pose_results])
                scores = np.array([pose_result['bbox'][-1] for pose_result in pose_results])
                keypoint_scores = np.array([pose_result['keypoints'][:, -1] for pose_result in pose_results])
                bboxes_det = np.array([pose_result['bbox'][:4] for pose_result in pose_results])
                bboxes = np.array([get_bbox_from_kps(keypoints[i], bboxes_det[i]) for i in range(len(keypoints))])
            else:
                keypoints = np.zeros((1, 17, 2))
                scores = np.zeros((1, ))
                keypoint_scores = np.zeros((1, 17, 1))
                bboxes = np.zeros((1, 4))
            
            ## Check if valid person in bbox
            height_criterion = ((bboxes[:, 3] - bboxes[:, 1]) > args.mimial_person_size * frame.shape[1])
            score_criterion = (scores > args.bbox_confidence_th)
            full_person_criterion = np.array([check_full_person(kps, args) for kps in keypoint_scores])

            criterion = np.logical_and(height_criterion, score_criterion)
            criterion = np.logical_and(full_person_criterion, criterion)
            bboxes_valid = bboxes[criterion]

            ### Check if frame is valid
            if previous_frame is None:
                previous_frame = rgb2gray(
                    resize(frame, (256, 256), preserve_range=True, anti_aliasing=True, mode='constant'))

                current_frame = previous_frame
                previous_intensity = np.median(frame.reshape((-1, frame.shape[-1])), axis=0)
                current_intensity = previous_intensity
            else:
                current_frame = rgb2gray(
                    resize(frame, (256, 256), preserve_range=True, anti_aliasing=True, mode='constant'))
                current_intensity = np.median(frame.reshape((-1, frame.shape[-1])), axis=0)

            flow_quantiles = check_camera_motion(current_frame, previous_frame)
            camera_criterion = flow_quantiles[1] > args.camera_change_threshold
            previous_frame = current_frame
            intensity_criterion = np.max(np.abs(previous_intensity - current_intensity)) > args.intensity_change_threshold
            previous_intensity = current_intensity
            no_person_criterion = len(person_results) == 0
            criterion = no_person_criterion or camera_criterion or intensity_criterion

            if criterion:
                bboxes_valid = []

            ## For each trajectory check the criterion
            not_valid_trajectories = []
            valid_trajectories = []

            for trajectory in trajectories:
                tube_bbox = trajectory[0]
                intersection = 0

                for bbox in bboxes_valid:
                    intersection = max(intersection, bb_intersection_over_union(tube_bbox, bbox))

                if (trajectory[3] - trajectory[2]) >= args.max_frames:
                    not_valid_trajectories.append(trajectory)
                elif intersection > 0.2:
                    valid_trajectories.append(trajectory)
                else:
                    not_valid_trajectories.append(trajectory)

            store(video_path, not_valid_trajectories, args, chunks_data, fps, frame)
            trajectories = valid_trajectories

            ## Assign bbox to trajectories, create new trajectories
            for bbox in bboxes_valid:
                intersection = 0
                current_trajectory = None
                for trajectory in trajectories:
                    tube_bbox = trajectory[0]
                    current_intersection = bb_intersection_over_union(tube_bbox, bbox)
                    
                    if intersection < current_intersection and current_intersection > 0.2:
                        intersection = bb_intersection_over_union(tube_bbox, bbox)
                        current_trajectory = trajectory

                ## Create new trajectory
                if current_trajectory is None:
                    trajectories.append([bbox, bbox, i, i])
                else:
                    current_trajectory[3] = i
                    current_trajectory[1] = join(current_trajectory[1], bbox)

            if len(chunks_data) > args.max_crops:
                break

    except IndexError:
        None

    store(video_path, trajectories, args, chunks_data, fps, previous_frame)
    return chunks_data

def run(params):
    video_file, device_id, args = params
    device = 'cuda:' + device_id
    det_model = init_detector(args.det_config, args.det_checkpoint, device)
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device)

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    return process_video(os.path.join(args.video_folder, video_file), det_model, pose_model, dataset, dataset_info, args)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--video_folder", default='youtube', help='Path to folder with videos')
    parser.add_argument("--device_ids", default="0,1", type=str, help="Device to run video on")
    parser.add_argument("--workers", default=1, type=int, help="Number of workers")

    parser.add_argument("--image_shape", default=None, type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape, None - for no resize")
    parser.add_argument("--increase", default=0.05, type=float, help='Increase bbox by this amount')
    # parser.add_argument("--min_frames", default=128, type=int, help='Mimimal number of frames')
    parser.add_argument("--max_frames", default=1024, type=int, help='Maximal number of frames')
    parser.add_argument("--min_size", default=256, type=int, help='Minimal allowed size')

    parser.add_argument("--out_folder", default="output-256", help="Folder with output videos")
    parser.add_argument("--bbox_confidence_th", default=0.9, type=float, help="Maskrcnn confidence for bbox")
    parser.add_argument("--kp_confidence_th", default=0.9, type=float, help="Maskrcnn confidence for keypoint")

    parser.add_argument("--det_config",
                        default="cfg/faster_rcnn_r50_fpn_coco.py",
                        help="Path to detector config file")

    parser.add_argument("--det_checkpoint",
                        default="https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
                        help="Path to detector checkpoint file")

    parser.add_argument("--pose_config",
                        default="cfg/hrnet_w48_coco_256x192.py",
                        help="Path to pose net config file")

    parser.add_argument("--pose_checkpoint",
                        default="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
                        help="Path to pose net checkpoint file")

    parser.add_argument("--mimial_person_size", default=0.10, type=float,
                        help="Minimal person size, e.g 10% of height")

    parser.add_argument("--minimal_video_size", default=300, type=int, help="Minimal size of the video")

    parser.add_argument("--camera_change_threshold", type=float, default=1)
    parser.add_argument("--intensity_change_threshold", type=float, default=1.5)
    parser.add_argument("--sample_rate", type=int, default=1, help="Sample video rate")
    parser.add_argument("--max_crops", type=int, default=1000, help="Maximal number of crops per video.")
    parser.add_argument("--chunks_metadata", default='output-metadata.csv', help="File to store metadata for taichi.")

    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    scheduler(os.listdir(args.video_folder), run, args)