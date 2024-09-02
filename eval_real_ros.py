"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

import os
import sys

sys.path.append("/home/dc/mambaforge/envs/robodiff/lib/python3.9/site-packages")
# sys.path.append(
#     "/media/dc/CLEAR/arx/dp_inference/dp"
# )

import time
import math
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
# from consistency_policy.policy_wrapper import PolicyWrapperRobomimic
from consistency_policy.utils import get_policy, get_cfg

from common.precise_sleep import precise_wait
from utils.real_inference_utils import get_real_obs_resolution, get_real_obs_dict
# from codebase.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply

from diffusion_policy.common.pytorch_util import dict_apply
# from codebase.diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
# from codebase.diffusion_policy.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from utils.cv2_utils import get_image_transform

from codebase.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from arm_control.msg import JointInformation
from arm_control.msg import JointControl
from arm_control.msg import PosCmd
from sensor_msgs.msg import Image
import threading
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

gripper_width_min = -0.016
gripper_width_max = 0.4
actual_gripper_width_min = 0.0
actual_gripper_width_max = 5.0

OmegaConf.register_new_resolver("eval", eval, replace=True)
np.set_printoptions(suppress=True)

@click.command()
@click.option("--input_path", "-ip", required=True, help="Path to checkpoint",)
@click.option("--output_path", "-op", required=True, default="/media/dc/SJQ/diffusion-policy-inference-master/dp_real_eval_video", help="Output video path")
@click.option("--frequency", "-f", default=10, type=int, help="control frequency")
@click.option("--steps_per_inference", "-si", default=8, type=int, help="Action horizon for inference.")
@click.option("--max_step", "-mt", default=1350, type=int, help="Max time step for deployment in an episode")

# @profile
def main(
    input_path,
    output_path,
    frequency,
    steps_per_inference,
    max_step
):
    global obs_ring_buffer, current_step


    dt = 1 / frequency
    video_capture_fps = 30
    max_obs_buffer_size = 30

    #################### get policy ####################
    # load checkpoint
    ckpt_path = input_path
    policy = get_policy(ckpt_path)
    cfg = get_cfg(ckpt_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = policy.to(device)
    policy.eval()
    for param in policy.parameters():
        param.requires_grad = False
    # payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    # cfg = payload["cfg"]
    # print(cfg)
    # # exit()
    # cfg._target_ = "codebase.diffusion_policy." + cfg._target_  # can delete

    # cls = hydra.utils.get_class(cfg._target_)
    # workspace: BaseWorkspace = cls(cfg)
    # workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # policy: BaseImagePolicy
    # policy = workspace.model
    # if cfg.training.use_ema:
    #     policy = workspace.ema_model

    # device = torch.device("cuda:0")
    # policy.eval().to(device)
    #################### get policy ####################

    ## set inference params
    # policy.num_inference_steps = 16  # DDIM inference iterations
    # policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    
    # shared memory
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    examples = dict()
    examples["mid"] = np.empty(shape=obs_res[::-1] + (3,), dtype=np.uint8)  # 添加RGB通道维度
    examples["right"] = np.empty(shape=obs_res[::-1] + (3,), dtype=np.uint8)
    examples["eef_qpos"] = np.empty(shape=(7,), dtype=np.float64)
    examples["qpos"] = np.empty(shape=(7,), dtype=np.float64)
    # examples["mid_orig"] = np.empty(shape=(480, 640, 3, ), dtype=np.uint8)
    # examples["right_orig"] = np.empty(shape=(480, 640, 3, ), dtype=np.uint8)
    examples["timestamp"] = 0.0
    obs_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
        shm_manager=shm_manager,
        examples=examples,
        get_max_k=max_obs_buffer_size,
        get_time_budget=0.2,
        put_desired_frequency=video_capture_fps,
    )

    # visuilization

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_video = cv2.VideoWriter(output_path + "/replay.mp4", fourcc, 1.0, (448, 224))

    session_dir = os.listdir(output_path)
    idx_max = 0
    for session in session_dir:
        idx_str = session.split('_')[-1]
        idx_max = max(idx_max, int(idx_str))
    new_dir_str = output_path + "/episode_" + str(idx_max + 1)
    new_dir = pathlib.Path(new_dir_str)
    new_dir.mkdir()

    mid_img_dir = new_dir.joinpath("mid")
    right_img_dir = new_dir.joinpath("right")
    mid_img_dir.mkdir()
    right_img_dir.mkdir()

    fourcc_visualization = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_visualization = cv2.VideoWriter(new_dir_str + "/video.mp4", fourcc_visualization, 30.0, (1280, 480))

    fourcc_mid = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_mid = cv2.VideoWriter(new_dir_str + "/mid/mid.mp4", fourcc_mid, 30.0, (640, 480))

    fourcc_right = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_right = cv2.VideoWriter(new_dir_str + "/right/right.mp4", fourcc_right, 30.0, (640, 480))
    
    current_step = 0
    start_time = math.inf

    # ros init and subscriber
    rospy.init_node("eval_real_ros")
    eef_qpos = Subscriber("follow1_pos_back", PosCmd)
    qpos = Subscriber("joint_information", JointInformation)
    mid = Subscriber("mid_camera", Image)
    right = Subscriber("right_camera", Image)
    # control_robot2 = rospy.Publisher("test_right", JointControl, queue_size=10)
    control_robot2 = rospy.Publisher("follow_joint_control_1", JointControl, queue_size=10)
    ats = ApproximateTimeSynchronizer(
        [eef_qpos, qpos, mid, right], queue_size=10, slop=0.1
    )
    ats.registerCallback(
        lambda *msg: callback(*msg, output_video_visualization, output_video_mid, output_video_right, max_step, start_time)
    )
    # ats.registerCallback(callback)
    rate = rospy.Rate(frequency)


    # data ??
    last_data = None
    right_control = JointControl()
    

    # start
    # ========== policy control loop ==============
    # start episode
    policy.reset()
    start_delay = 1.0
    eval_t_start = time.time() + start_delay  # time.time()返回系统当前时间的最佳估计,可能受系统时钟调整的影响(如NTP),
                     # 而time.monotonic()返回单调递增的时间,不受系统时钟变化的影响
    t_start = time.monotonic() + start_delay
    frame_latency = 1/30
    precise_wait(eval_t_start - frame_latency, time_func=time.time)
    print("Started!")
    iter_idx = 0
    start_time = time.time()
    

    # inference loop
    while not rospy.is_shutdown():
        if current_step >= max_step:
            break

        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

        # get observation
        k = math.ceil(n_obs_steps * (video_capture_fps / frequency))
        last_data = obs_ring_buffer.get_last_k(k=k, out=last_data)
        last_timestamp = last_data["timestamp"][-1] # 这行代码获取最近观测数据的时间戳
        obs_align_timestamps = last_timestamp - (np.arange(n_obs_steps)[::-1] * dt) # [::-1] 翻转序列
        
        # 对齐时间戳
        obs_dict = dict()
        this_timestamps = last_data["timestamp"] 
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps <= t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:  # 不止一个
                this_idx = is_before_idxs[-1] # 找最后一个
            this_idxs.append(this_idx)
        for key in last_data.keys(): 
            obs_dict[key] = last_data[key][this_idxs]

        obs_timestamps = obs_dict["timestamp"]
        print("Got Observation!")

        # run inference
        with torch.no_grad():
            test_t_start = time.perf_counter()
            obs_dict_np = get_real_obs_dict(
                env_obs=obs_dict, shape_meta=cfg.task.shape_meta)
            obs_dict = dict_apply(obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(torch.device("cuda:0")))

            # image = obs_dict['right'][0, 1, :, :, :]
            # image_converted = np.transpose(image.cpu().numpy(), (1, 2, 0))
            # plt.imshow(image_converted)
            # plt.show()
            # import ipdb;ipdb.set_trace()
            # exit()
            print(obs_dict['qpos'])
            result = policy.predict_action(obs_dict)
            
            action = result["action"][0].detach().to("cpu").numpy()
            # print(action)
            # print("----------------------------")
            print(f"Inference latency {steps_per_inference/(time.perf_counter() - test_t_start)}")

        # visualization
        mid_img = obs_dict["mid"][0, -1]
        mid_img = mid_img.permute(1, 2, 0)
        mid_img = mid_img.cpu().numpy()
        mid_img = (255 * mid_img).astype(np.uint8)

        right_img = obs_dict["right"][0, -1]
        right_img = right_img.permute(1, 2, 0)
        right_img = right_img.cpu().numpy()
        right_img = (255 * right_img).astype(np.uint8)

        # vis_img = np.zeros((224, 448, 3), dtype=np.uint8)
        vis_img = np.zeros((120, 160*2, 3), dtype=np.uint8)
        # vis_img[:, :224, :] = mid_img
        # vis_img[:, 224:448, :] = right_img
        vis_img[:, :160, :] = mid_img
        vis_img[:, 160:320, :] = right_img

        # cv2.imshow('Multi Camera Viewer', vis_img)
        # output_video.write(vis_img)
        key_stroke = cv2.pollKey()
        if key_stroke == ord('q'):
            # Stop episode
            # Hand control back to human
            print('Stopped.')
            break

        # preprocess action
        action = action[:steps_per_inference, :]
        action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]

        action_exec_latency = 0.01
        curr_time = time.time()
        is_new = action_timestamps > (curr_time + action_exec_latency)

        if np.sum(is_new) == 0:
            action = action[[-1]]
            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
            action_timestamp = eval_t_start + (next_step_idx) * dt
            print("Over budget", action_timestamp - curr_time)
            action_timestamps = np.array([action_timestamp])
        else:
            action = action[is_new]
            action_timestamps = action_timestamps[is_new]
            
        # execute actions
        print("Execute Action!")
        print(action)
        print("==================================")
        for item in action:
            right_control.joint_pos = item

            control_robot2.publish(right_control)

            rate.sleep()

        precise_wait(t_cycle_end - frame_latency)
        iter_idx += steps_per_inference

    # output_video.release()
    output_video_visualization.release()
    output_video_mid.release()
    output_video_right.release()
    cv2.destroyAllWindows()

    

def callback(eef_qpos, qpos, image_mid, image_right, output_video_visualization, output_video_mid, output_video_right, max_step, start_time):
    global obs_ring_buffer, current_step

    mid = image_mid
    right = image_right
    bridge = CvBridge()
    receive_time = time.time()

    obs_data = dict()
    obs_data["eef_qpos"] = np.array(
        [
            eef_qpos.x,
            eef_qpos.y,
            eef_qpos.z,
            eef_qpos.roll,
            eef_qpos.pitch,
            eef_qpos.yaw,
            eef_qpos.gripper,
        ]
    )
    gripper_width = qpos.joint_pos[6]

    # gripper width calibration
    # gripper_width = (gripper_width - actual_gripper_width_min) / (actual_gripper_width_max - actual_gripper_width_min) * \
    #                 (gripper_width_max - gripper_width_min) + gripper_width_min

    gripper_width = gripper_width / 12 - 0.016 
    obs_data["qpos"] = np.array(
        [
            qpos.joint_pos[0],
            qpos.joint_pos[1],
            qpos.joint_pos[2],
            qpos.joint_pos[3],
            qpos.joint_pos[4],
            qpos.joint_pos[5],
            gripper_width
        ]
    )

    # process images observation
    mid = bridge.imgmsg_to_cv2(mid, "bgr8")
    right = bridge.imgmsg_to_cv2(right, "bgr8")
    obs_data["mid"] = transform(mid,obs_image_resolution = (160,120))
    obs_data["right"] = transform(right,obs_image_resolution = (160,120))
    obs_data["timestamp"] = receive_time
    # obs_data["mid_orig"] = mid
    # obs_data["right_orig"] = right

    # img_to_save = np.zeros((480, 640, 3), dtype=np.uint8)
    # img_to_save[:,:,:] = mid
    # idx_max = 0
    # for saved_img in list(mid_img_dir.glob('*.jpg')):
    #     # print(saved_img.stem)
    #     idx_str = str(saved_img.stem).split('_')[-1]
    #     idx_max = max(idx_max, int(idx_str))
    # save_path = mid_img_dir.joinpath("mid_{}.jpg".format(idx_max + 1))
    # cv2.imwrite(str(save_path), img_to_save)

    # img_to_save = np.zeros((480, 640, 3), dtype=np.uint8)
    # img_to_save[:,:,:] = right
    # idx_max = 0
    # for saved_img in list(right_img_dir.glob('*.jpg')):
    #     # print(saved_img.stem)
    #     idx_str = str(saved_img.stem).split('_')[-1]
    #     idx_max = max(idx_max, int(idx_str))
    # save_path = right_img_dir.joinpath("right_{}.jpg".format(idx_max + 1))
    # cv2.imwrite(str(save_path), img_to_save)

    # save video
    if current_step < max_step and time.time() > start_time:
        output_video_mid.write(mid)
        output_video_right.write(right)

        canvas = np.zeros((480, 1280, 3), dtype=np.uint8)
        canvas[:, :640, :] = mid
        canvas[:, 640:1280, :] = right

        output_video_visualization.write(canvas)

        current_step += 1
    
    # cv2.imshow('Multi Camera Viewer2', canvas)
    # key_stroke = cv2.pollKey()
    # if cv2.waitKey(1) == ord("q"):
    #     rospy.signal_shutdown("User requested shutdown")
    
    # put data
    put_data = obs_data
    obs_ring_buffer.put(put_data, wait=False)

def transform(data, video_capture_resolution=(640, 480), obs_image_resolution=(224, 224)):
    color_tf = get_image_transform(
                input_res=video_capture_resolution,
                output_res=obs_image_resolution,
                # obs output rgb
                bgr_to_rgb=False,
            )
    
    tf_data = color_tf(data)
    return tf_data

if __name__ == "__main__":
    main()