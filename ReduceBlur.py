import sys
import time
import numpy as np
import cv2
import argparse
import os

sys.path.append("/usr/local/local/lib/python3.8/dist-packages/")

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_core.event_io.raw_reader import initiate_device  # ✅ 新增：先建 device 以配置相机
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent


def parse_args():
    parser = argparse.ArgumentParser(description='Metavision Simple Viewer sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="Path to input event file (RAW, DAT or HDF5). If not specified, the camera live stream is used. "
             "If it's a camera serial number, it will try to open that camera instead.")
    # ✅ 新增：控制事件数量（通过控制事件率）
    parser.add_argument(
        '--cd-event-rate', type=int, default=0,
        help="Target CD event rate (events/s). 0 means disabled. Works only on live cameras with ERC facility.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    output_video_path = "output.avi"
    CD_EVENT_RATE = 10000_000

    # ✅ 先创建 device，这样才能在创建 EventsIterator 前配置相机（ERC/ROI/bias 等）
    device = initiate_device(args.event_file_path)  # path="" 会打开第一台可用相机；也可传序列号或RAW路径

    # ✅ 仅对“直播相机”启用 ERC（文件读取不支持）
    # ✅ 仅对 live camera 启用 ERC（文件读取不支持）
    if is_live_camera(args.event_file_path):
        erc = device.get_i_erc_module()
        if erc is None:
            print("⚠️ ERC module not available on this device (sensor may not support ERC).")
        else:
            erc.enable(True)
            erc.set_cd_event_rate(CD_EVENT_RATE)
            print(f"✅ ERC enabled: target CD event rate = {CD_EVENT_RATE} events/s")

    # ✅ 用 from_device 创建 EventsIterator（关键：把你配置过的 device 传进去）
    mv_iterator = EventsIterator.from_device(device=device, delta_t=1000)
    height, width = mv_iterator.get_size()

    # Helper iterator to emulate realtime (only for files)
    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = None

    is_recording = False
    frame_start_time = None

    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            nonlocal is_recording, video_writer, frame_start_time

            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

            if key == UIKeyEvent.KEY_V:
                if not is_recording:
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, 25.0, (width, height))
                    frame_start_time = time.time()
                    print("🎥 Recording started...")
                    is_recording = True
                else:
                    video_writer.release()
                    print(f"✅ Video saved to {output_video_path}")
                    is_recording = False

        window.set_keyboard_callback(keyboard_cb)

        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height, fps=25,
                                                           palette=ColorPalette.CoolWarm)

        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)
            if is_recording and video_writer is not None:
                video_writer.write(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)

            if window.should_close():
                break

            if is_recording and frame_start_time is not None:
                elapsed_time = time.time() - frame_start_time
                if elapsed_time >= (evs['t'][-1] / 1e6):
                    video_writer.release()
                    print(f"✅ Video saved to {output_video_path}")
                    is_recording = False

        if video_writer is not None and is_recording:
            video_writer.release()
            print(f"✅ Video saved to {output_video_path}")


if __name__ == "__main__":
    main()
