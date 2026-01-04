import sys
import time
import numpy as np
import cv2  # 导入 OpenCV 用于保存视频
import argparse
import os

sys.path.append("/usr/local/local/lib/python3.8/dist-packages/")

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Simple Viewer sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="Path to input event file (RAW, DAT or HDF5). If not specified, the camera live stream is used. "
             "If it's a camera serial number, it will try to open that camera instead.")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    # 输出视频路径
    output_video_path = "output.avi"

    # Events iterator on Camera or event file
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=1000)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Helper iterator to emulate realtime
    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    # 设置视频保存参数，使用 OpenCV 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编码器
    video_writer = None  # 视频写入对象初始化为空

    # Window - Graphical User Interface
    is_recording = False  # 是否在录制视频
    frame_start_time = None  # 用于跟踪录制开始的时间
    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            nonlocal is_recording, video_writer, frame_start_time

            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

            # 按下 'v' 键开始/停止录制
            if key == UIKeyEvent.KEY_V:
                if not is_recording:
                    # 开始录制
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, 25.0, (width, height))
                    frame_start_time = time.time()  # 记录开始时间
                    print("🎥 Recording started...")
                    is_recording = True
                else:
                    # 停止录制并保存
                    video_writer.release()
                    print(f"✅ Video saved to {output_video_path}")
                    is_recording = False

        window.set_keyboard_callback(keyboard_cb)

        # Event Frame Generator
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height, fps=25,
                                                           palette=ColorPalette.CoolWarm)

        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)
            # 如果正在录制视频，则写入帧到视频
            if is_recording and video_writer is not None:
                video_writer.write(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        # Process events
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)

            if window.should_close():
                break

            # 如果正在录制，确保视频持续时间与事件时间一致
            if is_recording and frame_start_time is not None:
                elapsed_time = time.time() - frame_start_time
                if elapsed_time >= (evs['t'][-1] / 1e6):  # 如果视频录制时间超过事件流时间，则停止录制
                    video_writer.release()
                    print(f"✅ Video saved to {output_video_path}")
                    is_recording = False

        # 如果在结束时仍然在录制，释放视频写入对象
        if video_writer is not None and is_recording:
            video_writer.release()
            print(f"✅ Video saved to {output_video_path}")


if __name__ == "__main__":
    main()
