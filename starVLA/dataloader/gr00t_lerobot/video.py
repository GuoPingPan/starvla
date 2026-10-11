# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import av
import cv2
import numpy as np

import torch  # noqa: F401 # isort: skip
import torchvision  # noqa: F401 # isort: skip

# Import decord with graceful fallback
try:
    import decord  # noqa: F401

    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import torchcodec

    TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError):
    TORCHCODEC_AVAILABLE = False


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_at(indices=indices).data.numpy()
    elif video_backend == "opencv":
        frames = []
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "pyav":
        # Use PyAV directly without torchvision wrapper
        container = None
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            fps = float(stream.average_rate) if stream.average_rate else float(stream.guessed_rate)
            
            loaded_frames = []
            
            for target_idx in indices:
                # Convert frame index to timestamp, then to pts
                target_ts = target_idx / fps
                target_pts = int(target_ts / time_base)
                
                # Seek to the target position
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)
                
                closest_frame = None
                closest_idx_diff = float('inf')
                
                for frame in container.decode(video=0):
                    current_ts = float(frame.pts * time_base)
                    current_idx = int(round(current_ts * fps))
                    current_diff = abs(current_idx - target_idx)
                    
                    if current_diff < closest_idx_diff:
                        closest_idx_diff = current_diff
                        closest_frame = frame
                    
                    # If we've passed the target and diff is increasing, stop
                    if current_idx > target_idx and current_diff > closest_idx_diff:
                        break
                    
                    # Stop if we're significantly past the target
                    if current_idx > target_idx + 5:
                        break
                
                if closest_frame is not None:
                    frame_data = closest_frame.to_ndarray(format="rgb24")
                    loaded_frames.append(frame_data)
                else:
                    raise ValueError(f"Unable to find frame at index {target_idx}")
            
            frames = np.array(loaded_frames)
            return frames
            
        finally:
            if container is not None:
                container.close()
    else:
        raise NotImplementedError


# === Episode-level video reader cache (for fast eval) ============================
# 显式由 preload_video_cache() 填充、clear_video_cache() 释放;get_frames_by_timestamps()
# 只读、不自动填(避免训练时无界增长)。键为视频绝对路径。
#
# 只缓存"已打开的解码器"(decord: VideoReader + 每帧 timestamp 元数据;torchvision_av:
# VideoReader 句柄),**不解码全部帧**:UMI 等数据集的视频按视频帧率录整条长轨迹(几千~上万帧),
# 而 eval 只需其中少量 anchor 帧;若一次性解码全部帧,torchvision_av(pyav 逐帧)会慢到上百秒。
# 缓存打开的 reader 即可消除每次 get_video 重新 open/扫描的开销,取帧时仍只按需解码。
_VIDEO_READER_CACHE: dict[str, dict] = {}


def preload_video_cache(
    video_path: str,
    video_backend: str = "decord",
    video_backend_kwargs: dict | None = None,
) -> None:
    """打开视频并缓存其解码器(**不解码全部帧**),供后续 get_frames_by_timestamps 复用、省去每次重新 open。

    支持 decord / torchvision_av;同一 path 重复调用是 no-op。eval 在进推理循环前对每条轨迹的
    各相机调用一次(毫秒级),循环内取帧复用 reader 只按需解码。
    """
    if video_path in _VIDEO_READER_CACHE:
        return
    video_backend_kwargs = video_backend_kwargs or {}
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        # 只缓存 timestamp 元数据(扫描,不解码),供命中时 argmin 选帧
        frame_ts: np.ndarray = vr.get_frame_timestamp(range(len(vr)))
        _VIDEO_READER_CACHE[video_path] = {"vr": vr, "frame_ts": frame_ts}
    elif video_backend == "torchvision_av":
        torchvision.set_video_backend("pyav")
        reader = torchvision.io.VideoReader(video_path, "video")
        _VIDEO_READER_CACHE[video_path] = {"reader": reader}
    else:
        raise NotImplementedError(
            f"preload_video_cache only supports decord / torchvision_av, got {video_backend!r}"
        )


def clear_video_cache(video_paths: list[str] | None = None) -> None:
    """释放缓存的解码器(关闭文件句柄)。``video_paths=None`` 清空全部;否则只清指定路径。"""
    paths = list(_VIDEO_READER_CACHE.keys()) if video_paths is None else video_paths
    for p in paths:
        entry = _VIDEO_READER_CACHE.pop(p, None)
        if entry is None:
            continue
        vr = entry.get("vr") or entry.get("reader")
        try:  # torchvision.io.VideoReader 持有 container;decord.VideoReader 无显式 close
            if hasattr(vr, "container") and vr.container is not None:
                vr.container.close()
                vr.container = None
            if hasattr(vr, "_c"):
                vr._c = None
        except Exception:  # noqa: BLE001 — 释放 best-effort
            pass


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    """Get frames from a video at specified timestamps.
    Args:
        video_path (str): Path to the video file.
        timestamps (list[int] | np.ndarray): Timestamps to retrieve frames for, in seconds.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    if video_backend == "decord":
        # For some GPUs, AV format data cannot be read
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        cached = _VIDEO_READER_CACHE.get(video_path)
        if cached is not None:
            # 命中 reader 缓存:复用已打开的 VideoReader + timestamp 元数据,只按需 get_batch 解码(数值与现解一致)
            indices = np.abs(cached["frame_ts"][:, :1] - timestamps).argmin(axis=0)
            return cached["vr"].get_batch(indices).asnumpy()
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        num_frames = len(vr)
        # Retrieve the timestamps for each frame in the video
        frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
        # Map each requested timestamp to the closest frame index
        # Only take the first element of the frame_ts array which corresponds to start_seconds
        indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_played_at(seconds=timestamps).data.numpy()
    elif video_backend == "opencv":
        # Open the video file
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        # Retrieve the total number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate timestamps for each frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ts = np.arange(num_frames) / fps
        frame_ts = frame_ts[:, np.newaxis]  # Reshape to (num_frames, 1) for broadcasting
        # Map each requested timestamp to the closest frame index
        indices = np.abs(frame_ts - timestamps).argmin(axis=0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "pyav":
        # Use PyAV directly without torchvision wrapper
        container = None
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            # Get video properties
            time_base = float(stream.time_base)
            fps = float(stream.average_rate) if stream.average_rate else float(stream.guessed_rate)
            duration = float(stream.duration * time_base) if stream.duration else None
            
            loaded_frames = []
            
            for target_ts in timestamps:
                # Convert timestamp to pts (presentation timestamp)
                target_pts = int(target_ts / time_base)
                
                # Seek to the target timestamp (seek to keyframe before target)
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)
                
                closest_frame = None
                closest_ts_diff = float('inf')
                
                for frame in container.decode(video=0):
                    current_ts = float(frame.pts * time_base)
                    current_diff = abs(current_ts - target_ts)
                    
                    if current_diff < closest_ts_diff:
                        # Release the previous frame
                        if closest_frame is not None:
                            del closest_frame
                        closest_ts_diff = current_diff
                        closest_frame = frame
                    
                    # If we've passed the target and diff is increasing, stop
                    if current_ts > target_ts and current_diff > closest_ts_diff:
                        break
                    
                    # Also stop if we're significantly past the target (optimization)
                    if current_ts > target_ts + 1.0:
                        break
                
                if closest_frame is not None:
                    frame_data = closest_frame.to_ndarray(format="rgb24")
                    loaded_frames.append(frame_data)
                    del closest_frame
                else:
                    raise ValueError(f"Unable to find frame at timestamp {target_ts}")
            
            frames = np.array(loaded_frames)
            return frames
            
        finally:
            if container is not None:
                container.close()
                container = None
    
    elif video_backend == "torchvision_av":
        cached = _VIDEO_READER_CACHE.get(video_path)
        reader = cached["reader"] if cached is not None else None
        owned = reader is None  # 只有本次现 open 的 reader 才在 finally 关闭;复用缓存的留着
        loaded_frames = []
        loaded_ts = []

        try:
            if reader is None:
                torchvision.set_video_backend("pyav")
                reader = torchvision.io.VideoReader(video_path, "video")

            for target_ts in timestamps:
                # Reset reader state
                reader.seek(target_ts, keyframes_only=True)

                closest_frame = None
                closest_ts_diff = float('inf')

                for frame in reader:
                    current_ts = frame["pts"]
                    current_diff = abs(current_ts - target_ts)

                    if closest_frame is None:
                        closest_frame = frame

                    if current_diff < closest_ts_diff:
                        # Release the previous frame
                        if closest_frame is not None:
                            del closest_frame
                        closest_ts_diff = current_diff
                        closest_frame = frame
                    else:
                        # The time difference starts to increase, stop searching
                        break

                if closest_frame is not None:
                    frame_data = closest_frame["data"]
                    if isinstance(frame_data, torch.Tensor):
                        frame_data = frame_data.cpu().numpy()
                    loaded_frames.append(frame_data)
                    loaded_ts.append(closest_frame["pts"])

                    # Immediately release frame reference
                    del closest_frame

        finally:
            # 只关闭本次自建的 reader;缓存的 reader 留给后续 get_video 复用
            if owned and reader is not None:
                if hasattr(reader, '_c'):
                    reader._c = None
                if hasattr(reader, 'container'):
                    reader.container.close()
                    reader.container = None

        frames = np.array(loaded_frames)
        return frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError


def get_all_frames(
    video_path: str,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
    resize_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Get all frames from a video.
    Args:
        video_path (str): Path to the video file.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
        video_backend_kwargs (dict, optional): Keyword arguments for the video backend.
        resize_size (tuple[int, int], optional): Resize size for the frames. Defaults to None.
    """
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(range(len(vr))).asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        frames = decoder.get_frames_at(indices=range(len(decoder)))
        return frames.data.numpy(), frames.pts_seconds.numpy()
    elif video_backend == "pyav":
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format="rgb24")
            frames.append(frame)
        frames = np.array(frames)
    elif video_backend == "torchvision_av":
        # set backend and reader
        torchvision.set_video_backend("pyav")
        reader = torchvision.io.VideoReader(video_path, "video")
        frames = []
        for frame in reader:
            frames.append(frame["data"].numpy())
        frames = np.array(frames)
        frames = frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError(f"Video backend {video_backend} not implemented")
    # resize frames if specified
    if resize_size is not None:
        frames = [cv2.resize(frame, resize_size) for frame in frames]
        frames = np.array(frames)
    return frames