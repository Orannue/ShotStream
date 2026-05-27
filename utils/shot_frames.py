def nearest_4n_plus_1(frame_count):
    """Return the closest positive frame count compatible with 4n+1 decoding."""
    frame_count = int(frame_count)
    if frame_count <= 1:
        return 1
    n = (frame_count - 1 + 2) // 4
    return 4 * n + 1


def latent_frames_for_video_frames(frame_count):
    """Map a target 4n+1 video-frame count to Wan latent frames."""
    frame_count = nearest_4n_plus_1(frame_count)
    return ((frame_count - 1) // 4) + 1


def video_frames_for_latent_frames(latent_frame_count):
    return 4 * (int(latent_frame_count) - 1) + 1


def frame_ranges_from_counts(frame_counts):
    ranges = []
    start = 0
    for frame_count in frame_counts:
        end = start + int(frame_count)
        ranges.append([start, end])
        start = end
    return ranges


def frame_counts_from_ranges(frame_ranges):
    return [int(end) - int(start) for start, end in frame_ranges]
