import cv2
import numpy as np
import os

def save_least_movement_frames(video_path, output_folder, interval=120, min_distance=100):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_diffs = []
    prev_frame = None
    last_saved_frame_index = -min_distance  # Initialize to enable saving the first frame
    block_start_frame = 0  # Start frame of the current block
    frame_id = 2176  # Initialize frame ID for naming

    while True:
        # Move to the start frame of the next block
        cap.set(cv2.CAP_PROP_POS_FRAMES, block_start_frame)
        local_frame_count = 0  # Counter for frames within the current block

        while local_frame_count < interval:
            ret, frame = cap.read()
            if not ret:
                # End of video
                cap.release()
                print("Done processing and saving frames.")
                return

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate frame difference for frames after the first one in the block
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                frame_diff = np.sum(diff)
                frame_diffs.append((block_start_frame + local_frame_count, frame_diff, gray))

            prev_frame = gray
            local_frame_count += 1

        # Evaluate the block
        if frame_diffs:
            # Sort frames by movement, least to most
            sorted_frames = sorted(frame_diffs, key=lambda x: x[1])
            for frame in sorted_frames:
                frame_index, _, least_movement_gray_frame = frame
                # Ensure the frame is at least min_distance away from the last saved frame
                if frame_index - last_saved_frame_index >= min_distance:
                    frame_id += 1  # Increment frame ID for each saved frame
                    # Save the frame with new ID formatting
                    cv2.imwrite(os.path.join(output_folder, f"Id{frame_id:02}.png"), least_movement_gray_frame)
                    last_saved_frame_index = frame_index
                    break  # Break after saving to ensure only one frame is saved per interval

            # Clear the list for the next block
            frame_diffs = []

        # Prepare for the next block
        block_start_frame += interval
        prev_frame = None  # Reset prev_frame to avoid comparing frames across blocks

# Example usage
video_path = "input/c4/second/prod07.mp4"
output_folder = "saved_frames"
save_least_movement_frames(video_path, output_folder)
