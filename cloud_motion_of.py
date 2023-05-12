import cv2
import itertools
import numpy as np

def main():

    video_path = '../videos/27a93142-2a9a-4c77-80bf-85d860196208.mkv'
    # video_path = '../videos/9d569219-7c26-481f-b86b-adb4d79c83b9.mkv' # fast layered
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (800, 800))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Circular region of interest
    center_x, center_y = 400, 400
    radius = 400

    # Create a circular mask
    mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # Buffer for storing flow vectors
    flow_buffer = []
    buffer_length = 10

    frame_counter = 0

    avg_flow = np.zeros((800, 800, 2), dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        frame = cv2.resize(frame, (800, 800))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow every second
        if frame_counter % 25 == 0:
            masked_gray = cv2.bitwise_and(gray, mask)
            masked_prev_gray = cv2.bitwise_and(prev_gray, mask)

            flow = cv2.calcOpticalFlowFarneback(masked_prev_gray, masked_gray, None, 0.5, 3, 5, 3, 5, 1.1, 0)
            prev_gray = gray.copy()

            flow_buffer.append(flow)
            if len(flow_buffer) > buffer_length:
                flow_buffer.pop(0)

            avg_flow = np.mean(flow_buffer, axis=0)

        # Draw the flow vectors
        step = 64
        for y_coord, x_coord in itertools.product(range(center_y - radius, center_y + radius, step), range(center_x - radius, center_x + radius, step)):
            if (x_coord - center_x) ** 2 + (y_coord - center_y) ** 2 <= radius ** 2:
                fx, fy = avg_flow[y_coord, x_coord]
                cv2.arrowedLine(frame, (x_coord, y_coord), (int(x_coord + fx * 3), int(y_coord + fy * 3)), (0, 255, 0), 1, tipLength=0.3)

        # Draw the region of interest
        # cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
