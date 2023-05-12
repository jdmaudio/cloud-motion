import cv2
import numpy as np

# https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html

def cross_correlation(img1, img2, window_size, search_range=100):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1_gray = cv2.GaussianBlur(img1_gray, (window_size, window_size), 0)
    img2_gray = cv2.GaussianBlur(img2_gray, (window_size, window_size), 0)

    h, w = img1_gray.shape

    template = img1_gray[search_range:h - search_range, search_range:w - search_range]
    search_image = img2_gray

    # cv2.imshow("img1_gray", img1_gray)
    # cv2.imshow("img2_gray", img2_gray)

    result = cv2.matchTemplate(search_image, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # cv2.imshow("result", result)

    return max_loc[0] - search_range, max_loc[1] - search_range

def crop_frame(frame, x, y, width, height):
    return frame[y:y+height, x:x+width]


def main():
    video_path = '../videos/27a93142-2a9a-4c77-80bf-85d860196208.mkv'
    # video_path = '../videos/22ae4ec0-00cf-44bc-a4e1-f51d1544d43c.mkv'
    # video_path = '../videos/22ae4ec0-00cf-44bc-a4e1-f51d1544d43c.mkv'   # slow
    # video_path = '../videos/9d569219-7c26-481f-b86b-adb4d79c83b9.mkv' # fast layered
    # video_path = '../videos/7f6945b5-bd4f-4e65-becd-ee339e8f02cd.mkv' # layered
    # video_path = '../videos/7b712cfa-ce54-4d0b-aa3e-fa7590cf352e.mkv' # no motion
    # video_path = '../videos/6aeed808-3465-4109-b8a7-fad9c6f8524a.mkv' # slow
    # video_path = '../videos/4b65ac75-4761-41ae-a150-226e77858aeb.mkv' 
    window_size = 21

    # Crop parameters: x, y, width, and height
    x, y, width, height = 300, 300, 450, 450

    cap = cv2.VideoCapture(video_path)
    frame_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        frame = cv2.resize(frame, (1024, 1024))
        cropped_frame = crop_frame(frame, x, y, width, height)
        frame_buffer.append(cropped_frame)

        if len(frame_buffer) > 75:
            prev_frame = frame_buffer.pop(0)
            dx, dy = cross_correlation(prev_frame, cropped_frame, window_size)
            motion_str = f"Motion: dx={dx}, dy={dy}"
            cv2.putText(frame, motion_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw the motion vector
            start_point = (x + width // 2, y + height // 2)
            end_point = (start_point[0] + dx*2, start_point[1] + dy*2)
            cv2.arrowedLine(frame, start_point, end_point, (0, 0, 125), 2, tipLength=0.2)

        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 1)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(15) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()