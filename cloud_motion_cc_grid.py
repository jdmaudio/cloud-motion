import cv2
import itertools
import numpy as np

# https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html

def cross_correlation(img1, img2, window_size, search_range=20):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1_gray = cv2.GaussianBlur(img1_gray, (window_size, window_size), 0)
    img2_gray = cv2.GaussianBlur(img2_gray, (window_size, window_size), 0)

    h, w = img1_gray.shape

    template = img1_gray[search_range:h - search_range, search_range:w - search_range]
    search_image = img2_gray

    result = cv2.matchTemplate(search_image, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    return max_loc[0] - search_range, max_loc[1] - search_range

def crop_frame(frame, x, y, width, height):
    return frame[y:y+height, x:x+width]


def main():
    video_path = '../videos/22ae4ec0-00cf-44bc-a4e1-f51d1544d43c.mkv'
    # video_path = '../videos/7f6945b5-bd4f-4e65-becd-ee339e8f02cd.mkv'
    # video_path = '../videos/27a93142-2a9a-4c77-80bf-85d860196208.mkv'
    window_size = 21

    # Crop parameters: x, y, width, and height
    x, y, width, height = 300, 300, 450, 450

    cap = cv2.VideoCapture(video_path)
    frame_buffer = []

    block_size_x = width // 4
    block_size_y = height // 4

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (1024, 1024))
        cropped_frame = crop_frame(frame, x, y, width, height)
        frame_buffer.append(cropped_frame)

        if len(frame_buffer) > 75:
            prev_frame = frame_buffer.pop(0)

            for i, j in itertools.product(range(4), range(4)):
                block_x = x + j * block_size_x
                block_y = y + i * block_size_y

                prev_block = crop_frame(prev_frame, block_x - x, block_y - y, block_size_x, block_size_y)
                curr_block = crop_frame(cropped_frame, block_x - x, block_y - y, block_size_x, block_size_y)

                dx, dy = cross_correlation(prev_block, curr_block, window_size)

                start_point = (block_x + block_size_x // 2, block_y + block_size_y // 2)
                end_point = (start_point[0] + dx, start_point[1] + dy)
                cv2.arrowedLine(frame, start_point, end_point, (0, 0, 255), 1, tipLength=0.3)

        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)

        for i in range(3):
            cv2.line(frame, (x + (i + 1) * block_size_x, y), (x + (i + 1) * block_size_x, y + height), (255, 255, 0), 1)
            cv2.line(frame, (x, y + (i + 1) * block_size_y), (x + width, y + (i + 1) * block_size_y), (255, 255, 0), 1)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(15) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
