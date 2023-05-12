import cv2
import itertools
import numpy as np
import requests
import pandas as pd
import math

def vectorMagnitudeDirection(cmv_x, cmv_y):
    vec_mag = np.sqrt(cmv_x*cmv_x + cmv_y*cmv_y)
    vec_dir = (95+np.rad2deg(np.arctan2(cmv_y,-cmv_x))) % 360
    # vec_dir = (180-np.rad2deg(np.arctan2(cmv_y,cmv_x))) % 360
    
    return vec_mag, vec_dir

def draw_wind_direction(frame, winddir, length=50, color=(0, 0, 255), thickness=2):
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    # Convert wind direction to radians
    winddir_rad = math.radians(winddir + 7)

    # Calculate the end point of the vector
    end_x = int(center_x + length * math.sin(winddir_rad))
    end_y = int(center_y - length * math.cos(winddir_rad))

    # Mirror the x-coordinate
    mirrored_end_x = center_x - (end_x - center_x)

    # Draw the vector
    cv2.arrowedLine(frame, (center_x, center_y), (mirrored_end_x, end_y), color, thickness)


def main():

    video_path = '../videos/27a93142-2a9a-4c77-80bf-85d860196208.mkv' # 1
    video_path = '../videos/9d569219-7c26-481f-b86b-adb4d79c83b9.mkv' # 2
    video_path = '../videos/1af3523f-99ef-4132-8d07-2833977d7caa.mkv' # 3
    # video_path = '../videos/2cd44331-d19f-488a-b187-9391e3799723.mkv' # 4

    # Weather data info
    api_key = '' # enter your key

    start_date = '2021-08-24T12:30:00'     # date format yyyy-MM-dd or yyyy-MM-ddTHH:mm:ss (will be rounded to the closest hour)
    end_date = '2021-08-24T12:30:00'

    location = 'Vienna'
    lat = 48.18020277777777
    lon = 16.342366666666667
    unitGroup = 'metric' # Supported values are us, uk, metric, base - default US

    url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'+location+'/'+start_date+'/'+end_date+'?key='+api_key+'&unitGroup='+unitGroup+'&include=alerts'
    data = requests.get(url)
    data = data.json()
    data_df = pd.DataFrame.from_dict(data['days']).iloc[0]
    winddir = data_df['winddir'] 
    print(winddir)

    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (800, 800))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(prev_frame)
    hsv[...,1] = 255

    # Circular region of interest
    center_x, center_y = 400, 430
    radius = 375

    # Create a circular mask
    mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # Buffer for storing flow vectors
    flow_buffer = []
    buffer_length = 10

    frame_counter = 0

    avg_flow = np.zeros((800, 800, 2), dtype=np.float32)
    vel_factor = 60/1
    dir_mean = 0

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

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mag[np.isinf(mag) | np.isnan(mag)] = 0
            hue_offset = 90  # Adjust the offset value to shift the colors
            hsv[...,0] = (ang*180/np.pi/2 + hue_offset) % 180
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            alpha = 0.7 # Adjust the blending factor (0 <= alpha <= 1)
            blended = cv2.addWeighted(frame, alpha, bgr, 1 - alpha, 0)          

            # print("Magnitude - Min:", np.min(mag), "Max:", np.max(mag), "Mean:", np.mean(mag))
            # print("Angle - Min:", np.min(ang), "Max:", np.max(ang), "Mean:", np.mean(ang))
            cv2.imshow('bgr',bgr)
            # cv2.imshow("Frame", blended)

            flow_buffer.append(flow)
            if len(flow_buffer) > buffer_length:
                flow_buffer.pop(0)

            avg_flow = np.mean(flow_buffer, axis=0)

            # Computes the magnitude and angle of the 2D vectors
            flow= np.round(flow, decimals=0)

            flow_u = np.ma.masked_equal(flow[..., 0], 0)
            flow_v = np.ma.masked_equal(flow[..., 1], 0)
            flow_u = np.ma.masked_where(np.ma.getmask(flow_v), flow_u)
            flow_v = np.ma.masked_where(np.ma.getmask(flow_u), flow_v)
            mag_mean, dir_mean = vectorMagnitudeDirection(flow_u.mean(),
                                                        flow_v.mean())
            mag_mean_minute = mag_mean * vel_factor
            print(f"Pixels per minute: {str(mag_mean_minute)}")
            print(f"Degrees: {str(dir_mean)}  (indicates the direction from where cloud is moving from)")



        # Draw the flow vectors
        step = 32
        for y_coord, x_coord in itertools.product(range(center_y - radius, center_y + radius, step), range(center_x - radius, center_x + radius, step)):
            if (x_coord - center_x) ** 2 + (y_coord - center_y) ** 2 <= radius ** 2:
                fx, fy = avg_flow[y_coord, x_coord]
                cv2.arrowedLine(frame, (x_coord, y_coord), (int(x_coord + fx * 3), int(y_coord + fy * 3)), (0, 255, 0), 1, tipLength=0.3)

        # Draw the region of interest
        # cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 1)
        draw_wind_direction(frame, dir_mean, length=50, color=(0, 255, 0))
        draw_wind_direction(frame, winddir)
        cv2.imshow("Framevec", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        elif ord('s') == 0xFF:
            cv2.imwrite('opticalfb.png',frame)
            cv2.imwrite('opticalhsv.png',bgr)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
