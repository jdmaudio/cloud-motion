import itertools
import numpy as np
from scipy import ndimage
import cv2

def flowVectorSplit(array1, array2, info):

    nblock = info['nblock']
 
    
    array1_split = split2DArray(array1, nblock)
    array2_split = split2DArray(array2, nblock)
    
    cmv_x = np.zeros(nblock*nblock)
    cmv_y = np.zeros(nblock*nblock)
    for i in range(nblock*nblock):
        cmv_x[i], cmv_y[i] = fftFlowVector(array1_split[i], array2_split[i])

        
    cmv_x, cmv_y = rmLargeMagnitudes(cmv_x, cmv_y, v_max=info['v_max'])
    
    cmv_x = cmv_x.reshape([nblock, nblock])
    cmv_y = cmv_y.reshape([nblock, nblock])
    
    cmv_x, cmv_y, nmf_u, nmf_v = rmSpuriousVectors(cmv_x, cmv_y, info)
    
    cmv_x, cmv_y = flipVectors(cmv_x, cmv_y)
    cmv_x = -cmv_x
    cmv_y = -cmv_y

    return cmv_x, cmv_y, nmf_u, nmf_v


def flipVectors(cmv_x, cmv_y):
    cmv_x_correct = np.flip(cmv_x, axis=0)
    cmv_y_correct = np.flip(-cmv_y, axis=0)

    return cmv_x_correct, cmv_y_correct


def split2DArray(arr2d, nblock):
    split_0 = np.array_split(arr2d, nblock, axis=0)
    split_arr=[]
    for arr in split_0:
        split_01 = np.array_split(arr, nblock, axis=1)
        split_arr += split_01
    
    return split_arr


def fftFlowVector(im1, im2, global_shift=True):
    if not global_shift and (np.max(im1) == 0 or np.max(im2) == 0):
        return None

    crosscov = fftCrossCov(im1, im2)
    sigma = 3
    cov_smooth = ndimage.filters.gaussian_filter(crosscov, sigma)
    dims = np.array(im1.shape)

    pshift = np.argwhere(cov_smooth == np.max(cov_smooth))[0]
    
    rs = np.ceil(dims[0]/2).astype('int')
    cs = np.ceil(dims[1]/2).astype('int')

    # Calculate shift relative to center - see fft_shift.
    pshift = pshift - (dims - [rs, cs])
    return pshift


def fftCrossCov(im1, im2):
    fft1_conj = np.conj(np.fft.fft2(im1))
    fft2 = np.fft.fft2(im2)
    normalize = abs(fft2 * fft1_conj)
    try:  min_value = normalize[(normalize > 0)].min()
    except ValueError:  #raised if empty.
        min_value=0.01
    normalize[normalize == 0] = min_value  # prevent divide by zero error
    cross_power_spectrum = (fft2 * fft1_conj)/normalize
    crosscov = np.fft.ifft2(cross_power_spectrum)
    crosscov = np.real(crosscov)
    return motionVector(crosscov)


def motionVector(fft_mat):

    if type(fft_mat) is np.ndarray:
        rs = np.ceil(fft_mat.shape[0]/2).astype('int')
        cs = np.ceil(fft_mat.shape[1]/2).astype('int')
        quad1 = fft_mat[:rs, :cs]
        quad2 = fft_mat[:rs, cs:]
        quad3 = fft_mat[rs:, cs:]
        quad4 = fft_mat[rs:, :cs]
        centered_t = np.concatenate((quad4, quad1), axis=0)
        centered_b = np.concatenate((quad3, quad2), axis=0)
        return np.concatenate((centered_b, centered_t), axis=1)
    else:
        print('input to motionVector() should be a matrix')
        return

def rmLargeMagnitudes(cmv_x, cmv_y, v_max):
    
    vmag, vdir = vectorMagnitudeDirection(cmv_x, cmv_y) 
    cmv_x[np.where(abs(vmag)>v_max)]=0
    cmv_y[np.where(abs(vmag)>v_max)]=0
    
    return cmv_x, cmv_y



def rmSpuriousVectors(cmv_x, cmv_y, info):

    norm_fluct_u, norm_fluct_v  = getNormMedianFluctuation(cmv_x, cmv_y, info)
    norm_fluct_mag = np.sqrt(norm_fluct_u**2 + norm_fluct_v**2)
    error_flag = norm_fluct_mag > info['WS05-error_thres']
    cmv_x[np.where(error_flag)] = 0
    cmv_y[np.where(error_flag)] = 0
    
    return cmv_x, cmv_y, norm_fluct_u, norm_fluct_v


def vectorMagnitudeDirection(cmv_x, cmv_y, std_fact=1):

    vec_mag = np.sqrt(cmv_x*cmv_x + cmv_y*cmv_y)
    vec_dir = np.rad2deg(np.arctan2(cmv_y,cmv_x)) % 360
    
    return vec_mag, vec_dir

def getNormMedianFluctuation(u, v, info):

    d = info['WS05-neighborhood_dist']
    eps = info['WS05-eps'] 
    
    norm_fluctuation_u = normFluctuation(u, d, eps)
    norm_fluctuation_v = normFluctuation(v, d, eps)
    
    return norm_fluctuation_u, norm_fluctuation_v


def normFluctuation(vel_comp, d, eps): 
    v_shape = vel_comp.shape   
    norm_fluctuation = np.zeros(v_shape)
    norm_fluctuation[:] = np.NaN
    
    for i in range(d, v_shape[0]-d):
        for j in range(d, v_shape[1]-d):
            neighborhood = vel_comp[i-d:i+d+1, j-d:j+d+1]
            
            #remove central point
            neighborhood = neighborhood.flatten()
            mid_point = int(np.floor(neighborhood.size/2))
            neighborhood = np.delete(neighborhood, mid_point)
            
            neigh_median = np.median(neighborhood)
            
            fluctuation = vel_comp[i, j]-neigh_median
            
            residue = neighborhood-neigh_median
            residue_median = np.median(np.abs(residue))

            norm_fluctuation[i, j] = fluctuation/(residue_median+eps)
            
    return norm_fluctuation

def meanCMV(u, v):
    u_mean = 0 if (np.all(u==0)) else u[(np.abs(u) > 0) | (np.abs(v) > 0)].mean()
    v_mean = 0 if (np.all(v==0)) else v[(np.abs(u) > 0) | (np.abs(v) > 0)].mean()
    return u_mean, v_mean

def videoCropInfo(video_cap, nblock, block_len):
    frame_width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    return cropMarginInfo(frame_height, frame_width, nblock, block_len)


def cropMarginInfo(frame_height, frame_width, nblock, block_len):
    crop_len = block_len * nblock
    if(crop_len >= min([frame_height, frame_width])):
        exit("Error: The original frame size is smaller than \
             the provided crop-dimensions.")
    cent_x = int(frame_width/2)
    cent_y = int(frame_height/2)
    
    #crop a square region of interest to accomodate 
    y1 = int(cent_y - crop_len/2)
    y2 = int(cent_y + crop_len/2)
    x1 = int(cent_x - crop_len/2)
    x2 = int(cent_x + crop_len/2)
    
    #compute approximate central points of each block
    mid_loc = np.arange((block_len/2) - 1, nblock * block_len, block_len)
    mid_loc= mid_loc.astype('int32')
    return dict(frame_width=frame_width, frame_height=frame_height, 
                x1=x1, x2=x2, y1=y1, y2=y2, cent_x=cent_x, cent_y=cent_y, 
                block_mid=mid_loc, nblock=nblock, block_len=block_len)

def crop_frame(frame, x, y, width, height):
    return frame[y:y+height, x:x+width]

def main():    

    video_path = '../videos/27a93142-2a9a-4c77-80bf-85d860196208.mkv'
    # video_path = '../videos/9d569219-7c26-481f-b86b-adb4d79c83b9.mkv' # fast layered

    nblock = 4
    block_len = 15

    cap = cv2.VideoCapture(video_path)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Crop parameters: x, y, width, and height
    x, y, width, height = 300, 300, 450, 450

    

    inf = cropMarginInfo(600, 600, nblock, block_len)
    new_inf = {'channel': 2, 'v_max': int(np.ceil(block_len/3)), 'nblock': nblock, 'WS05-neighborhood_dist': 1, 'WS05-eps': 0.2, 'WS05-error_thres': 6, 'fleap': 100}
    inf.update(new_inf)

    fcount = 0
    first_frame = True

    cmv_x = np.zeros((nblock, nblock))  # Initialize cmv_x with zeros
    cmv_y = np.zeros((nblock, nblock))  # Initialize cmv_y with zeros

    vector_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fcount += 1
        frame = cv2.resize(frame, (800, 800))

        cropped_frame = crop_frame(frame, x, y, width, height)

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        if fcount == 1 or fcount % inf['fleap'] == 0:
            sky_new = blurred_frame[inf['y1']:inf['y2'], inf['x1']:inf['x2']]
            if np.any(inf["channel"] == np.array([0, 1, 2])):
                sky_new = frame[inf['y1']:inf['y2'], inf['x1']:inf['x2'], inf['channel']]
            elif inf["channel"] == 8:
                sky_new = frame[inf['y1']:inf['y2'], inf['x1']:inf['x2'], :]
                sky_new = (sky_new[:, :, 2]+0.5)/(sky_new[:, :, 0]+0.5)
            elif inf["channel"] == 9:
                sky_new = frame[inf['y1']:inf['y2'], inf['x1']:inf['x2'], :]
                sky_new = cv2.cvtColor(sky_new, cv2.COLOR_BGR2GRAY)
          
            #Store the sky data for first the frame as .
            if first_frame:
                sky_curr = sky_new
                first_frame = False
                continue
    
            #move one frame forward
            sky_prev = sky_curr
            sky_curr = sky_new
    
            cmv_x, cmv_y, nmf_x, nmf_y = flowVectorSplit(sky_prev, sky_curr, inf)
            u_mean,  v_mean = meanCMV(cmv_x, cmv_y)
            # print(u_mean)

            # Draw the vectors on the cropped_frame
            vector_frame = cropped_frame.copy()
            block_size = cropped_frame.shape[0] // nblock
            for i, j in itertools.product(range(nblock), range(nblock)):
                start_point = (j * block_size + block_size // 2, i * block_size + block_size // 2)
                end_point = (int(start_point[0] + cmv_y[i, j]), int(start_point[1] + cmv_x[i, j]))
                vector_frame = cv2.arrowedLine(vector_frame, start_point, end_point, (0, 0, 255), 2, tipLength=0.3)
        elif vector_frame is not None:
            vector_frame = cropped_frame.copy()  # Update the vector_frame with the current cropped_frame

            # Redraw the vectors on the updated vector_frame
            block_size = cropped_frame.shape[0] // nblock
            for i, j in itertools.product(range(nblock), range(nblock)):
                start_point = (j * block_size + block_size // 2, i * block_size + block_size // 2)
                end_point = (int(start_point[0] + cmv_y[i, j]), int(start_point[1] + cmv_x[i, j]))  # Corrected end_point calculation
                vector_frame = cv2.arrowedLine(vector_frame, start_point, end_point, (0, 0, 255), 2, tipLength=0.3)

        cv2.imshow("Cropped Frame with Vectors", vector_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
