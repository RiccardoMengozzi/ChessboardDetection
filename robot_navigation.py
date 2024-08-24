import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os 



def computeDisparityMap(frame_L, frame_R, offset, num_disp, block_size):

    # Create StereoSGBM object
    left_matcher = cv2.StereoSGBM_create(
        minDisparity = offset,
        numDisparities = num_disp + offset,
        blockSize = block_size,
        P1 = 8 * 3 * block_size * block_size,
        P2 = 32 * 3 * block_size * block_size,
        disp12MaxDiff = 1,
        uniquenessRatio = 15,
        speckleWindowSize = 0,
        speckleRange = 2,
        preFilterCap = 63,
        mode = cv2.StereoSGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # divide by 16 to have true diaprity value
    displ = left_matcher.compute(frame_L, frame_R).astype(np.float32) / 16
    dispr = right_matcher.compute(frame_R, frame_L).astype(np.float32) / 16
    
    # Convert -1 values to 0 so that following computations with mean don't get influenced.
    displ = np.array(displ)
    displ[displ == -1] = 0
    dispr = np.array(dispr)
    dispr[dispr == -1] = 0

    return displ, dispr, left_matcher


def WlsFilter(left_matcher, displ, dispr, imgL, lmbda, sigma, visual_multiplier):

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left = left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    wls_map = wls_filter.filter(displ, imgL, None, dispr)

    return wls_map


def computeMainDisparity(disp_map, center, win_radius, mode):
    # window in which perform computations
    roi = disp_map[center[1] - win_radius : center[1] + win_radius, center[0] - win_radius : center[0] + win_radius]

    # mean
    if mode == 0:
        dmain = np.mean(roi)

    # most frequent value
    if mode == 1:
        hist, bins = np.histogram(roi.flatten(), 256, [0,256])
        dmain = np.argmax(hist)

    return dmain


def computeMultipleMainDisparities(disp_map, corners):
    # Isolate vertices of left and right regions
    left_corners = [corners[0], corners[7], corners[8], corners[15]]
    right_corners = [corners[32], corners[39], corners[40], corners[47]]    

    # Choose x and y coords of the vertices such as to have a proper rectangular region
    left_vertices = np.array([[np.max([left_corners[0][0], left_corners[1][0]]), np.min([left_corners[0][1], left_corners[2][1]])],
                                 [np.max([left_corners[0][0], left_corners[1][0]]), np.max([left_corners[1][1], left_corners[3][1]])],
                                 [np.min([left_corners[2][0], left_corners[3][0]]), np.min([left_corners[0][1], left_corners[2][1]])],
                                 [np.min([left_corners[2][0], left_corners[3][0]]), np.max([left_corners[1][1], left_corners[3][1]])]]).astype(int)

    right_vertices = np.array([[np.max([right_corners[0][0], right_corners[1][0]]), np.min([right_corners[0][1], right_corners[2][1]])],
                                  [np.max([right_corners[0][0], right_corners[1][0]]), np.max([right_corners[1][1], right_corners[3][1]])],
                                  [np.min([right_corners[2][0], right_corners[3][0]]), np.min([right_corners[0][1], right_corners[2][1]])],
                                  [np.min([right_corners[2][0], right_corners[3][0]]), np.max([right_corners[1][1], right_corners[3][1]])]]).astype(int)
    
    # Compute main disparities
    dmain_left = np.mean(disp_map[left_vertices[1][1]: left_vertices[0][1], left_vertices[0][0]: left_vertices[3][0]])
    dmain_right = np.mean(disp_map[right_vertices[1][1]: right_vertices[0][1], right_vertices[0][0]: right_vertices[3][0]])


    return dmain_left, dmain_right


def computeAngle(distance1, distance2, width):
    difference = np.abs(distance1 - distance2)
    angle = np.arcsin(difference / width)

    return angle

def computeDistance(dmain, focal, baseline):
    distance = focal * baseline / dmain / 1000
    return distance

def find_percentile_value(hist, percentile):
    s = 0
    idx = 0
    total_pixel = np.sum(hist)
    while(s < total_pixel*percentile/100):
        s += hist[idx]
        idx += 1
    return idx

def linear_stretching(img, max_value, min_value):
    img[img<min_value] = min_value
    img[img>max_value] = max_value
    linear_stretched_img = 255./(max_value-min_value)*(img-min_value)
    return linear_stretched_img

def stretch(img, hist, min_percentile, max_percentile):
    max_value = find_percentile_value(hist, max_percentile)
    min_value = find_percentile_value(hist, min_percentile)
    linear_stretched_img = linear_stretching(np.copy(img), max_value, min_value)

    return linear_stretched_img

def compute_histogram(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])

    return hist

def processImage(img, pattern_size, square_size, idx, side, verbose):

    #Building 3D points: not really squares!! width = 25mm, height =~ 25.43mm
    indices = np.indices(pattern_size, dtype = np.float32)
    indices[0] *= square_size[0]
    indices[1] *= square_size[1]
    pattern_points = np.zeros([pattern_size[0] * pattern_size[1], 3], np.float32)
    coords_3D = indices.T.reshape(-1, 2)
    pattern_points[:, :2] = coords_3D
    
    if img is None:
        if verbose:
            print("frame_" + side  + "{}".format(idx) + " is not and image")
        return None

    found, corners = cv2.findChessboardCorners(img, pattern_size)

    if found:
        #Refining corner position to subpixel iteratively until criteria  max_count=30 or criteria_eps_error=1 is sutisfied
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 5, 1)
        #Image Corners 
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
 
        
    if not found:
        if verbose:
            print("chessboard_" + side + "{}".format(idx) + " not found")
        vis = None
        return None
    
    return [corners.reshape(-1,2), pattern_points]


def getVertices(chessboard, pattern_size):
    vertices = []
    hh = pattern_size[0]
    ww = pattern_size[1]

    vertix_1 = chessboard[0][0]
    vertix_2 = chessboard[0][0 + hh - 1]
    vertix_3 = chessboard[0][hh * ww - hh]
    vertix_4 = chessboard[0][hh * ww - 1]
    vertices = [vertix_1, vertix_2, vertix_3, vertix_4]

    return vertices


def computeSizeInPixels(vertices, pattern_size, mode):
    # diagonal method
    if mode == 0:
        angle = np.arctan(pattern_size[0] / pattern_size[1])
        diagonal = np.linalg.norm(vertices[0] - vertices[3])
        size_pixels = [diagonal * np.cos(angle), diagonal * np.sin(angle)]

    # sides method
    if mode == 1:
        size_pixels = [np.linalg.norm(vertices[1]- vertices[3]), np.linalg.norm(vertices[2] - vertices[3])]


    return size_pixels

def computeSize(size_pixels, distance, focal):
    size = [distance * 1000 * size_pixels[0] / focal, distance * 1000 * size_pixels[1] / focal]
    return size



def computeCenter(chess):
    center = []
    x_coord = chess[0].T[0]
    y_coord = chess[0].T[1]
    center = (np.mean(x_coord), np.mean(y_coord))

    return np.array(center).astype(int)


def uniform_corners_order(chess, pattern_size):
    # define uniform order as "Left to Right and Down to Up"

    if chess[0][0][0] > chess[0][-1][0] and chess[0][0][1] < chess[0][pattern_size[0] - 1][1]:
        chess = (chess[0][::-1], chess[1])


    return chess

def drawSquare(img, ctr, length):
    # Calculate the coordinates for the square
    half_length = length // 2

    # Define the square's bounding box
    top_left = (ctr[0] - half_length, ctr[1] - half_length)
    bottom_right = (ctr[0] + half_length, ctr[1] + half_length)

    # Draw the square on the image
    color = (255, 255, 255)  # White color
    thickness = 2
    cv2.rectangle(img, top_left, bottom_right, color, thickness)

def main():

    # Get current folder directory
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # variable to keep track of current frame
    i = 0

    # variables to save previous value of chessboard corners
    previous_chessL = None
    previous_chessR = None
    
    # Initialize interface windows
    cv2.namedWindow("image")
    cv2.moveWindow("Image", 0, 0)

    # Size of plot window
    wind_width = 320
    wind_height = 240

    # Capture left video
    videoL = cv2.VideoCapture(dir_path + "/robotL.avi")
    videoL.set(cv2.CAP_PROP_FRAME_WIDTH, wind_width)
    videoL.set(cv2.CAP_PROP_FRAME_HEIGHT, wind_height)

    # Capture right video
    videoR = cv2.VideoCapture(dir_path + "/robotR.avi")
    videoR.set(cv2.CAP_PROP_FRAME_WIDTH, wind_width)
    videoR.set(cv2.CAP_PROP_FRAME_HEIGHT, wind_height)

    #---------- Plots arrays ----------
    dmain_plot = []
    dmain_mfv_plot = []
    distance_plot = []
    estimated_size_plot = []
    estimated_size_angled_plot = []
    error_sides_plot = []
    error_diag_plot = []
    error_mixed_plot = []
    disparity_range_plot = []
    error_angled_plot = []

    # --------- Some parameters ---------
    # offset: horiontal offset for disparity range
    # wind_radius: size of the window to compute the main disparity 

    offset = 0 
    wind_radius = 20 

    # is_copied_L/R: to keep track if the current chessboard has been copied from the previous one because it couldn't be detected
    is_copied_L = 0
    is_copied_R = 0

    focal = 567.2
    baseline = 92.226 

    # Stream left and right frames
    while videoL.isOpened() and videoR.isOpened():
        checkL, frameL = videoL.read()
        checkR, frameR = videoR.read()
        if not checkL or frameL is None:
            videoL.release()
            print("release left video resource")
            break
        if not checkR or frameR is None:
            videoR.release()
            print("released right video resource")
            break
    
        # Convert frames in to grayscale
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        # Stretch gray intensities to enhanche chessboard contrast
        min_percentile = 25
        max_percentile = 88
        stretchedL = stretch(grayL, compute_histogram(grayL), min_percentile, max_percentile).astype(np.uint8)
        stretchedR = stretch(grayR, compute_histogram(grayR), min_percentile, max_percentile).astype(np.uint8)

        # Find chessboard corners in both videos
        pattern_size = [8, 6]
        square_size = [178 / pattern_size[0] - 1, 125 / pattern_size[1] - 1]
        side_L = "L"
        side_R = "R"

        chessL = processImage(stretchedL, pattern_size, square_size, 0, side_L, verbose = 0)
        chessR = processImage(stretchedR, pattern_size, square_size, 0, side_R, verbose = 0)

        # If chessboard is not detected, keep the previous one
        if chessL is not None:
            previous_chessL = chessL
            is_copied_L = 0
        else:
            chessL = previous_chessL
            is_copied_L = 1


        if chessR is not None:
            previous_chessR = chessR
            is_copied_R = 0
        else:
            chessR = previous_chessR
            is_copied_R = 1


        # Crop frames around center to speed up program
        crop_value = 200
        cropL = grayL[crop_value:400, 0:640]
        cropR = grayR[crop_value:400, 0:640]

        # Correcting coordinates due to cropping (ONLY ONCE, if chessboard is copied from previous chessb then I don't correct)
        if is_copied_L == 0:
            chessL[0][:,1] -= crop_value       
        if is_copied_R == 0:
            chessR[0][:,1] -= crop_value

        # Compute center of chessboard through frames to keep track of it
        centerL = computeCenter(chessL)
        centerR = computeCenter(chessR)


        # Compute disparity map (with offset: disparity range is [offset, 64 + offset])
        max_disparity = 64
        block_size = 11
        displ, dispr, left_matcher = computeDisparityMap(cropL, cropR, offset, max_disparity, block_size)

        # Apply Wls filter on the disparity map
        displ_wls = WlsFilter(left_matcher, displ, dispr, cropL, lmbda = 2000, sigma = 1.0, visual_multiplier = 1.0)

        # Orient all the corners order in the same way, useful for later computations...
        oriented_chessL = uniform_corners_order(chessL, pattern_size)
   
        # Normalze disparity map for visualization (NOT REAL VALUES)
        displ_norm = cv2.normalize(displ, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        dispr_norm = cv2.normalize(dispr, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # compute main disparity of window centered in center
        # mode = 0:mean ; mode = 1: most frequent value
        wind_radius = 20
        dmain = computeMainDisparity(displ, centerL, wind_radius, 0)
        dmain_mfv = computeMainDisparity(displ, centerL, wind_radius, 1)

        # Compute the offset such that the previous value of dmain stays in the middle of the range [offset, 64 + offset]
        offset = (int)(dmain - max_disparity / 2)
        if offset < 0:
            offset = 0
                
        # Compute distance from center
        distance = computeDistance(dmain, focal, baseline)

        # Compute 4 vertices for size estimation
        vertices = getVertices(oriented_chessL, pattern_size)

        # Compute size piixel wise
        # diag: first compute the diagonal then use the angle for width and height
        # sides: directly compute width and height from the vertices
        size_diag_px = computeSizeInPixels(vertices, pattern_size, mode=0)
        size_sides_px = computeSizeInPixels(vertices, pattern_size, mode=1)

        # Compute size in mm
        size_diag = computeSize(size_diag_px, distance, focal)
        size_sides = computeSize(size_sides_px, distance, focal)

        # width is from "sides method", height is from "diag method"
        size_mixed = [size_sides[0], size_diag[1]]

        # Compute disparities in left and right parts of the object
        dmain_L, dmain_R = computeMultipleMainDisparities(displ, oriented_chessL[0])

        # Estimate the horizontal orientation (angle theta) of the object wrt the camera
        horizontal_angle = computeAngle(computeDistance(dmain_L, focal, baseline), computeDistance(dmain_R, focal, baseline), size_sides[0] / 1000)

        # Replace old width with new width corrected with angle cosine
        size_angled = [size_sides[0] * np.cos(horizontal_angle), size_sides[1]]

        # Compute error for the various methods
        real_size = np.array([125, 178])

        error_diag = np.round(np.abs(np.array(size_diag) - real_size) / real_size, 3)
        error_sides = np.round(np.abs(np.array(size_sides) - real_size) / real_size, 3)
        error_mixed = np.round(np.abs(np.array(size_mixed) - real_size) / real_size, 3)
        error_angled = np.round(np.abs(np.array(size_angled) - real_size) / real_size, 3)


        # ---------- PRINT OUTPUTS -----------

        # Visualize disparity map and region in which dmain is computed
        vis = displ_norm.copy()
        drawSquare(vis, centerL, wind_radius * 2)
        cv2.imshow("image", vis)
 
        # Output the distance
        if distance < 0.8:
            print(f"[FRAME_{i}] distance = {distance:.3f}m  OBJECT TOO CLOSE!!!")
        else:
            print(f"[FRAME_{i}] distance = {distance:.3f}m")


        # Append values for plots
        dmain_plot.append(dmain)
        dmain_mfv_plot.append(dmain_mfv)
        distance_plot.append(distance)
        estimated_size_plot.append(size_sides)
        estimated_size_angled_plot.append(size_angled)
        error_sides_plot.append(error_sides)
        error_diag_plot.append(error_diag)
        error_mixed_plot.append(error_mixed)
        error_angled_plot.append(error_angled)
        disparity_range_plot.append([offset, offset + max_disparity])

        # Increment frame number
        i += 1
        if cv2.waitKey(1) == ord('q'):
            break

    videoL.release()
    videoR.release()

    # Total Execution time
    print("--- %s seconds ---" % np.round((time.time() - start_time), 2))

    # Compute RMS error
    print()
    rms_diag = np.mean(np.sqrt(np.array(error_diag_plot) ** 2), axis = 0)
    rms_sides = np.mean(np.sqrt(np.array(error_sides_plot) ** 2), axis = 0)

    print("RMS error diagonal method = ", rms_diag)
    print("RMS error sides method = ", rms_sides)

    # ---------- PLOTS ------------

    x = np.linspace(0, 388, 389)

    # N for convolve used to smooth the plots, if N = 1 plots are not smoothed
    N = 1
    # plt.plot(np.convolve(dmain_plot, np.ones(N)/N, mode='valid'), label="dmain")
    plt.plot(np.convolve(np.array(estimated_size_plot)[:,0], np.ones(N)/N, mode='valid'), label = "estimated width" )
    plt.plot(np.convolve(np.array(estimated_size_plot)[:,1], np.ones(N)/N, mode='valid'), label = "estimated height")
    plt.axhline(y=178, color='r', linestyle='-', label = "Real height")
    plt.axhline(y=125, color='r', linestyle='-', label = "Real width")
    
    plt.legend()
    plt.title("Estimated size (mm)")
    plt.xlabel("frames")
    plt.ylabel("size (mm)")
    plt.show()
    plt.pause(3)
    plt.close()
    exit()

if __name__ == "__main__":
    # Start timer
    start_time = time.time()
    main()

