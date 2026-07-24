import cv2
import sys	
sys.path.append("/usr/local/zed/samples/body tracking/python") #Where ogl_viewer is stored in system
import time
import pyzed.sl as sl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import mediapipe as mp
from datetime import datetime
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

    
hands = mp_hands.Hands(

    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )   

last_hand_results = None   


def process_hand_crop(image, wrist, elbow=None, depth=None):
    
    # Estimate hand centre
    cx = int(wrist[0])
    cy = int(wrist[1])

    hand_scale = 224  # fallback size

    if elbow is not None:
        # Elbow -> wrist vector
        vx = wrist[0] - elbow[0]
        vy = wrist[1] - elbow[1]

        arm_length = np.sqrt(vx*vx + vy*vy)

        # Move crop centre towards hand
        cx = int(wrist[0] + 0.55 * vx)
        cy = int(wrist[1] + 0.55 * vy)

        # Estimate crop size from arm length
        # Hand is roughly 35-50% of forearm length
        hand_scale = int(arm_length * 0.8)


    if depth is not None and depth > 0:

        ref_depth = 1.2  # metres

        scale = ref_depth / depth

        # Prevent the crop becoming ridiculously large or tiny
        scale = np.clip(scale, 0.5, 2.0)

        hand_scale = int(hand_scale * scale)


    margin = 1.35
    crop_size = int(hand_scale * margin)
    crop_size = max(128, min(crop_size, 420))

    half = crop_size // 2


    x1 = max(0, cx-half)
    y1 = max(0, cy-half)

    x2 = min(image.shape[1], cx+half)
    y2 = min(image.shape[0], cy+half)


    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None


    rgb_crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGRA2RGB
    )

    results = hands.process(rgb_crop)

    return results, (x1, y1, crop)
    
    

def get_xyz(point_cloud, x, y):

    err, value = point_cloud.get_value(int(x), int(y))

    if err != sl.ERROR_CODE.SUCCESS:
        return None

    X, Y, Z, A = value

    if (not np.isfinite(X) or
        not np.isfinite(Y) or
        not np.isfinite(Z)):
        return None

    return np.array([X, Y, Z])




    
if __name__ == "__main__":
    print("Running Body Tracking sample ... Press 'q' to quit")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    # If .svo file is inputted as an argument, use this rather than live feed
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.svo_real_time_mode = True
        init_params.set_from_svo_file(filepath)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    #Camera is static, so using set_as_static to improve tracking
    positional_tracking_parameters.set_as_static = True
    positional_tracking_parameters.enable_imu_fusion = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True            # Smooth skeleton move
    obj_param.enable_tracking = True                # Track people across images flow
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_ACCURATE
    obj_param.body_format = sl.BODY_FORMAT.POSE_18  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_object_detection(obj_param)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 75

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    #display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
     #                                  min(camera_info.camera_resolution.height, 720))
    display_resolution = sl.Resolution(960, 540)
    
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
                 , display_resolution.height / camera_info.camera_resolution.height]

    # Create OpenGL viewer
    

    # Create ZED objects filled in the main loop
    bodies = sl.Objects()
    image = sl.Mat()
    point_cloud = sl.Mat()

    frame_count = 0
    

    
    while True:
    
        # Grab an image  
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(
                image,
                sl.VIEW.LEFT,
                sl.MEM.CPU,
                display_resolution
            )

            zed.retrieve_measure(
                point_cloud,
                sl.MEASURE.XYZ,
                sl.MEM.CPU,
                display_resolution
            )

            zed.retrieve_objects(
                bodies,
                obj_runtime_param
            )
                
            
            # Update OCV view
            image_left_ocv = image.get_data()
            
            #Running hand detection every n frames to try and increase frame rate
            
            n = 1
            if frame_count % n == 0:
            
                hand_results = []
            
                if len(bodies.object_list) > 0:

                    body = bodies.object_list[0]
                    kp = body.keypoint_2d.copy()
                    
                    person = {
                        "id": body.id,
                        "body": [],
                        "left_hand": None,
                        "right_hand": None
                    }




                    # Scale to match display image
                    kp[:,0] *= image_scale[0]
                    kp[:,1] *= image_scale[1]
                    
                    
                    for joint in kp:

                        xyz = get_xyz(
                            point_cloud,
                            joint[0],
                            joint[1]
                        )

                        person["body"].append(xyz)
                    
                   
                    # Right side
                    right_wrist = kp[4]
                    right_elbow = kp[3]

                    # Left side
                    left_wrist = kp[7]
                    left_elbow = kp[6]


                    for side, wrist, elbow in [
                        ("right_hand", right_wrist, right_elbow),
                        ("left_hand", left_wrist, left_elbow)
                    ]:

                        wrist_xyz = get_xyz(
                            point_cloud,
                            wrist[0],
                            wrist[1]
                        )

                        depth = None

                        if wrist_xyz is not None:
                            depth = wrist_xyz[2]

                        result, data = process_hand_crop(
                            image_left_ocv,
                            wrist,
                            elbow,
                            depth
                        )

                        if result is not None:
                            hand_results.append(
                                (side, result, data)
                            )
           
            frame_count += 1

            
            #Convert for OpenCV drawing
            image_bgr = cv2.cvtColor(image_left_ocv, cv2.COLOR_BGRA2BGR)
            
            #Draw ZED body skeleton
            cv_viewer.render_2D(
                    image_bgr,
                    image_scale,
                    bodies.object_list, 
                    obj_param.enable_tracking, 
                    obj_param.body_format)
            
            #Draw MediaPipe hands
            
                            
            for side, results, data in hand_results:

                if results.multi_hand_landmarks:

                    x_offset, y_offset, crop = data

                    for hand in results.multi_hand_landmarks:

                        hand_xyz = []

                        for lm in hand.landmark:

                            x = int(lm.x * crop.shape[1]) + x_offset
                            y = int(lm.y * crop.shape[0]) + y_offset

                            xyz = get_xyz(point_cloud, x, y)
                            hand_xyz.append(xyz)


                            cv2.circle(
                                image_bgr,
                                (x, y),
                                3,
                                (0,255,0),
                                -1
                            )
                        person[side] = hand_xyz

                            
                                        
            # show both cropped images                    
            for i,(side,results,data) in enumerate(hand_results):

                crop = data[2]

                cv2.imshow(
                    f"Hand Crop {i}",
                    cv2.cvtColor(
                        crop,
                        cv2.COLOR_BGRA2BGR
                    )
                )
                             
            cv2.imshow("ZED | 2D View", image_bgr)

        key = cv2.waitKey(10)


        if key == ord('q'):
           print("\n")
           break
                
                
    
    # Disable modules and close camera
    
    image.free(sl.MEM.CPU)
    point_cloud.free(sl.MEM.CPU)
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()
