import cv2
import sys	
sys.path.append("/usr/local/zed/samples/body tracking/python") #Where ogl_viewer is stored in system
import pyzed.sl as sl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def unit_vector(vector):
    """Returns unit vector of the vector"""
    return vector / np.linalg.norm(vector)
    
def angle_between(v1, v2):
    """Returns angle in degrees between the input vectors 'v1' and 'v2'"""
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
     
    #Using np.clip to keep results within -1.0, 1.0
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    return angle_rad * 180/np.pi
    
    
hands = mp_hands.Hands(

    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )   

    
if __name__ == "__main__":
    print("Running Body Tracking sample ... Press 'q' to quit")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE #NEURAL
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
    s
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

    frame_count = 0
    
    while True:
    
        # Grab an image  
        t0 = time.time() 
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            print("Grab:", time.time()-t0)
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            
            # Retrieve objects
            t0 = time.time()
            zed.retrieve_objects(bodies, obj_runtime_param)
            print("Bodies:", time.time()-t0)
		
            
            
                
            
            # Update OCV view
            image_left_ocv = image.get_data()
            
            #Running hand detection every 2 frames to increase frame rate
            

            if frame_count % 1 == 0:
                #Convert BGRA to RGB for MediaPipe
                rgb = cv2.cvtColor(image_left_ocv, cv2.COLOR_BGRA2RGB)
                t0 = time.time()
                last_hand_results = hands.process(rgb)
                print("Hands:", time.time()-t0)
            
           
            frame_count += 1
            results = last_hand_results
            
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
            
            if results.multi_hand_landmarks:
            
                h, w = image_left_ocv.shape[:2]
            
                for hand_landmarks in results.multi_hand_landmarks:
                
                    mp_drawing.draw_landmarks(
                        image_bgr,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
                    
                    lm = results.multi_hand_landmarks[0] # wrist
         
            #Calculate elbow angle
            for body in bodies.object_list:
                r_wrist = body.keypoint[4]
                r_elbow = body.keypoint[3]
                r_shoulder = body.keypoint[2]
                    
                r_upper_arm = np.array(r_shoulder) - np.array(r_elbow)
                r_forearm = np.array(r_wrist) - np.array(r_elbow)
                    
                angle = angle_between(r_upper_arm, r_forearm)
                

                
                cv2.putText(
                            image_bgr,
                            f"Right elbow: {angle:.1f} deg",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                            )
            
            cv2.imshow("ZED | 2D View", image_bgr)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break
                
                
    
    # Disable modules and close camera
    
    image.free(sl.MEM.CPU)
    
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()
