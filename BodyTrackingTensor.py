import cv2
import sys
import time
import numpy as np

import pyzed.sl as sl

import tensorrt as trt
import pycuda.driver as cuda

cuda.init()


sys.path.append("/usr/local/zed/samples/body tracking/python") #Where ogl_viewer is stored in system


# ------------------------------------------------------------
# TensorRT Hand Landmark Engine
# ------------------------------------------------------------

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTHandLandmark:

    def __init__(self, engine_path):

        print("Loading TensorRT engine:", engine_path)

        with open(engine_path, "rb") as f:

            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(
                f.read()
            )

        self.context = self.engine.create_execution_context()

        self.cuda_context = cuda.Device(0).retain_primary_context()
        
        self.cuda_context.push()
        self.stream = cuda.Stream()

        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_bindings):
            print(
                self.engine.get_binding_name(i),
                self.engine.get_binding_shape(i)
            )


        for i in range(self.engine.num_bindings):

            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape)

            host = cuda.pagelocked_empty(size, dtype)
            device = cuda.mem_alloc(host.nbytes)

            self.bindings.append(int(device))

            if self.engine.binding_is_input(i):
                self.inputs.append({
                    "name": name,
                    "host": host,
                    "device": device,
                    "shape": shape,
                    "index": i
                })
            else:
                self.outputs.append({
                    "name": name,
                    "host": host,
                    "device": device,
                    "shape": shape,
                    "index": i
                })
 

        print("TensorRT bindings:")

        for x in self.inputs:
            print("INPUT :", x["name"], x["shape"])

        for x in self.outputs:
            print("OUTPUT:", x["name"], x["shape"])
        self.cuda_context.pop()


    def infer(self, image):

        self.cuda_context.push()

        try:

            img = cv2.resize(image, (224,224))
            
            img = img.astype(np.float32)

            img = (img - 127.5) / 127.5

            img = np.transpose(img, (2,0,1))

            img = np.expand_dims(img,0)

            np.copyto(
                self.inputs[0]["host"],
                img.ravel()
            )

            cuda.memcpy_htod_async(
                self.inputs[0]["device"],
                self.inputs[0]["host"],
                self.stream
            )

            ok = self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )

            if not ok:
                print("TensorRT execution failed")

            results = {}

            for out in self.outputs:
                cuda.memcpy_dtoh_async(
                    out["host"],
                    out["device"],
                    self.stream
                )

            self.stream.synchronize()

            for out in self.outputs:
                results[out["name"]] = (
                    out["host"]
                    .reshape(out["shape"])
                    .copy()
                )

            return results

        finally:
            self.cuda_context.pop()


def crop_hand(frame, wrist, size=120):

    x,y = wrist

    h,w = frame.shape[:2]

    x1=int(x-size)
    y1=int(y-size)
    x2=int(x+size)
    y2=int(y+size)


    pad_left=max(0,-x1)
    pad_top=max(0,-y1)
    pad_right=max(0,x2-w)
    pad_bottom=max(0,y2-h)


    crop=cv2.copyMakeBorder(
        frame,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0,0,0)
    )


    x1+=pad_left
    x2+=pad_left
    y1+=pad_top
    y2+=pad_top


    crop=crop[y1:y2,x1:x2]


    return crop,(max(x1-pad_left, 0), max(y1-pad_top, 0))

class LowPassFilter:
    def __init__(self):
        self.initialized = False
        self.prev = None

    def filter(self, value, alpha):
        if not self.initialized:
            self.prev = value
            self.initialized = True
            return value

        result = alpha * value + (1 - alpha) * self.prev
        self.prev = result
        return result

class OneEuroFilter:

    def __init__(self,
                 min_cutoff=1.2,
                 beta=0.02,
                 d_cutoff=1.0):

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

        self.last_time = None

    def smoothing_factor(self, dt, cutoff):
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)

    def filter(self, x):

        now = time.time()

        if self.last_time is None:
            self.last_time = now
            return self.x_filter.filter(x, 1.0)

        dt = now - self.last_time
        self.last_time = now

        if dt <= 0:
            dt = 1e-6

        # Estimate velocity
        dx = (x - self.x_filter.prev) / dt

        alpha_d = self.smoothing_factor(dt, self.d_cutoff)
        dx_hat = self.dx_filter.filter(dx, alpha_d)

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        alpha = self.smoothing_factor(dt, cutoff)

        return self.x_filter.filter(x, alpha)

class LandmarkSmoother:

    def __init__(self, num_landmarks, dims):

        self.filters = [
            [OneEuroFilter() for _ in range(dims)]
            for _ in range(num_landmarks)
        ]

    def smooth(self, landmarks):

        landmarks = np.asarray(landmarks, dtype=np.float32)

        smoothed = np.empty_like(landmarks)

        for i in range(landmarks.shape[0]):
            for j in range(landmarks.shape[1]):
                smoothed[i, j] = self.filters[i][j].filter(
                    landmarks[i, j]
                )

        return smoothed




# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------

def unit_vector(v):

    n = np.linalg.norm(v)

    if n == 0:
        return v

    return v / n



def angle_between(v1,v2):

    v1 = unit_vector(v1)
    v2 = unit_vector(v2)

    return np.degrees(
        np.arccos(
            np.clip(
                np.dot(v1,v2),
                -1,
                1
            )
        )
    )
    
class KeypointSmoother:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.previous = None

    def smooth(self, keypoints):

        keypoints = np.array(keypoints, dtype=np.float32).copy()

        if self.previous is None:
            self.previous = keypoints.copy()
            return keypoints

        smoothed = (
            self.alpha * keypoints +
            (1 - self.alpha) * self.previous
        )

        self.previous = smoothed.copy()

        return smoothed




def draw_body_skeleton(image, bodies, scale_x, scale_y, rgb, hand_net, body_smoother):

    if len(bodies.object_list) == 0:
        return

    result = None
    offset = (0, 0)
    crop_shape = (0, 0)
    
    body = bodies.object_list[0]
    print("BODY SHAPE:", body.keypoint_2d.shape)
    kp = body_smoother.smooth(body.keypoint_2d)
    
    kp[:,0] *= scale_x
    kp[:,1] *= scale_y
    print("image shape:", rgb.shape)
    print("body kp max:", np.max(body.keypoint_2d, axis=0))

    
    for wrist_id, elbow_id, shoulder_id in [
        (4,3,2),   # right hand
        (7,6,5)    # left hand
    ]:



        shoulder = kp[shoulder_id]
        elbow = kp[elbow_id]
        wrist = kp[wrist_id]

        arm_length = np.linalg.norm(shoulder-elbow)

        size = int(arm_length * 0.8)
        



        forearm = wrist - elbow
        length = np.linalg.norm(forearm)
        

        if length > 1:
            direction = forearm / length
        else:
            direction = np.array([0.0, 0.0])

        # Move from wrist into the hand
        palm = wrist + direction * (0.35 * length)

        hand_crop, offset = crop_hand(
            rgb,
            palm,
            size=size
        )

        if hand_crop is not None:
        
            cv2.imshow(
                "hand crop",
                cv2.cvtColor(hand_crop, cv2.COLOR_RGB2BGR)
            )
            result = hand_net.infer(hand_crop)



    keypoints = kp


    # Skeleton connections for BODY_FORMAT.POSE_18
    skeleton = [
        (0,1),        # nose-neck

        (1,2),        # neck-right shoulder
        (2,3),        # right shoulder-elbow
        (3,4),        # right elbow-wrist

        (1,5),        # neck-left shoulder
        (5,6),        # left shoulder-elbow
        (6,7),        # left elbow-wrist

        (2,8),        # right shoulder-hip
        (8,9),        # right hip-knee
        (9,10),       # right knee-ankle

        (5,11),       # left shoulder-hip
        (11,12),      # left hip-knee
        (12,13),      # left knee-ankle

        (0,14),       # nose-right eye
        (0,15),       # nose-left eye
        (14,16),      # right eye-ear
        (15,17)       # left eye-ear
    ]


    h,w=image.shape[:2]


    # Draw joints
    for p in keypoints:

        if p[0] <= 0 or p[1] <= 0:
            continue

        x=int(p[0])
        y=int(p[1])

        cv2.circle(
            image,
            (x,y),
            5,
            (0,255,255),
            -1
        )


    # Draw bones
    for a,b in skeleton:

        pa = keypoints[a]
        pb = keypoints[b]


        if (
            pa[0] <= 0 or pa[1] <= 0 or
            pb[0] <= 0 or pb[1] <= 0
        ):
            continue


        cv2.line(
            image,
            (int(pa[0]), int(pa[1])),
            (int(pb[0]), int(pb[1])),
            (255,0,0),
            3
        )
    return result, offset, hand_crop.shape[:2]
    
    
    
    
  
  

  
# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    
    engine_path = (
        "/home/user/hand_tensorrt/"
        "hand_landmark_lite.engine"
    )


    hand_net = TRTHandLandmark(
        engine_path
    )



    print(
        "Starting ZED..."
    )
    
    zed = sl.Camera()


    init = sl.InitParameters()

    init.coordinate_units = (
        sl.UNIT.METER
    )

    init.depth_mode = (
        sl.DEPTH_MODE.ULTRA
    )

    init.coordinate_system = (
        sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    )


    if len(sys.argv)==2:

        init.set_from_svo_file(
            sys.argv[1]
        )



    if zed.open(init)!=sl.ERROR_CODE.SUCCESS:

        print("ZED open failed")
        exit()



    track = (
        sl.PositionalTrackingParameters()
    )

    track.set_as_static=True

    zed.enable_positional_tracking(
        track
    )



    obj = sl.ObjectDetectionParameters()

    obj.enable_tracking=True

    obj.enable_body_fitting=True

    obj.detection_model = (
        sl.DETECTION_MODEL.HUMAN_BODY_ACCURATE
    )

    obj.body_format = (
        sl.BODY_FORMAT.POSE_18
    )


    zed.enable_object_detection(
        obj
    )



    runtime = (
        sl.ObjectDetectionRuntimeParameters()
    )

    runtime.detection_confidence_threshold = 75
    
    
    

    
    bodies = sl.Objects()

    image = sl.Mat()



    resolution = sl.Resolution(
        960,
        540
    )

    frame=0

    last_result=None




    body_smoother = LandmarkSmoother(num_landmarks=18, dims=2)
    hand_smoother = LandmarkSmoother(num_landmarks=21, dims=3)
   # crop_smoother = CropSmoother()



    while True:


        if zed.grab()==sl.ERROR_CODE.SUCCESS:


            zed.retrieve_image(
                image,
                sl.VIEW.LEFT,
                sl.MEM.CPU,
                resolution
            )


            zed.retrieve_objects(
                bodies,
                runtime
            )


            img=image.get_data()


            rgb=cv2.cvtColor(
                img,
                cv2.COLOR_BGRA2RGB
            )


            # TensorRT hand inference

            start=time.time()
            
            info = zed.get_camera_information()    
    
            print(
                "image mean:",
                rgb.mean(),
                "std:",
                rgb.std()
            )
            

            display=cv2.cvtColor(
                img,
                cv2.COLOR_BGRA2BGR
            )

            cam_width = resolution.width
            cam_height = resolution.height


            cam_info = zed.get_camera_information()

            native_w = cam_info.camera_configuration.camera_resolution.width
            native_h = cam_info.camera_configuration.camera_resolution.height


            result = draw_body_skeleton(
                display,
                bodies,
                display.shape[1] / native_w,
                display.shape[0] / native_h,
                rgb,
                hand_net,
                body_smoother
            )

            if result is None:
                cv2.imshow("ZED TensorRT Hand", display)
                continue
                
            hand, offset, crop_shape = result

            if hand is None:
                cv2.imshow(
                    "ZED TensorRT Hand",
                    display
                )
                continue


            landmarks = hand_smoother.smooth(
                hand["Identity"]
                .reshape(21,3)
            )

            print(landmarks.min(), landmarks.max())
            fps = 1/(time.time()-start)

            

            crop_h, crop_w = crop_shape

            ox, oy = offset

            for p in landmarks:

                x = int(p[0] * crop_w/224)
                y = int(p[1] * crop_h/224)


                cv2.circle(
                    display,
                    (x + ox, y + oy),
                    4,
                    (0,255,0),
                    -1
                )



            if fps:

                print(
                    f"\r"
                    f"FPS {fps:5.1f} | "
                    ,
                    end=""
                )

            print(hand.keys())

            print(hand["Identity"].shape)
 
            for k,v in hand.items():
                print(k, v.shape,
                "min:",
                v.min(),
                "max:",
                v.max()
                )

            cv2.imshow(
                "ZED TensorRT Hand",
                display
            )



        if cv2.waitKey(1)==ord('q'):
            break



    image.free(sl.MEM.CPU)

    zed.disable_object_detection()

    zed.disable_positional_tracking()
    
    zed.close()

    cv2.destroyAllWindows()

