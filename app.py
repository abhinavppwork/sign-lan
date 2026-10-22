import atexit, queue, threading, time
from collections import deque
import cv2, pyttsx3, mediapipe as mp
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

MODEL_PATH="best.pt"; CONFIDENCE=.55; STABLE_SECONDS=1.5; MIN_BOX_SIZE=35
HOLD_SECONDS=.85; SWITCH_SECONDS=.60; WINDOW_SECONDS=.45
app=Flask(__name__); model=YOLO(MODEL_PATH); speech_queue=queue.Queue()
hand_options=mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1,
    min_hand_detection_confidence=.35, min_hand_presence_confidence=.35,
    min_tracking_confidence=.35)
hand_tracker=mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
camera=None; camera_lock=threading.Lock(); lock=threading.Lock()
state={"running":False,"last":None,"stable":time.time(),"spoken":None,"pending":None,"pending_at":None,"seen_at":None,"box":None,"confidence":0,"recent":deque(),"history":[],"status":"Camera paused","progress":0,"fps":0,"frame_at":time.time()}

def speech_worker():
    while True:
        text=speech_queue.get(); engine=None
        try:
            if text is None: return
            # Fresh engine prevents Windows pyttsx3 from getting stuck after speech one.
            engine=pyttsx3.init(); engine.setProperty("rate",165); engine.setProperty("volume",1)
            engine.say(text); engine.runAndWait()
        except Exception as error: print("Speech error:",error)
        finally:
            if engine:
                try: engine.stop()
                except Exception: pass
            speech_queue.task_done()
threading.Thread(target=speech_worker,daemon=True).start()

def reset():
    state.update(last=None,stable=time.time(),spoken=None,pending=None,pending_at=None,seen_at=None,box=None,confidence=0,recent=deque(),progress=0)

def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera=cv2.VideoCapture(0)
            if not camera.isOpened(): raise RuntimeError("Cannot open webcam. Check camera permissions.")
        return camera

def release_camera():
    global camera
    with camera_lock:
        if camera: camera.release(); camera=None

def process(frame):
    frame=cv2.flip(frame,1); h,w=frame.shape[:2]
    hand_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    hand_result=hand_tracker.detect_for_video(hand_image,int(time.monotonic()*1000))
    raw=None; confidence=0; box=None
    crop_box=None
    if hand_result.hand_landmarks:
        points=hand_result.hand_landmarks[0]
        xs=[point.x*w for point in points]; ys=[point.y*h for point in points]
        # A generous square crop keeps the complete hand at any screen position
        # and makes signs at different distances look consistent to YOLO.
        size=max(max(xs)-min(xs),max(ys)-min(ys),80)*1.75; cx=(min(xs)+max(xs))/2; cy=(min(ys)+max(ys))/2
        left=max(0,int(cx-size/2)); top=max(0,int(cy-size/2)); right=min(w,int(cx+size/2)); bottom=min(h,int(cy+size/2))
        if right-left>=MIN_BOX_SIZE and bottom-top>=MIN_BOX_SIZE: crop_box=(left,top,right,bottom)
    if crop_box:
        left,top,right,bottom=crop_box; source=frame[top:bottom,left:right]
    else:
        source=None
    for result in model.predict(source,conf=CONFIDENCE,imgsz=640,verbose=False) if source is not None else []:
        for candidate in result.boxes:
            score=float(candidate.conf[0]); label=str(model.names[int(candidate.cls[0])]).upper(); bx1,by1,bx2,by2=map(int,candidate.xyxy[0].tolist())
            if label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and len(label)==1 and bx2-bx1>=MIN_BOX_SIZE and by2-by1>=MIN_BOX_SIZE and score>confidence:
                raw,confidence,box=label,score,(bx1+left,by1+top,bx2+left,by2+top)
    now=time.time()
    with lock:
        if raw:
            state["recent"].append((now,raw))
            while state["recent"] and now-state["recent"][0][0]>WINDOW_SECONDS: state["recent"].popleft()
            counts={}
            for _,letter in state["recent"]: counts[letter]=counts.get(letter,0)+1
            consensus=max(counts,key=counts.get); previous=state["last"]
            detected=previous if previous in counts and counts[previous]>=counts[consensus]*.8 else consensus
            state.update(seen_at=now,box=box,confidence=confidence)
        elif state["last"] and state["seen_at"] and now-state["seen_at"]<=HOLD_SECONDS:
            detected=state["last"]; box=state["box"]
        else: detected=None
        if not detected:
            reset(); state["status"]="Show one hand anywhere in view"
        else:
            if detected != state["last"]:
                if detected != state["pending"]: state.update(pending=detected,pending_at=now)
                elif now-state["pending_at"]>=SWITCH_SECONDS: state.update(last=state["pending"],stable=now,spoken=None,pending=None,pending_at=None)
            else: state.update(pending=None,pending_at=None)
            active=state["last"]; duration=now-state["stable"] if active else 0; state["progress"]=min(100,round(duration/STABLE_SECONDS*100))
            if active and duration>=STABLE_SECONDS and state["spoken"]!=active:
                speech_queue.put(active); state["spoken"]=active; state["history"]=([active]+state["history"])[:8]
            state["status"]=f"Spoken: {active}" if active and state["spoken"]==active else f"Reading {active or state['pending'] or '—'} · hold steady"
        state["fps"]=round(1/max(now-state["frame_at"],1e-6),1); state["frame_at"]=now; status=state["status"]
    if crop_box: cv2.rectangle(frame,crop_box[:2],crop_box[2:],(255,200,0),2)
    if box:
        bx1,by1,bx2,by2=box; cv2.rectangle(frame,(bx1,by1),(bx2,by2),(0,255,0),2)
    cv2.putText(frame,status,(20,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,220,255),2)
    return frame

def stream():
    while True:
        with lock: running=state["running"]
        if not running: break
        try:
            ok,frame=get_camera().read()
            if not ok: break
            ok,buffer=cv2.imencode(".jpg",process(frame))
            if ok: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buffer.tobytes()+b"\r\n"
        except RuntimeError as error: print(error); break
    release_camera()

@app.route("/")
def index():
    # The header intentionally keeps only the product identity; it has no inactive controls.
    return render_template("signbridge.html").replace("</head>", "<style>.round{display:none}.profile{margin-left:0}</style></head>")
@app.route("/video_feed")
def video_feed(): return Response(stream(),mimetype="multipart/x-mixed-replace; boundary=frame")
@app.post("/api/start")
def start():
    with lock: state.update(running=True,status="Starting camera…")
    return jsonify(ok=True)
@app.post("/api/stop")
def stop():
    with lock: state["running"]=False; reset(); state["status"]="Camera paused"
    return jsonify(ok=True)
@app.get("/api/status")
def status():
    with lock: return jsonify(running=state["running"],letter=state["last"] or state["pending"] or "—",status=state["status"],progress=state["progress"],history=state["history"],fps=state["fps"])
@atexit.register
def cleanup(): release_camera(); speech_queue.put(None)
if __name__=="__main__":
    print("SignBridge UI: http://127.0.0.1:5000"); app.run(host="127.0.0.1",port=5000,threaded=True)
