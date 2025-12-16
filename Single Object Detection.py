from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument(&quot;-v&quot;, &quot;--video&quot;, type=str,
help=&quot;path to input video file&quot;)
ap.add_argument(&quot;-t&quot;, &quot;--tracker&quot;, type=str, default=&quot;kcf&quot;,
help=&quot;OpenCV object tracker type&quot;)
args = vars(ap.parse_args())
(major, minor) = cv2.__version__.split(&quot;.&quot;)[:2]
# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) &lt; 3:
tracker = cv2.Tracker_create(args[&quot;tracker&quot;].upper())
# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
&quot;csrt&quot;: cv2.TrackerCSRT_create,
&quot;kcf&quot;: cv2.TrackerKCF_create,
#&quot;boosting&quot;: cv2.TrackerBoosting_create,

78

&quot;mil&quot;: cv2.TrackerMIL_create,
#&quot;tld&quot;: cv2.TrackerTLD_create,
#&quot;medianflow&quot;: cv2.TrackerMedianFlow_create,
#&quot;mosse&quot;: cv2.TrackerMOSSE_create
}
# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS[args[&quot;tracker&quot;]]()
# initialize the bounding box coordinates of the object we are going
# to track
initBB = None
color = (0,255,0)
pause_frame = 0
if not args.get(&quot;video&quot;, False):
print(&quot;[INFO] starting video stream...&quot;)
vs = VideoStream(src=0).start()
time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
vs = cv2.VideoCapture(args[&quot;video&quot;])
# initialize the FPS throughput estimator
Out = cv2.VideoWriter(&#39;outpy.avi&#39;,cv2.VideoWriter_fourcc(&#39;M&#39;,&#39;J&#39;,&#39;P&#39;,&#39;G&#39;), 10,
(int(vs.get(3)),int(vs.get(4))))
fps = None
trajectories = []
pause = 0
prev_box = [0,0]
count = 0
# loop over frames from the video stream
def distance(a,b) :
return (a[0]-b[0])**2+(a[1]-b[1])**2
while True:

79
# grab the current frame, then handle if we are using a
# VideoStream or VideoCapture object
frame = vs.read()
frame = frame[1] if args.get(&quot;video&quot;, False) else frame
# check to see if we have reached the end of the stream
if frame is None:
break
# resize the frame (so we can process it faster) and grab the
# frame dimensions
#frame = imutils.resize(frame, width=1920)
(H, W) = frame.shape[:2]
if initBB is not None:
# grab the new bounding box coordinates of the object
(success, box) = tracker.update(frame)
if frame is None :
break
# check to see if the tracking was a success
if success:
(x, y, w, h) = [int(v) for v in box]
cv2.rectangle(frame, (x, y), (x + w, y + h),
(0, 255, 0), 2)
centx = x + int(w/2)
centy = y + int(h/2)
d = distance((centx,centy),prev_box)
if d &lt; 2 :
cv2.putText(frame, &quot;Pause&quot;, (centx,centy),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
pause_frame = pause_frame+1
trajectories.append([centx,centy])
cv2.polylines(frame, [np.array(trajectories)],False,color)
if count%3 == 0 :

80

prev_box = (centx,centy)
cv2.putText(frame, f&quot;Pause :{round(pause_frame/60,2)}s&quot;, (10,50),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
# update the FPS counter
fps.update()
fps.stop()
# initialize the set of information we&#39;ll be displaying on
# the frame
info = [
(&quot;Tracker&quot;, args[&quot;tracker&quot;]),
(&quot;Success&quot;, &quot;Yes&quot; if success else &quot;No&quot;),
(&quot;FPS&quot;, &quot;{:.2f}&quot;.format(fps.fps())),
]
# loop over the info tuples and draw them on our frame
for (i, (k, v)) in enumerate(info):
text = &quot;{}: {}&quot;.format(k, v)
cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
out.write(frame)
cv2.imshow(&quot;Frame&quot;, frame)
key = cv2.waitKey(1) #&amp; 0xFF
# if the &#39;s&#39; key is selected, we are going to &quot;select&quot; a bounding
# box to track
if key == ord(&quot;s&quot;):
# select the bounding box of the object we want to track (make
# sure you press ENTER or SPACE after selecting the ROI)
initBB = cv2.selectROI(&quot;Frame&quot;, frame, fromCenter=False,
showCrosshair=True)
# start OpenCV object tracker using the supplied bounding box
# coordinates, then start the FPS throughput estimator as well
tracker.init(frame, initBB)
fps = FPS().start()

81
# if the `q` key was pressed, break from the loop
elif key == ord(&quot;q&quot;):
break
count += 1
# if we are using a webcam, release the pointer
if not args.get(&quot;video&quot;, False):
vs.stop()
# otherwise, release the file pointer
else:
vs.release()
# close all windows
out.release()
vs.release()
cv2.destroyAllWindows()
print(&#39;the code&#39;)