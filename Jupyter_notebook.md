# Image processing project (lane lines Detection)

# We are:

#17t0346_احمد جابر عبد العزيز موسى

#16W0015_حسين عبد الملك علي قاسم العبسي

#17E0020_خالد احمد محمد عبد العزيز


```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from IPython.display import HTML, Video,Image
from moviepy.editor import VideoFileClip
from PerspectiveTransformation import *
from LaneLines import *
from docopt import docopt
from CameraCalibration import CameraCalibration
from Thresholding import *
```


```python
class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False,fps=None,preset = 'ultrafast',logger='bar')
```

# Frame Working Steps
#1 Camera Calibration

#2 Transforming-forward

#3 Thresholding

#4 lane line detection

#5 Transforming-backward

#6 Applying result to original frame


# Original Frame 


```python
test_img = mpimg.imread('test_images/test1.jpg')
Final_img = np.copy(test_img)
plt.imshow(test_img)
```




    <matplotlib.image.AxesImage at 0x18ddac7cfd0>




    
![png](output_7_1.png)
    


# 1,2 - Calibrating camera & transforming-forward


```python
calibration = CameraCalibration('camera_cal', 9, 6)
test= calibration.undistort(test_img)
transform = PerspectiveTransformation()
test = transform.forward(test)
plt.imshow(test)
```




    <matplotlib.image.AxesImage at 0x18ddacebd30>




    
![png](output_9_1.png)
    


# 3 - Thresholding


```python
thresholding = Thresholding()
test = thresholding.forward(test)
plt.imshow(test)
```




    <matplotlib.image.AxesImage at 0x18dda9ef1c0>




    
![png](output_11_1.png)
    


# 4 - Lane Line detection


```python
lanelines = LaneLines()
test = lanelines.forward(test)
plt.imshow(test)
```




    <matplotlib.image.AxesImage at 0x18ddaad88b0>




    
![png](output_13_1.png)
    


# 5 - Transforming-backward


```python
test = transform.backward(test)
plt.imshow(test)
```




    <matplotlib.image.AxesImage at 0x18ddaa7df10>




    
![png](output_15_1.png)
    


# 6 - Final result


```python
Final_img = cv2.addWeighted(Final_img, 1, test, 0.6, 0)
plt.imshow(Final_img)
```




    <matplotlib.image.AxesImage at 0x18ddab2e730>




    
![png](output_17_1.png)
    



```python
Final_img = lanelines.plot(Final_img)
plt.imshow(Final_img)
```




    <matplotlib.image.AxesImage at 0x18ddab97dc0>




    
![png](output_18_1.png)
    


# Testing on Image


```python
input_path = 'test_images/test1.jpg'
output_path = 'F:/test.jpg'
fll2image = FindLaneLines()
fll2image.process_image(input_path, output_path)
Image(input_path,width=500,height = 500)
```




    
![jpeg](output_20_0.jpg)
    




```python
Image(output_path,width=500,height = 500)
```




    
![jpeg](output_21_0.jpg)
    



# Testing On Video


```python
input_path = 'project_video.mp4'
output_path = 'output_videos/'+input_path[:-4]+'_out.mp4'
Video(input_path,width=500,height = 500)
```




<video src="project_video.mp4" controls  width="500"  height="500">
      Your browser does not support the <code>video</code> element.
    </video>




```python
fll2video = FindLaneLines()
fll2video.process_video(input_path, output_path)
```

    t:  15%|██████████                                                        | 193/1260 [00:59<01:38, 10.81it/s, now=None]

    Moviepy - Building video output_videos/project_video_out.mp4.
    Moviepy - Writing video output_videos/project_video_out.mp4
    
    

    
    t:   0%|                                                                            | 0/1260 [00:00<?, ?it/s, now=None][A
    t:   0%|▏                                                                   | 3/1260 [00:00<01:12, 17.24it/s, now=None][A
    t:   0%|▎                                                                   | 5/1260 [00:00<01:35, 13.15it/s, now=None][A
    t:   1%|▍                                                                   | 7/1260 [00:00<01:44, 12.05it/s, now=None][A
    t:   1%|▍                                                                   | 9/1260 [00:00<01:48, 11.58it/s, now=None][A
    t:   1%|▌                                                                  | 11/1260 [00:00<01:50, 11.30it/s, now=None][A
    t:   1%|▋                                                                  | 13/1260 [00:01<01:50, 11.24it/s, now=None][A
    t:   1%|▊                                                                  | 15/1260 [00:01<01:52, 11.08it/s, now=None][A
    t:   1%|▉                                                                  | 17/1260 [00:01<01:50, 11.25it/s, now=None][A
    t:   2%|█                                                                  | 19/1260 [00:01<01:48, 11.44it/s, now=None][A
    t:   2%|█                                                                  | 21/1260 [00:01<01:52, 11.03it/s, now=None][A
    t:   2%|█▏                                                                 | 23/1260 [00:01<01:48, 11.40it/s, now=None][A
    t:   2%|█▎                                                                 | 25/1260 [00:02<01:47, 11.49it/s, now=None][A
    t:   2%|█▍                                                                 | 27/1260 [00:02<01:51, 11.03it/s, now=None][A
    t:   2%|█▌                                                                 | 29/1260 [00:02<01:57, 10.51it/s, now=None][A
    t:   2%|█▋                                                                 | 31/1260 [00:02<01:54, 10.72it/s, now=None][A
    t:   3%|█▊                                                                 | 33/1260 [00:02<01:54, 10.73it/s, now=None][A
    t:   3%|█▊                                                                 | 35/1260 [00:03<01:52, 10.89it/s, now=None][A
    t:   3%|█▉                                                                 | 37/1260 [00:03<01:53, 10.73it/s, now=None][A
    t:   3%|██                                                                 | 39/1260 [00:03<01:55, 10.60it/s, now=None][A
    t:   3%|██▏                                                                | 41/1260 [00:03<01:53, 10.73it/s, now=None][A
    t:   3%|██▎                                                                | 43/1260 [00:03<01:54, 10.63it/s, now=None][A
    t:   4%|██▍                                                                | 45/1260 [00:04<01:52, 10.83it/s, now=None][A
    t:   4%|██▍                                                                | 47/1260 [00:04<01:53, 10.67it/s, now=None][A
    t:   4%|██▌                                                                | 49/1260 [00:04<01:52, 10.78it/s, now=None][A
    t:   4%|██▋                                                                | 51/1260 [00:04<01:52, 10.74it/s, now=None][A
    t:   4%|██▊                                                                | 53/1260 [00:04<01:49, 11.06it/s, now=None][A
    t:   4%|██▉                                                                | 55/1260 [00:04<01:49, 10.97it/s, now=None][A
    t:   5%|███                                                                | 57/1260 [00:05<01:51, 10.74it/s, now=None][A
    t:   5%|███▏                                                               | 59/1260 [00:05<01:48, 11.10it/s, now=None][A


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_13772/1057035723.py in <module>
          1 fll2video = FindLaneLines()
    ----> 2 fll2video.process_video(input_path, output_path)
    

    ~\AppData\Local\Temp/ipykernel_13772/2335627761.py in process_video(self, input_path, output_path)
         32         clip = VideoFileClip(input_path)
         33         out_clip = clip.fl_image(self.forward)
    ---> 34         out_clip.write_videofile(output_path, audio=False,fps=None,preset = 'ultrafast',logger='bar')
    

    <decorator-gen-172> in write_videofile(self, filename, fps, codec, bitrate, audio, audio_fps, preset, audio_nbytes, audio_codec, audio_bitrate, audio_bufsize, temp_audiofile, rewrite_audio, remove_temp, write_logfile, verbose, threads, ffmpeg_params, logger)
    

    D:\MCT_Programs\anaconda3\lib\site-packages\moviepy\decorators.py in requires_duration(f, clip, *a, **k)
         52         raise ValueError("Attribute 'duration' not set")
         53     else:
    ---> 54         return f(clip, *a, **k)
         55 
         56 
    

    <decorator-gen-171> in write_videofile(self, filename, fps, codec, bitrate, audio, audio_fps, preset, audio_nbytes, audio_codec, audio_bitrate, audio_bufsize, temp_audiofile, rewrite_audio, remove_temp, write_logfile, verbose, threads, ffmpeg_params, logger)
    

    D:\MCT_Programs\anaconda3\lib\site-packages\moviepy\decorators.py in use_clip_fps_by_default(f, clip, *a, **k)
        133              for (k,v) in k.items()}
        134 
    --> 135     return f(clip, *new_a, **new_kw)
    

    <decorator-gen-170> in write_videofile(self, filename, fps, codec, bitrate, audio, audio_fps, preset, audio_nbytes, audio_codec, audio_bitrate, audio_bufsize, temp_audiofile, rewrite_audio, remove_temp, write_logfile, verbose, threads, ffmpeg_params, logger)
    

    D:\MCT_Programs\anaconda3\lib\site-packages\moviepy\decorators.py in convert_masks_to_RGB(f, clip, *a, **k)
         20     if clip.ismask:
         21         clip = clip.to_RGB()
    ---> 22     return f(clip, *a, **k)
         23 
         24 @decorator.decorator
    

    D:\MCT_Programs\anaconda3\lib\site-packages\moviepy\video\VideoClip.py in write_videofile(self, filename, fps, codec, bitrate, audio, audio_fps, preset, audio_nbytes, audio_codec, audio_bitrate, audio_bufsize, temp_audiofile, rewrite_audio, remove_temp, write_logfile, verbose, threads, ffmpeg_params, logger)
        298                                        logger=logger)
        299 
    --> 300         ffmpeg_write_video(self, filename, fps, codec,
        301                            bitrate=bitrate,
        302                            preset=preset,
    

    D:\MCT_Programs\anaconda3\lib\site-packages\moviepy\video\io\ffmpeg_writer.py in ffmpeg_write_video(clip, filename, fps, codec, bitrate, preset, withmask, write_logfile, audiofile, verbose, threads, ffmpeg_params, logger)
        218         nframes = int(clip.duration*fps)
        219 
    --> 220         for t,frame in clip.iter_frames(logger=logger, with_times=True,
        221                                         fps=fps, dtype="uint8"):
        222             if withmask:
    

    D:\MCT_Programs\anaconda3\lib\site-packages\moviepy\Clip.py in iter_frames(self, fps, with_times, logger, dtype)
        470         logger = proglog.default_bar_logger(logger)
        471         for t in logger.iter_bar(t=np.arange(0, self.duration, 1.0/fps)):
    --> 472             frame = self.get_frame(t)
        473             if (dtype is not None) and (frame.dtype != dtype):
        474                 frame = frame.astype(dtype)
    

    <decorator-gen-128> in get_frame(self, t)
    

    D:\MCT_Programs\anaconda3\lib\site-packages\moviepy\decorators.py in wrapper(f, *a, **kw)
         87         new_kw = {k: fun(v) if k in varnames else v
         88                  for (k,v) in kw.items()}
    ---> 89         return f(*new_a, **new_kw)
         90     return decorator.decorator(wrapper)
         91 
    

    D:\MCT_Programs\anaconda3\lib\site-packages\moviepy\Clip.py in get_frame(self, t)
         91                 return frame
         92         else:
    ---> 93             return self.make_frame(t)
         94 
         95     def fl(self, fun, apply_to=None, keep_duration=True):
    

    D:\MCT_Programs\anaconda3\lib\site-packages\moviepy\Clip.py in <lambda>(t)
        134 
        135         #mf = copy(self.make_frame)
    --> 136         newclip = self.set_make_frame(lambda t: fun(self.get_frame, t))
        137 
        138         if not keep_duration:
    

    D:\MCT_Programs\anaconda3\lib\site-packages\moviepy\video\VideoClip.py in <lambda>(gf, t)
        488         """
        489         apply_to = apply_to or []
    --> 490         return self.fl(lambda gf, t: image_func(gf(t)), apply_to)
        491 
        492     # --------------------------------------------------------------
    

    ~\AppData\Local\Temp/ipykernel_13772/2335627761.py in forward(self, img)
         16         img = self.calibration.undistort(img)
         17         img = self.transform.forward(img)
    ---> 18         img = self.thresholding.forward(img)
         19         img = self.lanelines.forward(img)
         20         img = self.transform.backward(img)
    

    ~\Downloads\Compressed\Advanced-Lane-Lines-master\Thresholding.py in forward(self, img)
         40 
         41         left_lane = threshold_abs(h_channel, 20, 30)
    ---> 42         left_lane &= threshold_rel(v_channel, 0.7, 1.0)
         43         left_lane[:,550:] = 0
         44 
    

    ~\Downloads\Compressed\Advanced-Lane-Lines-master\Thresholding.py in threshold_rel(img, lo, hi)
          8     vlo = vmin + (vmax - vmin) * lo
          9     vhi = vmin + (vmax - vmin) * hi
    ---> 10     return np.uint8((img >= vlo) & (img <= vhi)) * 255
         11 
         12 def threshold_abs(img, lo, hi):
    

    KeyboardInterrupt: 



```python
Video(output_path,width=500,height = 500)
```
