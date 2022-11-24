import tkinter
import PIL.Image
import PIL.ImageTk
import cv2
import tkinter.font as font


import time
import pose_module as pm
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
import numpy as np 
from numpy import dot
from numpy.linalg import norm
from tkinter import filedialog


class App:
    def __init__(self, window):
        
        
        # define font
        myFont = font.Font(size=10)
        #get the screen size
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        self.window = window
        self.window.title("body movement comparison")
        
        window.title('Body movement comparison')
        size_str = "1050x750+"+str(int((screen_width-1250)/2))+"+" + str(int((screen_height-750)/2))
        window.geometry(size_str)


        # Create a canvas that can fit the above video source size
        self.canvas1 = tkinter.Canvas(window, width=500, height=600)
        self.canvas2 = tkinter.Canvas(window, width=500, height=600)
        self.canvas1.pack(padx=5, pady=10, side="left")
        self.canvas2.pack(padx=5, pady=10, side="left")

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 2
        
        
        self.b1=tkinter.Button(window, text='Compare', command=self.compare)
        self.b1.config( height = 2, width =8)
        self.b1.place(x=480, y=680)
        self.b1['font'] = myFont
        
        self.b1=tkinter.Button(window, text='Select video1', command=self.select_video1)
        self.b1.config( height = 2, width =15)
        self.b1.place(x=50, y=670)
        self.b1['font'] = myFont        
        
        
        self.b1=tkinter.Button(window, text='Select video2', command=self.select_video2)
        self.b1.config( height = 2, width =15)
        self.b1.place(x=630, y=670)
        self.b1['font'] = myFont    
        

        self.benchmark=tkinter.Entry(self.window, width=40)
        self.benchmark.place(x=200, y=680)
        self.benchmark.insert(0, "compare1/benchmark.mp4")
        

        self.video2=tkinter.Entry(self.window, width=40)
        self.video2.place(x=770, y=680)
        self.video2.insert(0, "compare1/video1.mp4")   
        self.k = 0
       
        
        self.window.mainloop()
        
        
    def select_video1(self):
        self.benchmark_video = filedialog.askopenfilename(filetypes=(("MP4", "*.mp4"), ("AVI", "*.avi")))
        
        self.benchmark.delete(0 ,200)
      
        self.benchmark.insert(0, self.benchmark_video)
        
        
    def select_video2(self):
        self.user_vide = filedialog.askopenfilename(filetypes=(("MP4", "*.mp4"), ("AVI", "*.avi")))    
        
        self.video2.delete(0 ,200)
        self.video2.insert(0, self.user_vide)  
        
    def compare(self):
        self.benchmark_video = self.benchmark.get()
        self.user_video = self.video2.get()
        self.vid = MyVideoCapture(self.benchmark_video, self.user_video)
        self.run()

    def run(self):
        # Get a frame from the video source
        ret1, frame1, ret2, frame2 = self.vid.get_frame
       
        self.k = self.k +1
        
        if ret1 and ret2:
                self.photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame1))
                self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame2))
                self.canvas1.create_image(0, 0, image=self.photo1, anchor=tkinter.NW)
                self.canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)

        self.window.after(self.delay, self.run)


class MyVideoCapture:
    def __init__(self, video_source1, video_source2):
        # Open the video source
        self.benchmark_cam = cv2.VideoCapture(video_source1)
        self.user_cam = cv2.VideoCapture(video_source2)
        self.fps_time = 0 #Initializing fps to 0

        self.detector_1 = pm.poseDetector()
        self.detector_2 = pm.poseDetector()
        self.frame_counter = 0
        self.correct_frames = 0


        if not self.benchmark_cam.isOpened():
            raise ValueError("Unable to open video source", video_source1)

    @property
    def get_frame(self):
        ret1 = ""
        ret2 = ""
        if self.benchmark_cam.isOpened() and self.user_cam.isOpened():
            ret1, frame1 = self.benchmark_cam.read()
            ret2, frame2 = self.user_cam.read()
            frame1 = cv2.resize(frame1, (500, 600))
            frame2 = cv2.resize(frame2, (500, 600))
            
			#Loop the video if it ended. If the last frame is reached, reset the capture and the frame_counter
            if self.frame_counter == self.user_cam.get(cv2.CAP_PROP_FRAME_COUNT):
                self.frame_counter = 0 #Or whatever as long as it is the same as next line
                self.correct_frames = 0
                self.user_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ###winname = "User Video"
            ##cv2.namedWindow(winname)		   # Create a named window
            ###cv2.moveWindow(winname, 720,100)  # Move it to desired location
            ###frame2 = cv2.resize(frame2, (720,640))
            frame2 = self.detector_1.findPose(frame2)
            lmList_user = self.detector_1.findPosition(frame2)
            del lmList_user[1:11]

            ### ret_val_1,frame1 = self.benchmark_cam.read()
			#Loop the video if it ended. If the last frame is reached, reset the capture and the frame_counter
            if self.frame_counter == self.benchmark_cam.get(cv2.CAP_PROP_FRAME_COUNT):
                self.frame_counter = 0 #Or whatever as long as it is the same as next line
                self.correct_frames = 0
                self.benchmark_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ###frame1 = cv2.resize(frame1, (720,640))
            frame1 = self.detector_2.findPose(frame1)
            lmList_benchmark = self.detector_2.findPosition(frame1)
            del lmList_benchmark[1:11]

            
            
            lmList_user = np.array(lmList_user).astype(np.float)
            lmList_benchmark = np.array(lmList_benchmark).astype(np.float)
            
            
            if len(lmList_user) > 1 and len(lmList_benchmark) > 1:
                
                """
            
                max1_1 = max(lmList_user[:,1]);
                max1_2 = max(lmList_user[:,2]);
                
                max2_1 = max(lmList_benchmark[:,1]);
                max2_2 = max(lmList_benchmark[:,2]);            
                
                lmList_user[:,1] = lmList_user[:,1]/max1_1 * max2_1
                lmList_user[:,2] = lmList_user[:,2]/max1_2 * max2_2
                
                
                
                lmList_user[:,1] = np.linalg.norm(lmList_user[:,1])
                lmList_user[:,2] = np.linalg.norm(lmList_user[:,2])
                
                
                lmList_benchmark[:,1] = np.linalg.norm(lmList_benchmark[:,1])
                lmList_benchmark[:,2] = np.linalg.norm(lmList_benchmark[:,2])
                
                """
                min_ = min(lmList_user[:,1])
                max_ = max(lmList_user[:,1])
                
                                
                lmList_user[:,1] = (lmList_user[:,1] - min_)/(max_ - min_)
                
                
                min_ = min(lmList_user[:,2])
                max_ = max(lmList_user[:,2])
                
                                
                lmList_user[:,2] = (lmList_user[:,2] - min_)/(max_ - min_)
                

                min_ = min(lmList_benchmark[:,1])
                max_ = max(lmList_benchmark[:,1])
                
                                
                lmList_benchmark[:,1] = (lmList_benchmark[:,1] - min_)/(max_ - min_)
                
                
                min_ = min(lmList_benchmark[:,2])
                max_ = max(lmList_benchmark[:,2])
                
                                
                lmList_benchmark[:,2] = (lmList_benchmark[:,2] - min_)/(max_ - min_)

                
                
                #lmList_user[:,1] = lmList_user[:,1]/500.0
                #lmList_user[:,2] = lmList_user[:,2]/600.                
                
                
            
            
            if (ret1 or ret2) and (len(lmList_user) >1 or  len(lmList_benchmark) >1):
                self.frame_counter += 1
                try:
                    error, _ = fastdtw(lmList_user, lmList_benchmark, dist=cosine)
                    error = error *100
                    if error > 1:
                        error =1
                    
                    #error = self.compare(lmList_user, lmList_benchmark)
                    #error =self.get_error(lmList_user, lmList_benchmark) 
                    
                    
                except:
                    error = 0
                    
                if (len(lmList_user)<10) or (len(lmList_benchmark)<10):
                    error = 1;

				# Displaying the error percentage
                cv2.putText(frame2, 'Error: {}%'.format(str(round(100*(float(error)),2))), (10, 30),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

				# If the similarity is > 90%, take it as correct step. Otherwise incorrect step.
                if error < 0.10:
                    cv2.putText(frame2, "CORRECT STEPS", (40, 600),
								cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.correct_frames += 1
                else:
                    cv2.putText(frame2,  "INCORRECT STEPS", (40, 600),
								cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ##cv2.putText(frame2, "FPS: %f" % (1.0 / (time.time() - self.fps_time)), (10, 50),
				##			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

				# Display the dynamic accuracy of dance as the percentage of frames that appear as correct
                if self.frame_counter==0:
                    self.frame_counter = self.user_cam.get(cv2.CAP_PROP_FRAME_COUNT)
                cv2.putText(frame2, "Dance Steps Accurately Done: {}%".format(str(round(100*self.correct_frames/self.frame_counter, 2))), (10, 70), 
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
				

                fps_time = time.time()
       
            
            
            if ret1 and ret2:
                # Return a boolean success flag and the current frame converted to BGR
                return ret1, cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), ret2, cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            else:
                return ret1, None, ret2, None
        else:
            return ret1, None, ret2, None
    def get_error(self, vec1, vec2):
        aa = 1
        cos_sim1 = dot(vec1[:,1], vec2[:,1])/(norm(vec1[:,1])*norm(vec2[:,1]))
        cos_sim2 = dot(vec1[:,2], vec2[:,2])/(norm(vec1[:,2])*norm(vec2[:,2]))
        
        return (cos_sim1 + cos_sim2)/2
        
    
    def compare(self, frame_vectors, template_vectors):
        return [self.dot_or_none(i, t) for i, t in zip(frame_vectors, template_vectors)]

    def dot_or_none(self, vec1, vec2):
        return np.dot(vec1, vec2) if vec1 is not None and vec2 is not None else None


    # Release the video source when the object is destroyed
    def __del__(self):
        if self.benchmark_cam.isOpened():
            self.benchmark_cam.release()
        if self.user_cam.isOpened():
            self.user_cam.release()



# Create a window and pass it to the Application object
App(tkinter.Tk())