import tkinter
import PIL.Image
import PIL.ImageTk
import tkinter.font as font
import cv2

import os
import sys
import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet

import time
import pose_module as pm
from scipy.spatial.distance import cosine

import numpy as np 
from numpy import dot
from numpy.linalg import norm
from tkinter import filedialog

warnings.filterwarnings("ignore")


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
        self.canvas1 = tkinter.Canvas(window, width=300, height=600)
        self.canvas2 = tkinter.Canvas(window, width=300, height=600)
        self.canvas1.pack(padx=5, pady=10, side="left")
        self.canvas2.pack(padx=5, pady=10, side="left")

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 50
        
        
        self.b1=tkinter.Button(window, text='BackGround Change', command=self.background_change)
        self.b1.config( height = 2, width =18)
        self.b1.place(x=460, y=680)
        self.b1['font'] = myFont
        
        self.b1=tkinter.Button(window, text='Select video', command=self.select_video)
        self.b1.config( height = 2, width =15)
        self.b1.place(x=50, y=670)
        self.b1['font'] = myFont        
        
        self.label=tkinter.Label(window, text='output :')
        self.label.config( height = 2, width =15)
        self.label.place(x=50, y=715)
        self.label['font'] = myFont         
        
        
        self.b1=tkinter.Button(window, text='Select background', command=self.select_background)
        self.b1.config( height = 2, width =15)
        self.b1.place(x=630, y=670)
        self.b1['font'] = myFont    
        

        self.benchmark=tkinter.Entry(self.window, width=40)
        self.benchmark.place(x=200, y=680)
        self.benchmark.insert(0, "input_video/bandicam 2022-932.mp4")
        ### self.benchmark.insert(0, "input_video/rec8.mp4")
        

        self.background_video=tkinter.Entry(self.window, width=40)
        self.background_video.place(x=770, y=680)
        self.background_video.insert(0, "input_video/bandicam 2022-334.mp4")   
        self.k = 0
        
        self.output=tkinter.Entry(self.window, width=40)
        self.output.place(x=200, y=720)
        self.output.insert(0, "out_video/output.mp4")
      
        
        self.window.mainloop()
        
        
    def select_video(self):
        self.benchmark_video = filedialog.askopenfilename(filetypes=(("MP4", "*.mp4"), ("AVI", "*.avi")))
        
        self.benchmark.delete(0 ,200)
      
        self.benchmark.insert(0, self.benchmark_video)
        
        
    def select_background(self):
        self.user_vide = filedialog.askopenfilename(filetypes=(("jpeg files", "*.mp4"), ("png files", "*.avi")))    
        
        self.background_video.delete(0 ,200)
        self.background_video.insert(0, self.user_vide)  
        
    def background_change(self):
        
        
       
        
        
        self.benchmark_video = self.benchmark.get()
        self.background_path = self.background_video.get()
        
        
        self.output_video = self.output.get()
        
        self.vid = MyVideoCapture(self.benchmark_video)
        self.vid2 = MyVideoCapture(self.background_path)
        
        
        self.flag = 1
        
        self.BGRemove = BGRemove('pretrained/modnet_webcam_portrait_matting.ckpt')
        
        # Get a frame from the video source
        ret1, frame1 = self.vid.get_frame
        ret2, frame2 = self.vid2.get_frame
        
        self.height1, self.width1 = (frame1.shape[0], frame1.shape[1])   
        
        
        width1 = int(500/self.height1*self.width1)
        self.width1 = width1
        self.height1 = 500
       
        
        self.height2, self.width2 = (frame2.shape[0], frame2.shape[1])
        
        width2 = int(500/self.height2*self.width2)
        self.width2 = width2
        self.height2 = 500        
      
        
        self.canvas1.config(width = self.width1, height = self.height1)
        self.canvas2.config(width = self.width2, height = self.height2)
        
        self.run()

    def get_background(self, frame2):
        
        
        
       
        im = self.BGRemove.pre_process(frame2)
        _, _, matte = self.BGRemove.modnet(im, inference=False)
        
        #s2 = frame2.shape
        #width = int(500/s2[1]*s2[0])
            
        matte = F.interpolate(matte, size=(
             self.height2, self.width2), mode='area')
        
        
        matte = (matte.repeat(1, 3, 1, 1))
        matte = matte[0].data.cpu().numpy().transpose(1, 2, 0)
        
        #matte = matte.astype(int)
        
        height, width, _ = matte.shape



        matte =  (1 - matte) * frame2
        return matte
        
        """
        height, width, _ = matte.shape
        if background:
            back_image = self.file_load(backgound_path)
            back_image = cv2.resize(
                back_image, (width, height), cv2.INTER_AREA)        

        """
            

    def run(self):
        # Get a frame from the video source
        ret1, frame1 = self.vid.get_frame
        ret2, frame2 = self.vid2.get_frame
        
        print(ret1)
        print(ret2)
       
        
        if self.flag == 1:
    
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                                
            
            
            self.out = cv2.VideoWriter(self.output_video,
                                      fourcc, 20.0, (self.height2, self.width2))
           
            
            self.flag = 0
       
        if ret1 and ret2:
               
                print('Video is processing..', end='\r')
                
                
                frame1 = np.uint8(cv2.resize(
                    frame1, (self.width2, self.height2), cv2.INTER_AREA))
                
                frame2 = np.uint8(cv2.resize(
                    frame2, (self.width2, self.height2), cv2.INTER_AREA))
                
                
                self.backGroundImage = self.get_background(frame2)
                
                
                
                im = self.BGRemove.pre_process(frame1)
                _, _, matte = self.BGRemove.modnet(im, inference=False)
                
                matte = F.interpolate(matte, size=(
                self.width2, self.height2), mode='area')
               
               
                #self.backGroundImage = F.interpolate(self.backGroundImage, size=(
                #self.width, self.height), mode='area')
                
                matte = np.uint8(self.BGRemove.post_process2(matte, True, self.backGroundImage))
              
                
                full_image =  matte
                
                full_image = np.uint8(cv2.resize(
                    full_image, (self.width2, self.height2), cv2.INTER_AREA))
                
                full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
                self.out.write(full_image)
                
                
                
                
                
                
                
                frame1 = np.uint8(cv2.resize(
                    frame1, (self.width1, self.height1), cv2.INTER_AREA))
            
                
                self.photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame1))
                self.canvas1.create_image(0, 0, image=self.photo1, anchor=tkinter.NW)

                self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(matte))
               
                self.canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)
                
             
    
                self.window.after(self.delay, self.run)
        else :
                
                self.out.release()
                #cv2.destroyAllWindows()                 

class MyVideoCapture:
    def __init__(self, video_source1):
        # Open the video source
        self.benchmark_cam = cv2.VideoCapture(video_source1)
        
        self.fps_time = 0 #Initializing fps to 0
 
        self.frame_counter = 0
        self.correct_frames = 0


        if not self.benchmark_cam.isOpened():
            raise ValueError("Unable to open video source", video_source1)

    @property
    def get_frame(self):
        ret1 = ""
        
        if self.benchmark_cam.isOpened():
            ret1, frame1 = self.benchmark_cam.read()
           
           
            if ret1 :
                            
               
                # Return a boolean success flag and the current frame converted to BGR
                return ret1, cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            else:
                return ret1, None
        else:
            return ret1, None
   

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.benchmark_cam.isOpened():
            self.benchmark_cam.release()
      
class BGRemove():
    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    """

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    
    if device == 'cuda':
        modnet = modnet.cuda()
    """

    def __init__(self, ckpt_path):
        self.parameter_load(ckpt_path)

    def parameter_load(self, ckpt_path):
        """
        BGRemove.modnet.load_state_dict(
            torch.load(ckpt_path, map_location=BGRemove.device))
        BGRemove.modnet.eval()
        """
        
        self.modnet = MODNet(backbone_pretrained=False)
        self.modnet = nn.DataParallel(self.modnet)
        
        
        
        if BGRemove.device == 'cuda':
            self.modnet = self.modnet.cuda()
        
        self.modnet.load_state_dict(
            torch.load(ckpt_path, map_location=BGRemove.device))
        self.modnet.eval()
        
        
        

    def file_load(self, filename):
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        return im

    def dir_check(self, path):
        os.makedirs(path, exist_ok=True)
        if not path.endswith('/'):
            path += '/'
        return path

    def pre_process(self, im):
        self.original_im = copy.deepcopy(im)

        # convert image to PyTorch tensor
        im = BGRemove.im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        self.height, self.width = im_h, im_w

        if max(im_h, im_w) < BGRemove.ref_size or min(im_h, im_w) > BGRemove.ref_size:
            if im_w >= im_h:
                im_rh = BGRemove.ref_size
                im_rw = int(im_w / im_h * BGRemove.ref_size)
            elif im_w < im_h:
                im_rw = BGRemove.ref_size
                im_rh = int(im_h / im_w * BGRemove.ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rh = 300
        im_rw = 250
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
        if BGRemove.device == 'cuda':
            im = im.cuda()
        return im

    def post_process(self, mask_data, background=False, backgound_path='assets/background/background.jpg'):
        matte = F.interpolate(mask_data, size=(
            self.height, self.width), mode='area')
        matte = matte.repeat(1, 3, 1, 1)
        matte = matte[0].data.cpu().numpy().transpose(1, 2, 0)
        height, width, _ = matte.shape
        if background:
            back_image = self.file_load(backgound_path)
            back_image = cv2.resize(
                back_image, (width, height), cv2.INTER_AREA)
        else:
            back_image = np.full(self.original_im.shape, 255.0)

        self.alpha = np.uint8(matte[:, :, 0]*255)

        matte = matte * self.original_im + (1 - matte) * back_image
        return matte

    def post_process2(self, mask_data, background=False, backgound_image=''):
        matte = F.interpolate(mask_data, size=(
            self.height, self.width), mode='area')
        matte = matte.repeat(1, 3, 1, 1)
        matte = matte[0].data.cpu().numpy().transpose(1, 2, 0)
        height, width, _ = matte.shape
        
        
        if background:
            back_image = backgound_image
            back_image = cv2.resize(
                back_image, (width, height), cv2.INTER_AREA)
        else:
            back_image = np.full(self.original_im.shape, 255.0)

        self.alpha = np.uint8(matte[:, :, 0]*255)

        matte = matte * self.original_im + (1 - matte) * back_image
        return matte

    def save(self, matte, output_path='output/', background=False):
        name = '.'.join(self.im_name.split('.')[:-1])+'.png'
        path = os.path.join(output_path, name)

        if background:
            try:
                matte = cv2.cvtColor(matte, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, matte)
                return "Successfully saved {}".format(path), name
            except:
                return "Error while saving {}".format(path), ''
        else:
            w, h, _ = matte.shape
            png_image = np.zeros((w, h, 4))
            png_image[:, :, :3] = matte
            png_image[:, :, 3] = self.alpha
            png_image = png_image.astype(np.uint8)
            try:
                png_image = cv2.cvtColor(png_image, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(path, png_image, [
                            int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                return "Successfully saved {}".format(path), name
            except:
                return "Error while saving {}".format(path), ''


# Create a window and pass it to the Application object
App(tkinter.Tk())