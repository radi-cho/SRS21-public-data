import cv2
import os

FRAMES_PER_VIDEO = 4
FPS = 1
MAX_VIDEOS = 20000

class VideoGenerator():
    def __init__(self):
        self.image_folder = 'movingmnistdata'
        self.video_counter = 1
        self.image_counter = 0

    def gen_video(self):
        if not os.path.isfile(os.path.join(self.image_folder, f'{self.image_counter}.jpg')):
            print('THE END')
            return True

        if self.video_counter > MAX_VIDEOS:
            return
        
        video_name = f'video{self.video_counter}.mp4'

        frame = cv2.imread(os.path.join(self.image_folder, '0.jpg'))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, FPS, (width,height))

        for x in range(FRAMES_PER_VIDEO):
            video.write(cv2.imread(os.path.join(self.image_folder, f'{self.image_counter}.jpg')))
            self.image_counter += 1


        cv2.destroyAllWindows()
        video.release()

        self.video_counter += 1
        self.gen_video()
        
generator = VideoGenerator()
generator.gen_video()
