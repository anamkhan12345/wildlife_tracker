# Small Bird Detection Project

Hi! This is my project to detect birds from my apartment window. 

## Problem Overview

I am trying to detect birds both in motion and stationary / perched from my apartment window. These birds could be upto 70 meters away or as close as 1 meter. However, I am trying
to detect them from only one angle and one environment. I want to be able to detect multiple birds as well in a frame.

I also want a live web dashboard that will show the most recent detection, largest detection, total detections, and an average detection per hour of the day. 

## Architecture Overview

![Overall Architecture](./architecture.drawio)


### Datasets

After looking through multiple datasets online from sources like roboflow, I opted to create my own custom dataset. I'm doing so because I think I'll be able to create a higher
quality dataset that is fine tuned to my environment and the objects I'm interested in. This has forced me to learn classic computer vision methods, such as applying masks, Mixture of Gaussians (MoG),
temporal filtering, and adaptive learning to detect objects in motion. I then parsed the dataset to include a certain amount of negatives (humans, dogs, airplanes, helicopters) and
birds. I then used roboflow to get as precise annotations and bounding boxes that I could on those detections. I initially tried to automate the annotation immediately after 
detection - but the quality was poor. It took me multiple rounds of exploratory data analysis (EDA) to try to cultivate a dataset that accounted for varying weather (raining, cloud cover, sunny)
,size of detections, number of detections and more. Early experimentation showed overtaining and a bias for single perched birds. Cultivating the dataset was easily
the hardest part and took most of my time. No matter how good your CNN or model is, it doesn't matter if your dataset is crap. 

### Hardware

I wanted to keep this as low budget as possible, so I used my existing Raspberry Pi 5 8 GB. I used 2 camera, a basic webcam and the RPI HQ camera. I got the HQ camera in an attempt 
to get higher quality images. In setting up my camera I realized that the seller sent me the C to CS adapter and the camera could not zoom out at all. This took me way too long to
figure out. I was able to find one on Amazon for cheap and the camera worked perfectly fine. 

### Software / Models

All of my code was in python to take advantage of its robust computer vision and ML libraries. I opted to use YOLOv11 for transfer learning. I trained my model in a Kaggle Notebook
and then exported it as a YOLO NCNN model, which is better suited to run on smaller devices like my RPI. 

I used a SQLite database that that contained information about the detection. This database was stored locally on my RPI. I initially stored my images on the cloud via Google Cloud Storage, just to learn how to use the API
and the cloud. It worked, but realized saving the images locally was easier and faster for the limited amount of detetions I was making per day. 

I used streamlit to create my dashboard. It parse my SQLite database to find the latest image, largest image and other metrics. It then used the filepath or Goole Cloud Storage URL
to show images of interest. 
