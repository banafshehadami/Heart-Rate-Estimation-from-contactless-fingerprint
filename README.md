In this project, we are trying to detect whether an image of a finger
is real or fake. To do so, we are using the PhyNet architecture to extract the rPPG signal. 

rPPG, or remote photoplethysmography, is a technique for measuring physiological signals such as heart rate 
and respiration rate by analyzing subtle changes in skin color that occur with each heartbeat. 

This is the brief overview of how the algorithm works:
1-Read the image, and Crop the RGB image to 128 * 128.
2-Read each image 100 times to create a video with 24 fps.
3-Make .h5 file for each video.
4-Use PhysNet to generate an rPPG block (spatio-temporal).
5-calculating the PSD of the signal and apply a filter on that to achieve the heart rate.
Notes: We made some changes to the architecture PhysNet by applying the Swin Transformer.

Score:
The score ranges from 0 to 100. A score closer to 0 indicates a higher probability that the image is fake,
while a score between 50 and 100 indicates a higher probability that the image is real.
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--
