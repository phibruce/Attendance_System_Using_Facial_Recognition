#  🤖 ATTENDANCE SYSTEM USING FACIAL RECOGNITION TECHNOLOGY 🤖


## Build an automatic attendance system for students using facial recognition technology. Apply Haar Cascade algorithm and Local Binary Patterns Histogram method integrated in OpenCV. The system aims to increase efficiency and accuracy in attendance management, reduce cheating and save time for teachers and students.


###**Features**:
      1. Add new data: ID, Name, Images from real-time video.
      2. Attendance: Automately: Automatic attendance student via camera and display attendance history.
      
###**Algorithms & Method**:
      1. Haar Cascade: Use algorithms to detect faces in camera frames.
      This method uses Haar filters to detect facial features such as eyes, nose, and mouth. Haar filters are simple filters that are applied to each pixel in an image. Haar filters are designed to detect specific patterns, such as the outline of an eye or nose.
The way Haar Cascade works is based on using Haar-like features (i.e. simple geometric shapes such as rectangles and squares) to distinguish between object-containing and non-object-containing regions in an image.
The detection process using Haar Cascade works by sliding a fixed-size window across the entire image and applying a powerful classifier to determine whether that window contains an object. If the window is classified as containing an object, it is identified as an object-containing region and the detection result is returned.
      ![](https://miro.medium.com/proxy/1*m2UHkzWWJ0kfQyL5tBFNsQ.png)
      
      2.Local Binary Patterns Histograms: Use LBPH for features extraction and facial identification.
      
      Local Binary Patterns (LBP) is a feature extraction method in image processing. The extracted features will continue to be selected (feature selection) and reduced to a feature vector. This feature vector can then be used to input into a machine learning model for learning/classification.
LBPH (Local Binary Patterns Histogram) is a frequency graph built based on Local Binary Patterns. In other words, LBPH summarizes the distribution of local patterns extracted by LBP in an image region.
First, the LBP algorithm is applied to each pixel in the selected image region. This process creates a binary pattern for each pixel, describing the intensity relationship between the center pixel and its neighboring pixels.
Then, a histogram is built to store the frequency of each LBP pattern. The size of the histogram is equal to the total number of LBP patterns (usually depending on the number of bits in the binary pattern).
In short, with a face image, the system will convert it to grayscale to reduce complexity, then divide the image into many small squares and extract separate features on each square. For each square, the algorithm selects the center pixel, compares it with neighboring pixels, and assigns a binary value. Finally, combines the binary numbers of neighboring pixels to create an integer called the LBP code.
After calculating the LBP on all squares, the algorithm counts the frequency of the LBP codes and stores them in a histogram, and this histogram represents the features of the face.
When the system recognizes a face, the face is extracted using LBH and creates a local binary sample histogram. This histogram will be compared with the histograms in the database, usually calculated by Euclidean distance to find the face with the most similar features.
The LBP histogram acts as a feature representation for the selected image area. It summarizes the distribution of local textures in that region.
LBP for face recognition algorithms can simply insert new face samples without retraining, which is a clear advantage when working with face datasets where new face data is added or removed from the dataset with frequent frequency.
