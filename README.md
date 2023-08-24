<h1>YOLO Computer Vision Project</h1>
This document serves as a guide on how to train and test this Ganzin - Pupil Tracking model using YOLO (You Only Look Once) computer vision system.

<h2>Training Process</h2>

**Execute the following command:** ```python3 training.py -y {your yaml file} -m {your output model directory}```

The training process follows these steps:

**Step 1:** The first step involves setting up the yaml file, which should contain the following fields:

* `path`: This refers to the directory where your final project resides.
* `train`: This specifies the directory where your training data is stored.
* `val`: This is the directory for your validation data.
* `test`: This should be the directory for your testing data.
* `nc`: This represents the number of classes. In our case, it should be set to 1 as we are only classifying one class, which is the 'pupil'.
* `names`: This is simply the name of our class, which in this instance, we have named "pupil".

**Step 2:** We utilize our yaml file to construct a new YOLO v8 model from the ground up.

**Step 3:** We load the pretrained model, specifically ***yolov8n-seg.pt***.

**Step 4:** We establish the training parameters, which include settings such as epochs, patience, batch size, among others.

**Step 5:** We evaluate the model's performance using a validation set.

**Step 6:** We export the trained model as an ONNX file.
<h2>Training Result</h2>
The YOLO segmentation model will automatically generate a directory named "runs". Inside "runs", there is a directory called "segment" which contains two further directories labeled "pupil_tracking" followed by a number.

In the first "pupil_tracking" directory, a subdirectory named "weights" will contain the saved training weights (pt files) from each set of 25 epochs. Additionally, this directory also contains various files representing different training results, such as the confusion matrix, results CSV file, and training batch results.

In the second "pupil_tracking" directory, you will find files with the same names, but they will contain validation results.  

<h2>Testing Process</h2>
**Execute the following command:** ```python3 testing.py -o {your output directory} -m {your model pt file} -t {your organized test data}```

The testing process comprises of these steps:

**Step 1:** We utilize YOLO's native "predict" function to generate the model testing result. To avoid memory overflow errors, it's crucial to set the ***stream*** parameter to ***True***. After setting ***stream*** to ***True***, the function will automatically return a generator, so a for-loop is used to output the result.

**Step 2:** Next, we sketch the resulting pupil by employing ***sketching.py***, which is invoked within ***testing.py***. This function utilizes the mask and boxes function from Ultralytics to draw the expected outcome. It's vital to remember that some images might not include pupils, leading to the return of ***None***. To handle these instances, we implement a try-catch block. Ultimately, the sketching results are automatically saved in a file named "result".

<h2>Pretrained Model:</h2>
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt

# Pupil-tracking
