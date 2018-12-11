# Behavioral Cloning
[//]: # (Image References)

[image0]: ./examples/track1_scene.jpg "Scene in Track1"
[image1]: ./examples/track2_scene.jpg "Scene in Track2"
[image2]: ./examples/track2_trining_curve.png "Track2 training"

How to Run This Repository
---
This respository contains the following modules to demonstrate autonomous driving through behavior cloning from video of driving .
* dataLoader class
  * Data are collected from the [simulator](https://github.com/YihsunEthanCheng/self-driving-car-sim), to be stored in the [data/track1](./data/track1) or [data/track2](./data/track2) folders locally.
  * The data class provides following functionalities essential to the training modules.
    * Batch generator to enable epoch training in keras.
    * Collection of validation set for monitoring overtraining.
    * Randomized training batches.
    * Data augmentation
        * Each image/steering signal pair are flipped horizontally half of the time. 
  * The dataLoader can be initiated to load a data set from a folder.  In particular, if track #1 or #2 data are to be loaded, the class can quickly load and encapsulate the data as an object.
    ```sh
    trackID = 1
    data = behaviorCloneData('data/track{}'.format(trackID))
    ```

* Deep network model
  * The model for cloning driving behavior is chosed as a "4-conv + 3-FC network", which can be created by the python module [model.py](./moduals/model.py/) with a specified parameter set as below.
    ```sh
    params = {
        'input_shape':  (160,320, 3),
        'cropping': ((65,20), (0,0)),
        'nFilters': [24, 36, 48, 60],
        'convDropout' : 0.1,
        'nFC': [128, 16],
        'fcDropout': 0.4,
        'batch_size' : 128
        }
    ```
  * [model.py](./modules/model.py) contains the function to create the deep network model specified by the parameter "dict" above as,
    ```sh
    > model = createModel(params)
    ```
  * [model.py](./modules/model.py) also contains a simple training interface to take the "dataLoader" object and train for a specified number of epochs.
    ```sh
    > train(model, data, nEpoch, "trained_model/model.h5")
    ```
  * The above simple function and class allow us to create quick training scripts. Training scripts below used to clone the driving behavior serve as examples.
    * [run_model_train_track1.py](./run_model_train_track1.py)
    * [run_model_train_track2.py](./run_model_train_track2.py)   


* Trained models
  * [Trained models](./trained_models) are available for both tracks presented in the [simulator](https://github.com/YihsunEthanCheng/self-driving-car-sim)
  * To run the trained model live for track #1, use the following trained models.
     ```sh
    > python drive.py ./trained_models/track1/24-36-48-60_128-16_ep25.h5
     ```
  * To run the trained model live for track #2, use the following commands.
     ```sh
    > python drive.py ./trained_models/track2/32-48-64-80_256-32_ep50.h5
     ```

* Utitlity functions 
  * drive.py: take a keras model and send steering commands to the simulator with respect to the scene received from the simulator.
  * video.py: convert oupout image sequences into a mp4 video.

---
The Deep Network Architecture
---
The same deep network archtiecture used in both tasks with the difference in the number of kernels/neurons. Track #2 task is harder, thus needs a larger scale of deep network.

The network architecture can be described as a "4-convolution + 3-FC" layers of deep network, which is adopted from the Nvidia research report as an adequate starter deep network. The following highlights my customiztion of each layer.

* Normalization layer
  * Scales inputs to [-1,1]
* cropping layer
  * Removes distractions in the top/bottom rows of the inputs
* Conv#1, #2
  * 5x5 kernels
  * 2x2 max pooling
  * Relu 
  * batch normalization
  * 0.1 dropout
* Conv#3, #4
  * 3x3 kernels
  * 1x1 max pooling
  * Relu 
  * batch normalization
  * 0.1 dropout rate
* flatten layer
* FC #1, #2
  * batch normalization
  * 0.4 dropout rate
  * Relu
* FC #3
  * single output as steering control
  * Tanh activation to clip output within [-1,1]

* A Keras' model summary for Track #2 task.

  ```sh
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lambda_1 (Lambda)            (None, 160, 320, 3)       0         
    _________________________________________________________________
    cropping2d_1 (Cropping2D)    (None, 75, 320, 3)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 75, 320, 32)       2432      
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 75, 320, 32)       128       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 37, 160, 32)       0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 37, 160, 32)       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 37, 160, 48)       38448     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 37, 160, 48)       192       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 18, 80, 48)        0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 18, 80, 48)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 18, 80, 64)        27712     
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 18, 80, 64)        256       
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 9, 40, 64)         0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 9, 40, 64)         0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 9, 40, 80)         46160     
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 9, 40, 80)         320       
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 4, 20, 80)         0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 4, 20, 80)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 6400)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 256)               1638656   
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 256)               1024      
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 32)                8224      
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 32)                128       
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 1,763,713
    Trainable params: 1,762,689
    Non-trainable params: 1,024
    _________________________________________________________________

  ```
---
Data collection
---
 The success of training is strongly tied up to the diversity of training data set.  To have a successful training session, the following summarizes my data collection strategy.

 * At least two laps of driving in each direction, this results in 4 laps of data for each track (8x data size by data augmentation through horizontal flipping).
 * Data cleansing
    * Data collection are often interrupted due to failures during collection. Withoug removing the failure data, the network often latches on bad driving behavior at certain locations.
* Additional data collection at certain difficult spots could avoid recollection of the entire track.
* Abundant data are keys to the success to difficult tasks. To train track #2 task, I have collected more than 6 laps of training data with over 400mb of video stream.  

---
Training
---
* Training progress monitoring
    * Below shows a training session through 50 epochs.

     ![alt text][image2]

* Overtraining prevention
  * dropouts are used in every convolution and fully connected layer with a lower rate at convolutional layer (0.1) vs. fully connected layers (0.4).
  * With the depth of the network, overfitting is likely to happen. Even with dropout to regularize the training, we can still see overtraining from the plot above. In the plot, the validation error begins to flatten out after 10 epochs while the training error continue to drop.
  * The overtraining appears in the live testing sesssion.  One can see the jittering tire movements during the driving session while a non-overfitted model shows smooth driving behavior. After testing, I found a 50 epochs in our data collection gives the best performance.

* batch normalzation
 * Like dropout, batch normalization is installed in every layer with parameters. It is known to stabilize and speed up the training by removing the scaling and bias in each minibatch. This helps the network to focus and quickly finish the training no more than 50 epochs.


Results
---
* Track #1
    * The deep network works seemlessly in track #1 with just 20 epochs of training. The successful compeletion of track #1 can be found at [track #1 video](./examples/track1_output_video.mp4)

  | ![alt text][image0] | 
  |:--:| 
  | *Scene of autonomous driving in track #1* |


* Track #2
  * The netowrk seems to have no problem handling the task as well except it is stuck near the very last end as you can see it at [track #2 video](./examples/track2_output_video.mp4)

  | ![alt text][image1] | 
  |:--:| 
  | *Scene of autonomous driving in track #2* |


---
Takaways
---
* The 4-conv + 3-FC deep network structure is very capable of handling the two tasks as it appears to show successful completion of track #1 and near successful completion of track #2 in my tests.
* The incompletion of track #2 is caused by the lack of training data at the stuck location. During the data cleansy stage, I had over trimmed the data due to the "backing-up attemps" to resume the driving.  Should I recollected data at the failure spot, the completion of track #2 is expectable.
* Despite incompletion of track #2 task, the driving behavior shows amazing skills in handling the difficulty in this track after only a few spochs of training. This leads me to believe the success is achievable with proper data.
* Overtraining is an issue for deep network.  More aggressive regularization might prevent hitting stuck in local minimum but may not be effective. In my opinion, mor sophisicated early stopping might be the key to the success of training.
* In some overtrained network, the vehicle appears to often latches on the opposite lanes. The observations tell me how important the side cameras are to stay in the right lane. Personally, I believe a multi-agents colloaboration would be a better solution than burdening an agent with two other cameras having different field of views.

