# Encoding Models for the Higher Visual Areas

*Jonas Kubilius and Drew Linsley*

*TAs: Leyla Isik and Alex Kell*

*PI: Nancy Kanwisher*

**NOTE: Below is an unfinished report of a project we did at the [CBMM Summer School 2015](http://cbmm.mit.edu/summer-school/2015). Please let us know if you intend to use any of the materials in this repository.**

## Introduction

Humans are able to gather a rich description of their surroundings through a single glance. For instance, driving through downtown Boston may require an individual to not only accurately perceive the types of objects around them, but also to quickly estimate the intentions of pedestrians and other drivers. While great steps have been taken to untangle the mechanisms underlying object perception, investigations into high-level visual understanding have yielded inconsistent results. A potential reason for this non-success is the typical approach to measuring the human visual system, in which brain regions are tested for their responses to a relatively small set of systematically selected stimuli. However, this approach appears unfit, as high-level visual understanding is tightly linked to the task at hand and spans a large feature space. Another approach has emerged from a growing body of research, in which videos are used to interrogate responses in visual areas enabling researchers to more effectively sample from the feature space of interest. Thus, in the current study we adopted this data-driven approach to identify regions involved in high-level visual understanding using deep neural networks. We demonstrate that deep nets can serve as a valuable encoding model for fMRI activation in higher visual cortex.

## Materials and Methods

### fMRI data collection

One subject (male/female, aged X years) viewed sections of the movie *Home Alone 2* while being scanned with a 3T Siemens Trio (?) scanner. During each scanning session, T2* weighted scans sensitive to blood oxygenation level-dependent (BOLD) contrasts were acquired using a gradient-echo echo-planar pulse sequence (??? TR = 3000ms, TE = 30ms, voxel size = 3x3x3mm, matrix size = 64 x 64 x 45). Initially, the subject was twice scanned over two days while watching the full movie. However, technical issues prompted the subject to complete an addition four scanning sessions in which the subject watched the same excerpt lasting approximately 20 minutes. All analyses were performed on these four scanning sessions. The participant passively watched the movie during each session and did not complete any task.

### fMRI preprocessing

Brain volumes gathered during scanning were preprocessed with FSFAST to remove motion artifacts and transform to a flattened cortical surface. These volumes were then passed through MATLAB scripts that detrended and normalized each to 0 mean and unit variance. Voxel timecourses were shifted 4 seconds (2 TRs) back in time to correct for hemodynamic response lag. This was done by removing the first two TRs of each voxel timecourse then adding two TRs of mean signal to the end. These procedures were performed separately for each voxel within each scan run.

### fMRI analysis

In contrast to typical neuroimaging experiments in which subjects view static and tightly controlled stimuli, the subject in this experiment viewed *Home Alone 2*, a popular adventure movie from 1992 featuring a famous lead actor. In order to understand how brain activity related to the action on the screen, we adopted an encoding model approach that has performed well under similar circumstances. In this approach, linear models are fit to each voxel’s timecourse to estimate the relationship between the voxel’s activity and a stimuli features. Key to this type of supervised approach is an overcomplete set of annotations describing the action on screen. Rather than hand-label stimuli as in similar approaches, we opted to instead utilize convolutional neural networks (CNN) to produce automatic annotations for each scene.

### Convolutional neural networks

CNNs have been successfully applied in a variety of computer vision approaches, identifying objects, places, and other features of real-world images with accuracy that at times approaches human performance (CITATION). Thus, we reasoned that movie annotations produced by these models could provide reasonable matches to human perception.

In order to capture as much perceptual variability in the movie as possible, we used four CNNs that were pretrained to identifiy non-overlapping visual features. Each model was trained on the AlexNet CNN architecture and instantiated with the Caffe Library. These models were:

- “Objects”, trained to identify the objects within a scene. The experimenters reduced the number of object categories from 978 to 30 (?) to simplify the task to identify the basic-level object category.

- “Places”, trained to identify the place depicted by a scene. The experimenters reduced the number of place categories from 205 to 21 (?) to simplify the task to identify the basic-level place category.

- “Visual style”, trained to identify a movie frame’s style as being e.g. geometric or vintage. The style was chosen from a set of 20 categories. This model was created by removing the softmax layer of the Objects model and replacing it with a set of labels indicating the visual style of images gathered from Flickr.

- “Object count”, trained to count the number of salient objects in an image. Counts produced by this model were zero, one, two, three, or four or greater. As in the visual style model, this model was created by removing the softmax layer of the Objects model and replacing it with a set of labels indicating the number of salient objects in each image.

These models were used to annotate the entire *Home Alone 2* movie, which was recorded at 23.98 frames per second. In order to align these annotations with the slower fMRI acquisition timecourse, we only used annotations corresponding to the frame presented in the middle of a scan acquisition. The resulting feature matrix was used to construct models for each voxel, which is described in further detail below.

### Encoding models

1. Find relationship between image features and voxel activity (regularized L2)

2. The resultant set of weights form an “encoding model” with an estimate of this relationship.

3. The encoding model is applied to a held out set of data, producing an estimate of the voxel time course that it believes would result from the held out features.

4. We correlated this “synthetic timecourse” with the actual time course to evaluate model fit. Further calculating the fraction of variance that this synthetic timecourse captured from the estimated prediction ceiling yielded the % variance captured.

Specifically, we first split the fMRI dataset into 481 timepoints for obtaining model fits and the remaining 50 timepoints that we held out for validation of model fits. Note that 3 time points in between the training and validation set were removed in order to minimize potential carry-over effects from the training set to the validation set. Next, in a ten-fold cross-validation procedure, we selected the optimal regularization parameter that would result in the most accurate model fit. Regularization parameters were chosen from a list of the following values: … Finally, the optimal regularization parameter was used to obtain the weights of the model fit.

## Results

### Predictibility of different convnets

![](presentation/img/preds_rois.png)

![](presentation/img/preds.png)

### Decoding

![](presentation/img/decoding.png)

### ROI tuning properties

![](presentation/img/tuning.png)
