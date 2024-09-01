# ml-proj - Detecting and Classifying Musical Instruments with Convolutional Neural Networks

Students: Anđela Bašić, Luka Radanović

Link to original report: https://cs230.stanford.edu/projects_winter_2021/reports/70770755.pdf

Keywords: multilabel classifier, musical instrument, audio,  transfer learning, CNN

Link to .wav dataset: https://www.upf.edu/web/mtg/irmas

Dataset is already split into training and testing. We further performed a 90-10 split of the training set to obtain the validation set. 
Raw training, validation and test set contain .wav files that we later transform to images. Regarding labels, training and validation instances are labeled by a single instrument which is included in the file name. Test set instances can be labeled by more than one instrument and their labels are saved in the corresponding .txt files.
The original training set (training + validation) consists of 6705 3-second audio files, while the test set consists of 2800 stereo .wav files, each of which is between 5 and 20 seconds.

Instrument classes: cello, clarinet, flute, acoustic guitar, electric guitar, organ, piano, saxophone, trumpet, violin, and human singing voice.
Samples are from a wide ranges of genres (from pop to classical).

Machine learning library: pytorch 

Project Description: Implementation of single/multi label (multi-class) musical instrument classifiers
that determine which musical instruments are present in audio files. There are 11 possible musical instrument classes.
We closely followed the work of Dominick Sovana Hing and Connor Joseph Settle from Stanford Univerity linked above consisting of following steps:

1. Preprocessing
   
   1. Data Augmentation - splitting original files into 3-seconds segments and doubling the number of files by channel swapping
   2. Audio to Image - extracting melspectograms from audio files with librosa
   Note: We used utils.py, irmasTestUtils.py, irmasTrainUtils.py for steps 1.1 and 1.2 provided by fellows from Stanford. :)
   
   Notebook: 01_DataPreprocessing.ipynb

3. Single Label Classifier
   
   Implementation of a base CNN described in the original paper. We performed manual parameter tuning which resulted in obtaining a slightly different model that is later    
   used for transfer learning. Resulting model has slightly worse accuracy but yields better results in overfitting analysis. 

   Notebook: 02_SingleLabelClassifier.ipynb (and library classifier.py)
   
4. Multilabel Classifier
   
   We added 3 dense layers to the single label model and used pre-trained weights from single classifier to obtain the final multilabel model. We tuned parameters on   
   validation set. Finally, we performed testing on the test set to obtain the relevant metrics.

   Notebook: 03_MultilabelClassifier.ipynb (and library classifier.py)

6. Demo
   
   Short demo for a quick overview of model outputs. Input a path to a song and model should output a list of 1s and 0s representing included/excluded instruments.

   Notebook: 04_Demo.ipynb

Note: Classifiers are saved separately in: classifier.py. Helper functions regarding training, validation, parameter-tuning and testing are saved in classifier_util.py. Required libraries and versions can be found in requirements.txt.

