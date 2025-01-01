Traffic Sign Classification - Team Alfa
by: Nirmal Raja Karuppiah Loganathan, Puja Ammineni, Poojith Mendem, and Deva Priya Mankena

The project focuses on developing a system capable enough for accurate classification
of traffic signals

Training and Model Building - FinalCodeAlfa.ipynb

Data:
Download the dataset from - https://drive.google.com/file/d/1AZeKw90Cb6GgamTBO3mvDdz6PjBwqCCt/view
and change the path of the data directory
* It contains various traffic sign images that are classified into different folders according to their classes
* For preparing data we used Img2Vec extracts the feature from images and the features are saved as "features.npy" and labels are saved as "labels.txt"

Feature Extraction and Visualization 
* We employed PCA and to reduce the dimensionality of the extracted features to two dimensionalities for visualization 

Model Training and Evaluation 
* Several classifiers were trained and evaluated by using stratified k-fold cross validation and those classifiers are Decision Tree, Random Forest, Gaussian Naive Bayes, Multi-Layer Perceptron, Support Vector Machines
* Performed evaluation metrics such as accuracy, precision, recall, and F1 score we estimated classifiers to assess their performance

Model Development 
* Based on Model Training and evaluation the SVM Classifiers achieved the highest accuracy
* The model was serialized by using a pickle and saved as a model.p


Testing sample images - testing.ipynb

Dependencies
* The dependencies we had used in Our Project
* NumPy
* img2vec-pytorch
* pillow
* sklearn
* matplotlib
* seaborn
* xgboost
* lightbgm



