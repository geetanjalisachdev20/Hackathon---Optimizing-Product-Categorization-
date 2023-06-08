# Hackathon---Optimizing-Product-Categorization-
**Project Summary: Optimizing Product Categorization in Criteo – ESCP Hackathon x Criteo**

This project summary provides an overview of the work completed during the Criteo x ESCP Hackathon, where I led a team of five members to optimize product categorization for Criteo. The main objective was to develop a methodology that could automatically map millions of consumer products into a Universal Product Catalogue for Criteo.
Our methodology demonstrated innovation and creativity in tackling the problem. As a result, they awarded the Most Innovative Methodology Prize for our exceptional creative approach to problem-solving

In the project, two main datasets were utilized: the bulk dataset and the golden dataset. Here is some additional information about each dataset:
**Bulk Dataset:**
•	Size: The bulk dataset was over 6 GB in size, indicating a large volume of data.
•	Rows: It contained more than 4.2 million rows, indicating a substantial number of entries.
•	Columns: The dataset consisted of 201 columns, representing various attributes and characteristics of the consumer products.
•	Labeling Accuracy: The labeling accuracy of the bulk dataset was unknown, which posed a challenge in correctly classifying and mapping the products.
**Golden Dataset:**
•	Size: The golden dataset was comparatively smaller than the bulk dataset.
•	Rows: It contained less than 200,000 rows, indicating a reduced volume of data compared to the bulk dataset.
•	Columns: Similar to the bulk dataset, the golden dataset also consisted of 201 columns, providing the necessary attributes for product classification.
•	Labeling Accuracy: The labeling accuracy of the golden dataset was correct, ensuring that the products were accurately labeled.

The challenges posed by the large size and unknown labeling accuracy of the bulk dataset were addressed through data preprocessing, hyperparameter tuning, and the utilization of transfer learning techniques. The golden dataset, with its correct labeling, served as a reference and training ground for the development of the final transfer learning neural network model.
Developed a Multi-Class Neural Network model using Keras and PyTorch. Extensive research was conducted to determine the optimal number of neural network layers and carefully select hyperparameters to maximize the model's performance and accuracy in product classification and mapping.
The project involved working with a bulk dataset and a golden dataset. The bulk dataset was initially too large, so various sampling techniques were used to reduce its size while preserving important characteristics. The sampling approaches included random sampling, stratified sampling, indexing, PCA and incremental PCA.Finally, bulk data was iterated over each label and sample data based on the label count. If the count of a label is less than 10, we sample all available data points for that label. Otherwise, we sample 10% of the label count. We also check that the sample size is greater than 0 to avoid errors when sampling a label with very low count. Finally, we concatenate the sampled data for all labels, check that all unique labels are present, and save the sampled data to a file.

The chosen approach involved leveraging transfer learning, wherein a first neural network model was trained on the sampled bulk dataset. This model was then used as a starting point to develop a transfer learning neural network. 
A neural network classification model was then trained on the sampled bulk dataset. The labels were encoded as integers, and the data was converted to PyTorch tensors. The model architecture consisted of three fully connected layers. The network was trained using the cross-entropy loss function, RMSprop optimizer, and a learning rate scheduler. The accuracy and loss were tracked during training.
After training the classification model on sampled bulk dataset, transfer learning was performed on the entire golden dataset. The pretrained model's weights were loaded, and a new network architecture was defined to incorporate the pretrained layers. The pretrained layers were frozen, and only the new layers were trained. The model was trained using the same loss function, optimizer, and learning rate scheduler as before.
The transfer model was applied to the whole golden dataset(with labels) and also to final new dataset(without labels)  provided by Criteo to predict the labels of the consumer products.

The task also involved predicting correct lables for the new dataset(without labels) that was provide one day to evaluate the accuracy of the model.
As the new dataset was supposed to be given one day before the presentation so for our reference to check the accuracy of the model trained. The approach followed we followed was to split the golden dataset was split into 80% training and 20% testing sets. For the 20% of the golden dataset the embeddings and labels were separated and saved as separate files named as “golden _embeddings” and “golden_actual_labels” respectively. And trained the transfer model on the 80% of the golden dataset and used to predict the labels for the 20% of the golden dataset (golden _embeddings). The predicted labels were then saved as “golden_predicted_labels.” Then the actual and predicted labels were examined to test the reliability of the model.
In conclusion, the project successfully optimized product categorization for Criteo by developing an advanced methodology that exceeded expectations and stood out among the competition. The use of machine learning algorithms, neural network architecture, and transfer learning techniques showcased expertise and dedication to achieving outstanding results.


