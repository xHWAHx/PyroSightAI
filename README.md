Collaborators: Hamzah Hamad, Hussam Al-Nabtiti

Introduction and background:

In fire safety, early detection is crucial to minimizing damage and saving lives. Most ‘modern’ fire-detecting technologies, such as fire alarms, date back to the 1800’s. While proven beneficial, their ability to detect fires usually comes after a fire has intensified and caused major damage to property and possibly lives. This project proposes an AI-based fire detection system that uses real-time video feed analysis to detect and classify fires at various levels (small flames to large). The system will classify fire intensity, hypothetically trigger sprinklers at an appropriate threshold, and send initial and regular updates to the owner of the property allowing for immediate, targeted responses.

Objectives: 

  The primary objectives of the project are: 

    Develop a computer vision model, using Convolutional Neural Networks (CNNs), to effectively classify fire intensity in real-time through a camera feed.
    Design a threshold system for different fire levels triggering automated sprinklers and  sending notifications to the owner. 
    Evaluate the system’s effectiveness in accurately detecting fire stages and responding quickly to potential hazards.    

Proposed Methods for Artificial Intelligence:

In this project we will leverage the Convolutional Neural Networks (CNN) model, designed and trained to classify fire levels based on image data. 

The core components include but are not limited to: 

  Image Classification Model with CNNs 
      
      Our team developed a CNN model trained on images, obtained from open source data providers such as Kaggle, classified by fire intensity (no fire, small fire, medium fire, large fire). 
      
      The CNN model will process live data from the camera feed and classify it according to one of the four intensity levels. 

  Threshold-Based Triggering System 
      
      This system defines a set of thresholds for fire intensity levels based on the classification outputs. For example, if the model detects ‘large fire’ it automatically triggers the sprinklers, sends notifications to the owner, and signals emergency responders. 

  Real-Time Processing 
      
      We integrate live camera feed with our CNN model for demonstration.

Dataset used: 

  Open-source Fire Datasets: Existing labelled fire image datasets from platforms like Kaggle.
  
  Simulated Data: Synthetic data such as a simulated fire video. 

Novelty of the project:

The novelty of this project is found in connecting the computer vision system with real-time safety response mechanisms. Unlike sprinklers that rely on heating gases, or smoke detectors that wait for a plume of smoke to arise, we can provide quicker and more reliable safety responses through real-time assessment of visual cues and immediate location-specific safety response. The integration of classification and automated sprinkler activation will offer a new level of fire safety by minimising fire damage through early responses, whilst also reducing costs as opposed to other thermal imaging alternatives and updating property owners with live notifications. The system will also ensure timely and automated emergency service contact with the nearest fire stations.

