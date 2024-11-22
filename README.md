# Hindi Image Caption Generator 📸  

**An AI-powered application to generate image captions in Hindi, promoting inclusivity, accessibility, and cultural representation.**

---

## **Table of Contents**  
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Technologies Used](#technologies-used)  
4. [System Architecture](#system-architecture)  
5. [Models Implemented](#models-implemented)  
6. [Dataset](#dataset)  
7. [Setup and Installation](#setup-and-installation)  
8. [Usage](#usage)  
9. [Screenshots](#screenshots)  
10. [Team](#team)  
11. [Future Scope](#future-scope)  
12. [Contributing](#contributing)  
13. [License](#license)  

---

## **Project Overview**  
India is a diverse country with Hindi being the most spoken language. Yet, most AI applications are designed for English, leaving millions underserved. This project bridges the gap by enabling image captioning in Hindi.  

### **Why Hindi Captioning?**  
- **Inclusivity**: Makes AI accessible to non-English speakers.  
- **Accessibility**: Helps visually impaired users understand images with Hindi descriptions.  
- **Cultural Representation**: Integrates native language and culture into modern AI solutions.  

---

## **Features**  
- Upload an image and generate captions in **Hindi**.  
- Choose from multiple advanced machine learning models:  
  - Without Attention  
  - With Attention  
  - Multi-Head Attention (MHA)  
  - BERT-based Captioning  
  - Transformer-based Captioning  
- Toggle between **English** and **Hindi** for project details.  
- Display generated captions alongside the uploaded image.  

---

## **Technologies Used**  
### **Frontend**  
- [Streamlit](https://streamlit.io/): Interactive web application for UI development.  

### **Backend**  
- [TensorFlow](https://www.tensorflow.org/): Deep learning framework for training and inference.  
- [InceptionV3](https://keras.io/api/applications/inceptionv3/): Pre-trained model for image feature extraction.  
- [Pillow](https://python-pillow.org/): Python Imaging Library for image processing.  

### **Languages**  
- Python  

---

## **System Architecture**  
The system architecture includes the following steps:  
1. **Image Upload**: User uploads an image via the Streamlit UI.  
2. **Feature Extraction**: The uploaded image is processed using InceptionV3 to extract features.  
3. **Caption Generation**: The features are passed through the selected deep learning model to generate captions.  
4. **Caption Display**: The generated caption in Hindi is shown to the user, alongside the uploaded image.

```

```
##**Models Implemented**
1. **Without Attention**: Traditional image captioning model without attention mechanisms.
2. **With Attention**: Uses attention mechanisms to focus on specific parts of an image while generating captions.
3. **Multi-Head Attention (MHA)**: A more advanced version of attention, helping the model to learn multiple aspects of the image.
4. **BERT-based Captioning**: Utilizes BERT for contextual understanding and generates captions in Hindi.
5. **Transformer-based Captioning**: Uses Transformer architecture to generate more accurate captions.

```
```
##**Usage**

Upload an image in jpg, jpeg, or png format.
Select the model for caption generation from the dropdown menu.
The app will process the image and display the generated Hindi caption.
```

```
##**Future Scope**
**Expand Language Support**: Implement multi-language support for captions.
**Improve Model Accuracy**: Fine-tune models for better accuracy and diverse captioning.
**Real-time Image Captioning**: Enable real-time captioning for live streams or video feeds.

```

```
##**Contributing**
We welcome contributions! To contribute to this project:

Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes.
- Commit your changes (git commit -am 'Add new feature').
- Push to the branch (git push origin feature-branch).
- Create a new pull request.
```
