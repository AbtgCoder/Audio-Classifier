# Deep Audio Classification

This repository contains code for deep audio classification, a deep neural network is constructed and trained to classify a forest audio recording and identify whether sound produced by Capunchin Bird is present in the audio or not. 
The data used for training and testing this model can be found [here](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing).
The trained model can be found [here](https://drive.google.com/file/d/1s6W6m2hANp62JsjNZoRQC1Br3uXF5sw8/view?usp=sharing)


## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python (version 3.6 or higher)
- Tensorflow 
- NumPy
- librosa
- Matplotlib

## Getting Started

Follow the steps below to set up and run the project:

1. Clone the repository:
   ```shell
   git clone https://github.com/AbtgCoder/Audio-Classifier.git
   ```

2. Navigate to the repository folder:
   ```shell
   cd Audio-Classifier
   ```

3. Train the model and see result:
   - Run the `visualize_result.py` file to preprocess audio data and create datasets, and train and visualize result of the deep audio classifier model.
   ```shell
   python visualize_result.py
   ```


## File Descriptions

- `model_train.py`: Deep Audio Classifier implementaion in Tensorflow.
- `visualize_result.py`: Preprocess audio data, train and visualize result of model.
- `model_eval.py`: Generate and evaluate result on test dataset using the trained model.
- `result_saliency_map.png`: Sample result of model.
- `audio_class_model.h5`: The trained model can be found [here](https://drive.google.com/file/d/1s6W6m2hANp62JsjNZoRQC1Br3uXF5sw8/view?usp=sharing).
  

## License

[MIT License](LICENSE.txt)

The project is open source and released under the terms of the MIT License. See the [LICENSE](LICENSE.txt) file for more details.

## Contact

For any questions or inquiries, you can reach me at:
- Email:  [abtgofficial@gmail.com](mailto:abtgofficial@gmail.com)

