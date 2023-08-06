# Data from : https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing

import os
import matplotlib.pyplot as plt
import tensorflow as tf

import scipy.signal as sps
import librosa


# Set GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.set_logical_device_configuration(
        gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=6000)]
    )


def load_wav_16k_mono(filename):
    # convert sample_rate: 44100Hz to 16000Hz,
    new_rate = 16000
    wav, s = librosa.load(filename.numpy(), sr=new_rate)
    # wav, s = librosa.load(filename, sr=new_rate)

    
    return wav


CAPUCHIN_FILE = os.path.join("data", "Parsed_Capuchinbird_Clips", "XC3776-3.wav")
NOT_CAPUCHIN_FILE = os.path.join(
    "data", "Parsed_Not_Capuchinbird_Clips", "afternoon-birds-song-in-forest-0.wav"
)

POS = os.path.join("data", "Parsed_Capuchinbird_Clips")
NEG = os.path.join("data", "Parsed_Not_Capuchinbird_Clips")

# Create Tensorflow Datasets
pos = tf.data.Dataset.list_files(POS + "\*.wav")
neg = tf.data.Dataset.list_files(NEG + "\*.wav")

# Add labels, Combine all samples
positives = tf.data.Dataset.zip(
    (pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos))))
)
negatives = tf.data.Dataset.zip(
    (neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg))))
)
data = positives.concatenate(negatives)


# Preprocessing function
def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)

    spectrogram = tf.signal.stft(wav, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)

    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]

    return (spectrogram, label)


# Visualizing spectrogram
# filepath, label = data.shuffle(buffer_size=10000).as_numpy_iterator().next()
# spectrogram, label = preprocess(filepath, label)
# plt.imshow(tf.transpose(spectrogram)[0])
# plt.show()


# Create Tensorflow Data Pipeline
data = data.map(
    lambda f, l: tf.py_function(
        func=preprocess, inp=[f, l], Tout=[tf.float32, tf.float32]
    )
)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

train_data = data.take(36)
test_data = data.skip(36).take(15)


import numpy as np
import matplotlib.pyplot as plt

# ...

# Function to generate saliency map
def generate_saliency_map(spectrogram, model):
    with tf.GradientTape() as tape:
        tape.watch(spectrogram)
        predictions = model(spectrogram)
        target_class = tf.argmax(predictions[0])
        gradient = tape.gradient(predictions[:, target_class], spectrogram)
        saliency_map = tf.reduce_max(tf.abs(gradient), axis=-1)
    return saliency_map

# ...

# Load the trained model
loaded_model = tf.keras.models.load_model("audio_class_model.h5")

# Select a test input for generating saliency map
test_input, y_true = test_data.as_numpy_iterator().next()
test_input = tf.convert_to_tensor(test_input, dtype=tf.float32)
# Generate the saliency map
saliency_map = generate_saliency_map(test_input, loaded_model)


plt.imshow(saliency_map[0], cmap='hot')
plt.axis('off')
plt.savefig('result_saliency_map.png')
plt.show()
