"""Siamese network for face verification.

Two face images go through the SAME embedding network (shared weights).
The L1Dist layer takes |e1 - e2| element by element, and a Dense + sigmoid
layer turns that into a 0..1 "same person" score.

Design follows Koch et al. (2015), "Siamese Neural Networks for One-shot
Image Recognition".
"""

import tensorflow as tf
import keras
from keras import layers

from .config import CHANNELS, IMG_SIZE


@keras.saving.register_keras_serializable(package="visitrack")
class L1Dist(layers.Layer):
    """Element-wise absolute difference between two embeddings."""

    def call(self, anchor_embedding, other_embedding):
        return tf.math.abs(anchor_embedding - other_embedding)


def build_embedding(embedding_dim: int = 4096) -> keras.Model:
    """CNN that turns a 100x100x3 face into a vector (the embedding)."""
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS), name="face")

    x = layers.Conv2D(64, (10, 10), activation="relu")(inp)
    x = layers.MaxPooling2D(2, padding="same")(x)

    x = layers.Conv2D(128, (7, 7), activation="relu")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)

    x = layers.Conv2D(128, (4, 4), activation="relu")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)

    x = layers.Conv2D(256, (4, 4), activation="relu")(x)
    x = layers.Flatten()(x)
    out = layers.Dense(embedding_dim, activation="sigmoid", name="embedding_vector")(x)

    return keras.Model(inp, out, name="embedding")


def build_siamese(embedding_dim: int = 4096) -> keras.Model:
    """Two inputs -> shared embedding -> L1 distance -> same/different score.

    Built with the Functional API because the model has two inputs that
    share one sub-network, which the Sequential API cannot express.
    """
    embedding = build_embedding(embedding_dim)

    input_a = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS), name="input_image")
    input_b = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS), name="validation_image")

    # Calling the same model object twice = shared weights.
    e_a = embedding(input_a)
    e_b = embedding(input_b)

    distance = L1Dist(name="l1_distance")(e_a, e_b)
    score = layers.Dense(1, activation="sigmoid", name="classifier")(distance)

    return keras.Model([input_a, input_b], score, name="siamese")


def split_model(siamese: keras.Model):
    """Return (embedding_model, classifier_layer) from a trained Siamese model.

    At the gate we embed each stored face ONCE at registration, then only run the
    cheap L1 + Dense head per comparison instead of the full network twice.
    """
    embedding = siamese.get_layer("embedding")
    classifier = siamese.get_layer("classifier")
    return embedding, classifier


def score_embeddings(classifier, live_embedding, stored_embeddings):
    """Scores (0..1) of one live embedding against N stored embeddings.

    live_embedding: shape (D,)   stored_embeddings: shape (N, D)
    """
    diff = tf.math.abs(stored_embeddings - live_embedding[None, :])
    return tf.reshape(classifier(diff), [-1]).numpy()


def load_siamese(path) -> keras.Model:
    return keras.models.load_model(path, custom_objects={"L1Dist": L1Dist})
