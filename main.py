import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from common import set_seed
from common import get_logger
from common import get_session
from model import Distiller
from model import set_model

import numpy as np
import tensorflow as tf


def main():
    set_seed()
    get_session('2')
    logger = get_logger("MyLogger")

    ##########################
    # Prepare the dataset
    ##########################
    logger.info('##### Build Dataset #####')
    batch_size = 64
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.reshape(x_train, (-1, 28, 28, 1))

    x_test = x_test.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    ##########################
    # Build models
    ##########################
    logger.info('##### Build Models  #####')
    teacher, student, student_scratch = set_model()

    ##########################
    # Train the teacher
    ##########################
    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    logger.info('##### Train Teacher #####')
    teacher.fit(x_train, y_train, epochs=5)

    logger.info('##### Evaluate Teacher #####')
    teacher.evaluate(x_test, y_test)

    ##########################
    # Distill teacher to student
    ##########################
    distiller = Distiller(teacher=teacher, student=student)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss=tf.keras.losses.KLDivergence(),
        alpha=.1,
        temperature=10)
    
    logger.info('##### Distillation #####')
    distiller.fit(x_train, y_train, epochs=3)

    logger.info('##### Evaluate Distillation #####')
    distiller.evaluate(x_test, y_test)

    ##########################
    # Train student from scratch for comparison
    ##########################
    student_scratch.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    logger.info('##### Student Scratch #####')
    student_scratch.fit(x_train, y_train, epochs=3)

    logger.info('##### Evaluate Student Scratch #####')
    student_scratch.evaluate(x_test, y_test)


if __name__ == '__main__':
    main()