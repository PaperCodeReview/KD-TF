import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


class Distiller(tf.keras.Model):
    def __init__(self, teacher, student):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self, 
        optimizer, 
        metrics, 
        student_loss, 
        distillation_loss, 
        alpha=.1, 
        temperature=3):

        """
        Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss: Loss function of difference between student 
                predictions and ground-truth
            distillation_loss: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss and 1-alpha to distillation_loss
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss = student_loss
        self.distillation_loss = distillation_loss
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        inputs, labels = data
        logits_teacher = self.teacher(inputs, training=False)
        with tf.GradientTape() as tape:
            logits_student = self.student(inputs, training=True)
            loss_student = self.student_loss(labels, logits_student)
            distillation_loss = self.distillation_loss(
                tf.nn.softmax(logits_teacher / self.temperature, axis=1),
                tf.nn.softmax(logits_student / self.temperature, axis=1))
            loss = self.alpha * loss_student + (1-self.alpha) * distillation_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(labels, logits_student)
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {'loss_student': loss_student, 'distillation_loss': distillation_loss})
        return results

    def test_step(self, data):
        inputs, labels = data
        logits_student = self.student(inputs, training=False)
        loss_student = self.student_loss(labels, logits_student)
        
        self.compiled_metrics.update_state(labels, logits_student)
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss_student': loss_student})
        return results

    
def set_model():
    teacher = tf.keras.Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=.2),
        MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        Conv2D(512, (3, 3), strides=(2, 2), padding='same'),
        Flatten(),
        Dense(10)], name='teacher')
    
    student = tf.keras.Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(16, (3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=.2),
        MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        Conv2D(32, (3, 3), strides=(2, 2), padding='same'),
        Flatten(),
        Dense(10)], name='student')

    student_scratch = tf.keras.models.clone_model(student)
    return teacher, student, student_scratch