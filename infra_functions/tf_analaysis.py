# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


class nn_model:
    def __init__(self, hidden_layer_structure=None):
        self.model = None
        self._hiddel_layer_structure = hidden_layer_structure

    def build_nn_model(self, hidden_layer_structure = None):
        hidden_layer_structure = [(100, tf.nn.relu)] if hidden_layer_structure is None else hidden_layer_structure

        model = keras.Sequential()
        for layer_idx in range(len(hidden_layer_structure)):
            layer_structure = hidden_layer_structure[layer_idx]
            if type(layer_structure) is dict:
                type_of_layer = 'dense'
                layer_structure = (layer_structure, type_of_layer)
            else:
                type_of_layer = layer_structure[1].lower()
            if type_of_layer == 'dense':
                model.add(keras.layers.Dense(**layer_structure[0]))
            elif type_of_layer == 'flatten':
                model.add(keras.layers.Flatten(**layer_structure[0]))

        self.model = model

    def compile_nn_model(self, optimizer=tf.train.AdamOptimizer(), loss='mse',
                         metrics=None):
        metrics = ['mae', 'mse'] if metrics is None else metrics
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, train_inputs, train_outputs, epochs=5):
        self.model.fit(train_inputs, train_outputs, epochs=epochs)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def evaluate_model(self, test_inputs, test_outputs, verbose=True):
        test_loss_res= self.model.evaluate(test_inputs, test_outputs)

        # losses
        losses = self.model.loss
        if type(losses) is str:
            number_of_losses = 1
            losses = [losses]
        else:
            number_of_losses = len(losses)

        loss_dict = {}
        for loss_name, loss_value in zip(losses, test_loss_res[:number_of_losses]):
            loss_dict.update({loss_name: loss_value})

        # metrics
        metrics = self.model.metrics
        if type(metrics) is str:
            number_of_metrics = 1
            metrics = [metrics]
        else:
            number_of_metrics = len(metrics)

        metrics_dict = {}
        for metric_name, metric_value in zip(metrics, test_loss_res[number_of_losses:number_of_losses+number_of_metrics]):
            metrics_dict.update({metric_name: metric_value})

        if verbose:
            print(f'Loss = {loss_dict}\nMetrics = {metrics_dict}')
        return loss_dict, metrics_dict