import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import *
from tensorflow.python.framework import *
from tensorflow.math import confusion_matrix
from tensorflow.keras.metrics import Metric
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras import backend
from tensorflow.python.keras import backend as K

#Adapted from the tensorflow code
class ConfusionMatrixMetric(tf.keras.metrics.Metric):
    """
        Compute the mean intersection over union metrics based on Tensorflow
        mean IOU metrics.
        Mean Intersection-Over-Union is a common evaluation metric for semantic
        image segmentation, which first computes the IOU for each semantic class
        and then computes the average over classes. IOU is defined as follows:
        IOU = true_positive / (true_positive + false_positive + false_negative).
        The predictions are accumulated in a confusion matrix, weighted by
        `sample_weight` and the metric is then calculated from it.

        If `sample_weight` is `None`, weights default to 1.
        Use `sample_weight` of 0 to mask values.
        Inputs:
            num_classes: The possible number of labels the prediction task
            can have. This value must be provided, since a confusion matrix
            of dimension = [num_classes, num_classes] will be allocated.
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.

        Standalone usage:
        >>> # cm = [[1, 1],
        >>> #        [1, 1]]
        >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
        >>> # iou = true_positives / (sum_row + sum_col - true_positives))
        >>> # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
        >>> m = tf.keras.metrics.MeanIoU(num_classes=2)
        >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
        >>> m.result().numpy()
        0.33333334
        >>> m.reset_states()
        >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
        ...                sample_weight=[0.3, 0.3, 0.3, 0.1])
        >>> m.result().numpy()
        0.23809525
        Usage with `compile()` API:
        ```python
        model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
        ```
    """
    def __init__(self, num_classes , name='confusion_matrix_metric', **kwargs):
        # Variable to accumulate the predictions in the confusion matrix. Setting
        # the type to be `float64` as required by confusion_matrix_ops.
        super(ConfusionMatrixMetric,self).__init__(name=name,**kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros", dtype=dtypes.float32)


    def update_state(self, y_true, y_pred,sample_weight=None):
        """
        Accumulates the confusion matrix statistics.
        Inputs:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Defaults to 1.
            Can be a `Tensor` whose rank is either 0, or the same rank as
            `y_true`, and must broadcastable to `y_true`.
        Outputs:
            Update op.
        """
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = y_pred[..., tf.newaxis]

        if y_pred.shape.ndims > 1:
            y_pred = array_ops.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = array_ops.reshape(y_true, [-1])



        if sample_weight is not None:
            sample_weight = math_ops.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = array_ops.reshape(sample_weight, [-1])
        current_cm = confusion_matrix(y_true,y_pred,self.num_classes,weights=sample_weight,dtype=dtypes.float32)
        return self.total_cm.assign_add(current_cm)

    def result(self):
        return self.process_confusion_matrix()

    def reset_states(self):
        K.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def process_confusion_matrix(self):
        """
            Compute the mean intersection-over-union via the confusion matrix.
        """
        sum_over_row = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = math_ops.cast(
            array_ops.diag_part(self.total_cm), dtype=self._dtype)
        denominator = sum_over_row + sum_over_col - true_positives
        num_valid_entries = math_ops.reduce_sum(
            math_ops.cast(math_ops.not_equal(denominator, 0), dtype=self._dtype))

        iou = math_ops.div_no_nan(true_positives, denominator)

        mean_iou = math_ops.div_no_nan(
        math_ops.reduce_sum(iou, name='confusion_matrix_metric'), num_valid_entries)

        return mean_iou

    def get_config(self):
        """
            Needed to save and load a model using this metric.
        """
        config = {'num_classes': self.num_classes}
        base_config = super(ConfusionMatrixMetric, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

