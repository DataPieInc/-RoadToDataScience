from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_arg_values
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import linear
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.linear_optimizer.python import sdca_optimizer

def _as_iterable(preds, output):

	for pred in preds:
		yield pred[output]


class SVM(estimator.Estimator):

	def __init__(self,
		example_id_column,
		feature_columns,
		weight_column_name=None,
		model_dir=None,
		l1_regularization=0.0,
		l2_regularization=0.0,
		num_loss_partitions=1,
		kernels=None,
		config=None,
		feature_engineering_fn=None):
	if kernels is not None:
		raise ValueError("Kernel SVMs are not currently supported.")

	optimizer = sdca_optimizer.SDCAOptimizer(
		example_id_column=example_id_column,
		num_loss_partitions=num_loss_partitions,
		symmetric_l1_regularization=l1_regularization,
		symmetric_l2_regularization=l2_regularization)

	self._feature_columns = feature_columns
	chief_hook = linear._SdcaUpdateWeightsHook()
	super(SVM, self).__init__(
		model_fn=linear.sdca_model_fn,
		model_dir=model_dir,
		config=config,
		params={
		"head" : head_lib.binary_svm_head(
			weight_column_name=weight_column_name,
			enable_centered_bias=False),
		"feature_columns" : feature_columns,
		"optimizer": optimizer,
		"weight_column_name" : weight_column_name,
		"update_weights_hook" : chief_hook,
		},
		feature_engineering_fn=feature_engineering_fn)

	@deprecated_arg_values(
		estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS, as_iterable=False)

	def predict_proba(self, x=None, input_fn=None, batch_size=None, outputs=None, as_iterable=False):

		"""Runs inference to determine the class probability predictions."""
		key = prediction_key.PredictionKey.PROBABILITIES
		preds = super(SVM, self).predict(
			x=x,
			input_fn=input_fn,
			batch_size=batch_size,
			outputs=[key],
			as_iterable=as_iterable)
		if as_iterable:
			return _as_iterable(preds, output=key)

		return preds[key]

	@deprecated("2017-09-25", "Please use Estimator.export_savedmodel() instead.")
	def export(self, export_dir, signature_fn=None,input_fn=None, default_batch_size=1, exports_to_keep=None):
		return self.export_with_defaults(
			export_dir=export_dir,
			signature_fn=signature_fn,
			input_fn=input_fn,
			default_batch_size=default_batch_size,
			exports_to_keep=exports_to_keep)

	@deprecated("2017-09-25", "Please use Estimator.export_savedmodel() instead.")
	def export_with_defaults(self, export_dir, signature_fn=None, default_batch_size=1, exports_to_keep=None):
		def default_input_fn(unused_estimator, examples):
			return layers.parse_feature_columns_from_examples(
				examples, self._feature_columns)
		return super(SVM, self).export(export_dir=export_dir,
			signature_fn=signature_fn,
			input_fn=input_fn or default_input_fn,
			default_batch_size=default_batch_size,
			exports_to_keep=exports_to_keep)