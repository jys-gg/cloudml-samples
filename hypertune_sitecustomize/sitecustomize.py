# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Site level customizations.

This script is imported by python when any python script is run.
https://docs.python.org/2/library/site.html.
This script configures reporting hypertune metrics by monkey patching
tensorflow summary writer.
"""

import logging
import os
import sys
import hypertune
import tensorflow as tf

from wrapt import register_post_import_hook
from wrapt import wrap_function_wrapper


class SummaryFileWriterWrapper(object):
  """Overrides tf.summary.FileWriter.add_summary() so that it logs metrics.
  """

  def __init__(self, hp_metric_tag):
    self.hp_metric_tag = hp_metric_tag
    self.hpt = hypertune.HyperTune()

  def run_post_import_hook(self, module):
    """Called after the 'tensorflow' module is loaded (to patch it)."""
    try:
      wrap_function_wrapper(module, 'summary.FileWriter.add_summary',
                            self.new_add_summary)
    except Exception:
      wrap_function_wrapper(module, 'train.SummaryWriter.add_summary',
                            self.new_add_summary)

  def new_add_summary(self, wrapped, _, args, kwargs):
    """New implementation of tf.summary.FileWriter.add_summary()."""
    try:
      if len(args):
        summary = args[0]
        global_step = kwargs.get('global_step', 0)
        if len(args) > 1:
          global_step = args[1]
        global_step = int(global_step)

        if isinstance(summary, bytes):
          summ = tf.Summary()
          summ.ParseFromString(summary)
          summary = summ

        objective = None
        for value in summary.value:
          # custom hp metric tags must match exactly, but for the default
          # we accept anything ending in the string since starting in 0.12 the
          # tag names started having scopes added as prefixes.
          tag_match = False
          if self.hp_metric_tag:
            tag_match = value.tag == self.hp_metric_tag
          else:
            tag_match = value.tag.endswith('training/hptuning/metric')
          if tag_match:
            objective = value.simple_value
            self.hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag=self.hp_metric_tag,
                metric_value=objective,
                global_step=global_step)

    except Exception as e:
      logging.error('Unable to log objective metric due to exception %s.', e)
    return wrapped(*args, **kwargs)


def configure_summary_writer():
  """Configures override of tf.summary.FileWriter.add_summary().

  This is only performed for hyperparameter runs. See SummaryFileWriterWrapper
  above for more details.
  """
  # We only hook the summary if the env vars are set.
  # Specifically, Trial Id must not be empty, so this only happens on
  # hyperparameter runs.
  trial_id = os.environ.get('CLOUD_ML_TRIAL_ID', None)
  if not trial_id:
    return

  hp_metric_tag = os.environ.get('CLOUD_ML_HP_METRIC_TAG', '')
  wrapper = SummaryFileWriterWrapper(hp_metric_tag)
  register_post_import_hook(wrapper.run_post_import_hook, 'tensorflow')

def _customize():
  try:
    configure_summary_writer()
  except:
    e, v = sys.exc_info()[:2]
    logging.error('Unhandled exception %s:%s.', e, v)
    raise

_customize()
