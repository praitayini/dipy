import pytest

from dipy.data import get_fnames
from dipy.utils.optpkg import optional_package
import numpy as np
from numpy.testing import assert_almost_equal

tf, have_tf, _ = optional_package('tensorflow', min_version='2.0.0')
tfa, have_tfa, _ = optional_package('tensorflow_addons')

if have_tf and have_tfa:
    from dipy.nn.deepn4 import DeepN4


@pytest.mark.skipif(not all([have_tf, have_tfa]), reason='Requires TensorFlow \
                                                          , TensorFlow_addons')
def test_default_weights():
    file_names = get_fnames('deepn4_test_data')
    input_arr = np.load(file_names[0])
    target_arr = np.load(file_names[1])

    deepn4_model = DeepN4()
    deepn4_model.fetch_default_weights(0)
    results_arr = deepn4_model.predict(input_arr)
    assert_almost_equal(results_arr, target_arr, decimal=1)


@pytest.mark.skipif(not all([have_tf, have_tfa]), reason='Requires TensorFlow \
                                                          , TensorFlow_addons')
def test_default_weights_batch():
    file_names = get_fnames('deepn4_test_data')
    input_arr = np.load(file_names[0])
    target_arr = np.load(file_names[1])
    deepn4_model = DeepN4()
    deepn4_model.fetch_default_weights(0)
    results_arr = deepn4_model.predict(input_arr)
    assert_almost_equal(results_arr, target_arr, decimal=1)
