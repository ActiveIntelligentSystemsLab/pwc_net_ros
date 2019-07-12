import rospy

import caffe

class PWCNetROS:
  def __init__(self):
    rospy.
    self._model_definition = rospy.get_param('model_definition', 'TODO:setpath')
    self._model_weights = rospy.get_param('model_weights', 'TODO:setweights')

    caffe.set_device(0)
    caffe.set_model_gpu()
    self._net = caffe.Net(self._model_definition, self._model_weights, caffe.Test)

    rospy.
  
  def 