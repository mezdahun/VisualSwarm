# Vision Module (visualswarm.vision)
Vision related methods from establishing the raw visual stream to process this raw stream further to a high level stream

## vacquire

### raw_vision:
#### Description:
Main process to capture raw input with the camera module using the `picamera` package. This method includes an infinite capturing loop and therefore can not be
called from the min process but rather in a dedicated `multiprocessing.Process` object. The parameters of the captured camera image can be set from `visualswarm.contrib.camera`
#### Args:
* raw_vision_stream: multiprocessing.Queue object to pass raw visual stream from camera

## vprocess

### high_level_vision:
#### Description:
Main process to preprocess raw video stream using `openCV` package. This method includes an infinite capturing loop and therefore can not be
called from the min process but rather in a dedicated `multiprocessing.Process` object. 
#### Args:
* raw_vision_stream: multiprocessing.Queue object which the process will consume and sequentially preprocess into a high-level stream
* high_level_vision_stream:  multiprocessing.Queue object to which the preprocessed stream will be forwarded so control processes can consume it for behavior.
