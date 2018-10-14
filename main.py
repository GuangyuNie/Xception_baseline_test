from adv_gen import create_fmodel
from adversarial_vision_challenge import model_server
import numpy as np
import tensorflow as tf



if __name__ == '__main__':
	fmodel = create_fmodel()
	model_server(fmodel)


