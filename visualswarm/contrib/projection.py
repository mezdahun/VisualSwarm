"""
@author: mezdahun
@description: Parameters related to visual projection
"""
from visualswarm.contrib import camera

w, h = camera.RESOLUTION

top = 10
right = 50
down = 15
left = 80

w_limit = ((w-down)+top)
h_limit = ((h-left)+right)
