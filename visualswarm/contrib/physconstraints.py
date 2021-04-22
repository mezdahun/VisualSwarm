"""
@author: mezdahun
@description: Parameters related to dimensions, and navigating in the physical environment
"""
# this parameter transforms a motor value pair of (n, -n) to degree/seconds. as an example if you set your motor
# values to 50, -50 the robot will rotate around the center of mass with 50*ROT_MULTIPLIER degree/second.
# this value will depend on the physical environment, i.e. the quality of the floor, etc.
ROT_MULTIPLIER = 0.386