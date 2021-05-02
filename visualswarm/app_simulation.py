import logging
import threading
from queue import Queue
import sys

# import visualswarm.contrib.vision
# from visualswarm import env
# from visualswarm.monitoring import ifdb, system_monitor
# from visualswarm.vision import vacquire, vprocess
# from visualswarm.contrib import logparams, vision, simulation
# from visualswarm.behavior import behavior
# from visualswarm.control import motorinterface, motoroutput


def test_reader(sensor_stream, motor_set_stream):
    while True:
        prox_vals = sensor_stream.get()
        motor_set_stream.put(prox_vals[0])
        print(f'put done: {prox_vals[0]}')


def webots_interface(robot, sensors, motors, timestep):

    # sensor and motor value queues shared across subprocesses
    sensor_stream = Queue()
    motor_set_stream = Queue()

    # A process to read and act according to sensor values
    t = threading.Thread(target=test_reader, args=(sensor_stream, motor_set_stream))
    t.start()

    # The main thread to interact with non-pickleable objects that can not be passed
    # to subprocesses
    while robot.step(timestep) != -1:
        prox_vals = [i.getValue() for i in sensors['prox']['horizontal']]
        # theoretically here we pass pickleable sensor values, etc to subprocesses
        # now we just print to see if it is running
        sensor_stream.put(prox_vals)
        motor_vals = motor_set_stream.get()
        if motor_vals != 0:
            print(f'setting motors to {-motor_vals / 1000}')
            motors['left'].setVelocity(-motor_vals / 1000)
            motors['right'].setVelocity(-motor_vals / 1000)
        else:
            print(f'setting motors to {0}')
            motors['left'].setVelocity(0)
            motors['right'].setVelocity(0)

    t.join()
    sensor_stream.close()
    motor_set_stream.close()
    print('Thread killed')