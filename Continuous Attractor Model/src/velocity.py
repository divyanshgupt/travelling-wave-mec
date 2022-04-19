import numpy as np
from brian2 import *

def zero_velocity(dt, duration):
    nb_steps = int(duration/dt)
    x = zeros(nb_steps)
    y = zeros(nb_steps)
    velocity = column_stack((x, y))
    return velocity


def within_boundary(x, y, n):
    if x < n and y < n and y > 0 and x > 0:
        return True
    else:
        return False

def close_to_boundary(x, y, n, epsilon):
    if abs(x - n) < epsilon:
        return True
    elif abs(y -n) < epsilon:
        return True
    elif abs(x - 0) < epsilon:
        return True
    elif  abs(y - 0) < epsilon:
        return True
    else:
        return False

def smooth_random_trajectory(n, step_size, dt, duration, epsilon=0.05):
    """
    Returns rat trajectory and velocity based on the method 
    suggested by Mittal & Narayanan, 2020

    Args:
    n - size of neural sheet (also size of field for animal)
    step_size - each step is drawn from a uniform distribution over [0, step_size]
    dt - timestep size (in ms)
    duration - total simulation size (in ms)

    """

    nb_steps = int(duration/dt)
    angle = zeros(nb_steps)
    x = empty(nb_steps + 1)
    y = empty(nb_steps + 1)
    x[0] = y[0] = n/2

    for i in range(1, nb_steps+1):


        if close_to_boundary(x[i-1], y[i-1],n, epsilon):
            new_angle = np.random.uniform(0, 2*pi)
        else:
            new_angle = np.random.uniform(-pi/36, pi/36)

        step = np.random.uniform(0, step_size)
        x[i] = x[i-1] + step*sin(angle[i-1] + new_angle)
        y[i] = y[i-1] + step*cos(angle[i-1] + new_angle)
        
        while not(within_boundary(x[i], y[i], n)):
            new_angle = np.random.uniform(-pi/36, pi/36)
            step = np.random.uniform(0, step_size)
            x[i] = x[i-1] + step*sin(angle[i-1] + new_angle)
            y[i] = y[i-1] + step*cos(angle[i-1] + new_angle)
        
        angle += new_angle
            
    velocity_array = column_stack((diff(x)/dt, diff(y)/dt))
    position_array = column_stack((x, y))

    return position_array, velocity_array, angle


def straight_trajectory(dt, duration, speed):
    """
    
    Args:
        dt - 
        duration - 
        speed - in metres/sec
    """
    
    nb_steps = int(duration/dt)
    angle = np.random.random()*2*pi
    
    x = cos(angle)*arange(0, nb_steps+1)*speed*dt
    y = sin(angle)*arange(0, nb_steps+1)*speed*dt

    velocity_x = diff(x)/dt
    velocity_y = diff(y)/dt

    velocity = column_stack((velocity_x, velocity_y)) *metre/second
    trajectory = column_stack((x, y))


    return trajectory, velocity