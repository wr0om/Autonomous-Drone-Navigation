"""DQN_MODIFIED_VERSION_17 AUG 23"""
import os
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#from Tartarus import Tarpy
from controller import Robot
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        neurons = 128
        self.layer_1 = nn.Linear(state_dim, neurons)
        self.layer_2 = nn.Linear(neurons, neurons)
        self.layer_3 = nn.Linear(neurons, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.max_action * torch.tanh(self.layer_3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        neurons = 128
        self.layer_1 = nn.Linear(state_dim + action_dim, neurons)
        self.layer_2 = nn.Linear(neurons, neurons)
        self.layer_3 = nn.Linear(neurons, 1)
        # Q2 architecture
        self.layer_4 = nn.Linear(state_dim + action_dim, neurons)
        self.layer_5 = nn.Linear(neurons, neurons)
        self.layer_6 = nn.Linear(neurons, 1)

    def forward(self, x, u):
        # Remove the middle dimension
        x = x.squeeze(1)  # This changes shape from [100, 1, 5] to [100, 5]
        u = u.squeeze(1)  # This changes shape from [100, 1, 2] to [100, 2]

        # print(f"Shape of x (state tensor): {x.shape}")
        # print(f"Shape of u (action tensor): {u.shape}")


        xu = torch.cat([x, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        return self.layer_3(x1)
import random

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
    def __len__(self):
        return len(self.storage)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind: 
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r), np.array(d)

def train(actor, critic, actor_target, critic_target, replay_buffer, iterations, batch_size=64, discount=0.98, tau=0.005):
    for it in range(iterations):
        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(x).squeeze(1)
        action = torch.FloatTensor(u).squeeze(1)
        next_state = torch.FloatTensor(y).squeeze(1)
        done = torch.FloatTensor(1 - d)
        reward = torch.FloatTensor(r)

        # Compute the target Q value
        target_Q1, target_Q2 = critic_target(next_state, actor_target(next_state))
        current_Q1, current_Q2 = critic(state, actor(state))
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * discount * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = critic(state, action)

        # Debugging prints
        # print(f"target_Q1 shape: {target_Q1.shape}")
        # print(f"target_Q2 shape: {target_Q2.shape}")
        # print(f"reward shape before unsqueeze: {reward.shape}")
        # print(f"done shape before unsqueeze: {done.shape}")

        # Ensure reward and done are of correct shape
        reward = reward.unsqueeze(-1)  # Reshape from [100] to [100, 1]
        done = done.unsqueeze(-1)      # Reshape from [100] to [100, 1]

        # print(f"reward shape after unsqueeze: {reward.shape}")
        # print(f"done shape after unsqueeze: {done.shape}")

        # Compute the target Q value
        # target_Q = torch.min(target_Q1, target_Q2)
        # target_Q = reward + ((1 - done) * discount * target_Q).detach()
        
        
        Q1_dash = reward + ((1 - done) * discount * target_Q1).detach()
        
        Q2_dash = reward + ((1 - done) * discount * target_Q2).detach()
        
        

        # print(f"target_Q shape after calculation: {target_Q.shape}")

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, Q1_dash) + F.mse_loss(current_Q2, Q2_dash)
        # print("Critic Loss")
        # print(critic_loss)
        

        
        # Optimize the critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Compute actor loss
        actor_loss = -2*(critic.Q1(state, actor(state)).mean())
        # print("actor loss")
        # print(actor_loss)
        # Optimize the actor

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)





def bin_sensor_values(sensor_values):
    no_obstacle_thresh=75
    obstacle_proximity_thresh=175
    obstacle_close_thresh=400
    bin_sensors=[]
    for val in sensor_values:
        #code for if val < 75 then  val=0 elif 75<=val<175 then val=1 elif 175<=val<1000 then val=2 elif val>1000 then val=3
        if val<no_obstacle_thresh:
            val=0
        elif val>=no_obstacle_thresh and val<obstacle_proximity_thresh:
            val=1
        elif val>=obstacle_proximity_thresh and val<obstacle_close_thresh:
            val=2
        elif val>=obstacle_close_thresh:
            val=3
        bin_sensors.append(val)

    return bin_sensors
    
#Function to read sensor values from the robot
def read_sensors():
    psValues = []
    psValues=[sensor.getValue() for sensor in ps]
    
    # print(psValues)
    return psValues
    

    
    
    # print(psValues)
    #proximity_values_meter = [value * sensor.getLookupTable()[0] for value, sensor in zip(psValues, ps)]
    #for i in psValues:
       #     print("psvals  ", i)

def select_action(state):

    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert list to tensor and add batch dimension
    return actor(state_tensor).cpu().data.numpy().flatten()

def calculate_prox_reward(previous_state, current_state, action):
    reward=0
    
    for prev_val, curr_val in zip(previous_state, current_state):
        #0->0 +10; 0->1 +1; 0->2 -5; 0->3 -10;
        #1->0 +5; 1->1 +1; 1->2 -5; 1->3 -10;
        #2->0 +10; 2->1 +5; 2->2 -5; 2->3 -10;
        if prev_val==0 and curr_val==0 and all(curr==0 for curr in current_state)==0:
            reward+=2
            #print("1. +10")
        elif prev_val==0 and curr_val==1:
            reward+=0.1
            #print("2. +1")
        elif prev_val==0 and curr_val==2:
            reward-=1
            #print("3. -5")
        elif prev_val==0 and curr_val==3:
            reward-=2#2
            #print("4. -10")
            
        elif prev_val==1 and curr_val==0:
            reward+=1
            #print("5. +5")
        elif prev_val==1 and curr_val==1:
            reward+=0.1
            #print("6. +1")
        elif prev_val==1 and curr_val==2:
            reward-=1
            #print("7. -5")
        elif prev_val==1 and curr_val==3:
            reward-=2#2
            
            #print("8. -10")
        elif prev_val==2 and curr_val==0:
            reward+=2
            #print("9. +10")
        elif prev_val==2 and curr_val==1:
            reward+=1

        elif prev_val==2 and curr_val==2:
            reward-=1

        elif prev_val==2 and curr_val==3:
            reward-=2#2

            
        #3->0 +10; 3->1 +5; 3->2 +1; 3->3 -10;
        elif prev_val==3 and curr_val==0:
            reward+=2

        elif prev_val==3 and curr_val==1:
            reward+=1

        elif prev_val==3 and curr_val==2:
            reward+=0.1
    
        elif prev_val==3 and curr_val==3:
            reward-=2.5#2

            
        a=action[0][0]
        b=action[0][1]
        # if abs(a)<0.1 and abs(b)<0.1: 
            # reward-=0.05
        if a>0.0 and b>0.0 and current_state==[0,0,0,0,0] and previous_state==[0,0,0,0,0]:
            reward+=00

        # if a>0.9 and b>0.9 and current_state==[0,0,0,0,0]:
            # reward+=0.4
        if a==1 and b==1 and current_state==[0,0,0,0,0]:
            reward+=2

    return reward
    
    


    
class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    # def __repr__(self):
        # return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            # self.mu, self.sigma)

# def calculate_reward(previous_state, current_state, action):
    # reward=0
    # reward = 0
    # reward = sum(current_state)-sum(previous_state)
    
    # a=action[0][0]
    # b=action[0][1]
    # if a>0.95 and b>0.95 and current_state==[0,0,0,0,0]:
        # reward+=5

    # return reward
import math

def calculate_reward(current_state, next_state, action,yaw, desired_orientation, target_point,gps,diag):
    reward = 0
    
    # Extract the yaw value from the current state
    # Assuming the yaw value is at a certain index in your state tensor
    current_yaw = np.pi*current_state[0][2].item()
    desired_yaw = np.pi*current_state[0][1].item()
    # Calculate the difference in yaw from the desired yaw
    yaw_error = min(abs(current_yaw - desired_yaw), 2 * np.pi - abs(current_yaw - desired_yaw))
    # reward += pi-yaw_erro
    # Reward for facing the desired direction
    yaw_threshold = 3*0.1  # Define a threshold for acceptable yaw error


    # Reward based on forward movement towards the target
    current_distance_to_target = diag * current_state[0][0].item()
    next_distance_to_target = diag * next_state[0][0].item()
    if (yaw_error < yaw_threshold) and (next_distance_to_target < current_distance_to_target):
        reward += 1  # Reward for correct orientation
    else:
        reward+= -1
        
    dist = math.sqrt((target_point[0]-gps[0])**2+(target_point[1]-gps[1])**2)
    if dist<0.2:
        reward += 100
        print("reached")
        # time.sleep(10**100)# reached at 10523 step at the start no change
        
    if dist<0.3:
        reward += 30
        print("near target")

    # ## Reward/Penalty based on proximity to obstacles
    # proximity_threshold = 2  # Define a threshold for proximity
    # obstacle_penalty = -5      # Penalty for getting too close to an obstacle
    # skilled_navigation_reward = 2  # Reward for navigating close without collision

    # ## Assuming proximity sensor values are in the state tensor starting from index 2
    # proximity_values = current_state[0][8:].numpy()
    # if any(value >= proximity_threshold for value in proximity_values):
        # reward += obstacle_penalty
    # elif any(0 < value < proximity_threshold for value in proximity_values):
        # reward += skilled_navigation_reward

    return reward
    
def calculate_obstacle_reward(action,curr_sensed_dist_value):
    vtrans = 0
    vrot = 0
    if (action[0]*action[1] < 0 ):
        vtrans = 0
    else:
        if action[0]>0:
            vtrans = min(action)
        else:
            vtrans = max(action)
            
    if(action[0]<action[1]):
        vrot = (0.05/2)*(action[1]-action[0])
        
    elif(action[0]>action[1]):
        vrot = (0.05/2)*(action[0]-action[1])
        
    
    vrot = vrot*(1/0.05)
    reward = 0
    min_dist = min(curr_sensed_dist_value)
    # if min_dist == 0.06:
        # min_dist = 1
    reward = vtrans*(1-vrot)*(min_dist/0.06)
    
    return reward
    
    
    
    
    
def read_distance_sensors():
    psValues = []
    psValues=[sensor.getValue() for sensor in ps]
    for i in range(8):
        
        if psValues[i]<80:
            # print("no_obstacles")
            psValues[i]=0.06
        elif psValues[i]>1000:
            # print("full_obstacles")
            psValues[i]=0.0
        else:
            psValues[i]=0.06-((psValues[i]-80)/(1000-80))*0.06
        # print(a)
    return psValues
        
def read_distance_sensors_normalize(psValues):

    for i in range(8):
        psValues[i] = psValues[i]/0.06
    # print(psValues)
    return psValues
    
TIME_STEP = 1000 #in millisconds #increase / decrease for slowing speeding up the simulation
MAX_SPEED = 6.28
send_weights_iter=100 #change this to change the num of iterations afgter which the weights are sent

done = False
robot=Robot()

#initialize devices
ps=[0,0,0,0,0,0,0,0]
psNames=["ps0","ps1", "ps2","ps3", "ps4","ps5","ps6","ps7"] #two front ones 7,0; two side ones 5,2; one back 4
ps=[robot.getDevice(name) for name in psNames]
sampling_period=50 #in ms  It determines how often the proximity sensors will be sampled to get their values. A smaller value means the sensors will be read more frequently, while a larger value reduces the frequency.
for sensor in ps:
    sensor.enable(sampling_period)

# Initialize motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)
psValues = [0,0,0,0,0,0,0,0]

# Initialize the DDPG components
state_dim = 17  # Assuming five dimensions in the state space
action_dim = 2  # Assuming four possible continuous actions
max_action = 1.0  # Maximum value of action (adjust as needed)

actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim, action_dim)
actor_target = Actor(state_dim, action_dim, max_action)
critic_target = Critic(state_dim, action_dim)
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())
actor_optimizer = torch.optim.Adam(actor.parameters())
critic_optimizer = torch.optim.Adam(critic.parameters())
replay_buffer = ReplayBuffer()

imu = robot.getDevice("inertial unit")
imu.enable(TIME_STEP)
gps = robot.getDevice("gps")
gps.enable(TIME_STEP)
comp = robot.getDevice("compass")
comp.enable(TIME_STEP)


num_episodes = 5*15000
batch_size = 64
target_point = [-0.8, 0.8]
desired_orientation = 0.0  # Define the desired orientation here
# Load the saved model weights
# actor.load_state_dict(torch.load('actor_episode_15000.pth'))
# critic.load_state_dict(torch.load('critic_episode_15000.pth'))
action_noise = OUActionNoise(mu=np.zeros(action_dim), sigma=0.15, theta=0.2)
episode_reward=0
prev_action = [0, 0]
diag = math.sqrt(32)
for episode in range(num_episodes):
    robot.step(TIME_STEP)  # Update robot sensor values
    
    sensed_dist = read_distance_sensors()
    sensed_dist_norm=read_distance_sensors_normalize(sensed_dist)
    
    # bin_sensors1 = read_sensors()
    
    # bin_sensors1 = bin_sensor_values(bin_sensors1)
    
    robot_position = gps.getValues()
    # robot_orientation = comp.getValues()
    yaw = imu.getRollPitchYaw()[2]
    # Calculate distance and angle to target
    # print(yaw)
    relative_position = [target_point[0] - robot_position[0], target_point[1] - robot_position[1]]
    distance_to_target = np.linalg.norm(relative_position)
    angle_to_target = np.arctan2(relative_position[1], relative_position[0]) - robot_orientation[2]
    
    # Create current state
    current_state = torch.FloatTensor([distance_to_target/diag, angle_to_target/np.pi, yaw/np.pi, robot_position[0]/2,robot_position[1]/2,target_point[0]/2,target_point[1]/2,prev_action[0],prev_action[1]] + sensed_dist_norm).unsqueeze(0)
    
    # Select action based on current state
    action = select_action(current_state.numpy().flatten())
    noise = action_noise()
    action = np.clip(action + noise, -max_action, max_action)  # Apply noise and clip
    # print(action)
    # Set motor speeds based on action
    leftSpeed = MAX_SPEED * action[0]
    rightSpeed = MAX_SPEED * action[1]
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
    time.sleep(100)
    robot.step(TIME_STEP)  # Update robot sensor values
    
    leftMotor.setVelocity(leftSpeed*0)
    rightMotor.setVelocity(rightSpeed*0)
    # Calculate next state
    sensed_dist2 = read_distance_sensors()
    sensed_dist_norm2=read_distance_sensors_normalize(sensed_dist2)
    
    # bin_sensors2 = read_sensors()
    # bin_sensors2 = bin_sensor_values(bin_sensors2)
    robot_position2 = gps.getValues()
    yaw2 = imu.getRollPitchYaw()[2]
    
    relative_position2 = [target_point[0] - robot_position2[0], target_point[1] - robot_position2[1]]
    distance_to_target2 = np.linalg.norm(relative_position2)
    angle_to_target2 = np.arctan2(relative_position2[1], relative_position2[0]) - robot_orientation2[2]
    # robot_orientation2 = comp.getValues()
    

    

    next_state = torch.FloatTensor([distance_to_target2/diag, angle_to_target2/np.pi, yaw2/np.pi, robot_position2[0]/2, robot_position2[1]/2, target_point[0]/2, target_point[1]/2, action[0], action[1]] + sensed_dist_norm2).unsqueeze(0)

    reward = calculate_reward(current_state, next_state, action,yaw2, desired_orientation,target_point, robot_position2,diag)
    reward += calculate_obstacle_reward(action,sensed_dist)

    # Store the transition in the replay buffer
    replay_buffer.add((current_state, next_state, action, reward, float(done)))

    episode_reward += reward

    # Train the agent after collecting enough samples
    if len(replay_buffer) > 150:
        train(actor, critic, actor_target, critic_target, replay_buffer, iterations=batch_size)
    # print(f" episode {episode} action: {action} state : {current_state} reward {reward}")
    prev_action=action
    # state = next_state
    # bin_sensors1 = bin_sensors2
    # senses1=senses2
    if episode % 1000 == 0:
        actor_model_name = f'actor_episode_{episode}.pth'
        critic_model_name = f'critic_episode_{episode}.pth'
        torch.save(actor.state_dict(), actor_model_name)
        torch.save(critic.state_dict(), critic_model_name)

print(f"Episode: {episode}, Reward: {episode_reward}")

# Save models periodically


print("Training Complete!")



