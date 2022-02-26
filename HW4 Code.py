#!/usr/bin/env python
# coding: utf-8

# # Ab Initio Protein Folding with Random Psi/Psi Perturbations
# ### In this program, I will be using the random residue selection, phi/psi perturbation, scoring, and decision making strategy for ab initio folding that we learned about in workshop 4.

# In[83]:


# import all necessary packages
from pyrosetta import *
from pyrosetta.teaching import *
from pyrosetta.toolbox import *
import matplotlib.pyplot as plt
import numpy as np
import math
import random
init()


# ### First, define a function that will choose a random residue in the passed pose and modify the phi and psi angles randomly.

# In[22]:


def randChange(old_pose):
    new_pose = Pose()
    new_pose.assign(old_pose) 
    randres = random.randint(1, old_pose.total_residue()) 
    old_phi = old_pose.phi(randres) 
    old_psi = old_pose.psi(randres) 
    new_phi = random.gauss(old_phi, 25)  
    new_psi = random.gauss(old_psi, 25)
    new_pose.set_phi(randres, new_phi)
    new_pose.set_psi(randres, new_psi)
    return new_pose


# ### Second, we define a scoring function that returns the score of a passed residue (I will use the default pyrosetta scoring function).

# In[11]:


sfxn = get_fa_scorefxn()

def score(pose):
    score = sfxn(pose)
    return score


# ### Third, we define a function that decides whether the new pose generated will replace the original. The probability of this will be determined by the change in energy where if the change in energy is negative, the new pose will always be adopted. If the change in energy is positive, the probability will be determined by the equation $P = e^{\frac{-\triangle G}{kT}}$ where we define kT = 1. To implement this, I will generate a random number (x) between 0 and 1, and if the P > x, then the new pose is adopted.

# In[92]:


def decision(before_pose, after_pose):
    x = random.random()
    decided = Pose()
    delG = score(after_pose) - score(before_pose)
    if delG <= 0:
        decided.assign(after_pose)
    else:
        prob = np.exp(-delG)
        if prob > x:
            decided.assign(after_pose)
        else:
            decided.assign(before_pose)
    return decided


# ### Finally, we need to put the functions together to modify the passed pose a set number of times and return the lowest energy pose instance across the iterations

# In[19]:


def foldit(init_pose, cycles):
    lowest_pose = Pose()
    new_pose = Pose()
    cycles = cycles
    curr_pose = Pose()
    curr_pose.assign(init_pose)
    for i in range(cycles):
        if i == 0:
            lowest_pose.assign(curr_pose)
        old_pose = Pose()
        old_pose.assign(curr_pose)
        new_pose.assign(randChange(curr_pose))
        curr_pose.assign(decision(old_pose, new_pose))
        if score(curr_pose) < score(lowest_pose):
            lowest_pose.assign(curr_pose)
        #print('Lowest Energy: {:.3f}, Current Energy: {:.3f}'.format(score(lowest_pose), score(curr_pose)))
    
    return lowest_pose
              
            
        


# ### Define a function that takes a list of poses that we will generate and extracts the phi and psi angles by iterating through the list and then iterating through the residues of each pose. Additionally, it will correct the angles calculated to be above 180 or below -180 degrees by subtracting or adding 360 degrees respectively to fit the formatting of a Ramachandran Plot.

# In[72]:


def ramachan(poses):
    phis = []
    psis = []
    for j in range(len(poses)):
        for i in range(poses[j].total_residue()):
            psis.append(poses[j].psi(i+1))
            phis.append(poses[j].phi(i+1))
            if (psis[-1] < -1000 or psis[-1]>1000) or (phis[-1] < -1000 or psis[-1] > 1000):
                phis[-1] = 0
                psis[-1] = 0
            if psis[-1]> 180:
                psis[-1] -= 360
            if phis[-1]> 180:
                phis[-1] -= 360
            if psis[-1]< -180:
                psis[-1] += 360
            if phis[-1]< -180:
                phis[-1] += 360
    '''           
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
    ], N=256)
    '''
    fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1, projection = 'scatter_density')
    #density = ax.scatter_density(phis, psis, cmap=white_viridis)
    #fig.colorbar(density, label='# of points per pixel')
    plt.xlim([-180, 180]);plt.ylim([-180, 180]);plt.xlabel('Phi (degrees)');plt.ylabel('Psi (degrees)')
    plt.grid();plt.title('Ramachandran Plot')
    plt.scatter(phis, psis, c = 'green', s = 50, alpha = 0.3)


# ### Generate a list of a specified number of decoys by running through the same number of trajectories (calling the folding function n times). 

# In[73]:


def getDecoys(original_pose, num_decoys, num_cycles):
    decoys = []
    cycles = num_cycles
    for i in range(num_decoys):
        new_decoy = Pose()
        new_decoy.assign(foldit(original_pose, cycles))
        decoys.append(new_decoy)
        print('Decoy {} score: {:.3f}'.format(i+1, score(decoys[-1])))
    return decoys


# In[74]:


#Run 100 trajectories and collect the decoys, then plot their phi and psi angles for a Poly-Alanine
polyA = pose_from_sequence('A'*10)
decoys = getDecoys(polyA, 100, 100000)
ramachan(decoys)


# In[76]:


#Run 100 Trajectories and collect their decoys for a Poly-Glycine
polyG = pose_from_sequence('G'*10)
decoysG = getDecoys(polyG, 100, 100000)
ramachan(decoysG)


# In[78]:


ramachan(decoysG)


# # Question 5: Predicting a real protein from scratch
# ### In this question, we are asked to attempt to fold a real protein from scratch and evaluate the stuctures and energies compared to the experimentally determined structure.
# ### To do this, I will create a pose from the sequence of the RecA protein (2REB). Then, I will create a list of decoys from 100 trajectories. Then, I will graph the energies of these decoys vs the RMSD between the decoy and pose from the 2REB pdb.

# In[80]:


def energyvsrmsd(decoys, experimental):
    energies = []
    rmsds = []
    for decoy in decoys:
        energies.append(score(decoy))
        rmsds.append(CA_rmsd(decoy, experimental))
    expscore = score(experimental)
    exprmsd = CA_rmsd(experimental, experimental)
    
    fig = plt.figure()
    plt.scatter(rmsds, energies, c='k', s = 30, label = 'Trials')
    plt.scatter([exprmsd], [expscore], c = 'green', s = 80, label = 'Native')
    plt.xlabel('RMSD');plt.ylabel('Energy Score');plt.legend();plt.grid()


# In[88]:


native = pyrosetta.toolbox.pose_from_rcsb('2REB')
'''
print(score(native))
seq_pose = pose_from_sequence(native.sequence())
decoys_2reb = getDecoys(seq_pose, 10, 1000)
energyvsrmsd(decoys_2reb, native)
'''


# In[89]:


print(score(native))
seq_pose = pose_from_sequence(native.sequence())
decoys_2reb = getDecoys(seq_pose, 10, 1000)
energyvsrmsd(decoys_2reb, native)


# In[ ]:


print(score(native))
decoys_2reb2 = getDecoys(seq_pose, 5, 10000)
energyvsrmsd(decoys_2reb2, native)


# In[ ]:




