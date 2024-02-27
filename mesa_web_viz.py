###################
# SPECIAL FEATURE #
#     Doctors     #
###################
import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from mesa.datacollection import DataCollector
from mesa.experimental import JupyterViz
from collections import defaultdict

def compute_gini(model):
    humans = sum(1 for agent in model.schedule.agents if not agent.isZombie)
    return humans

class OutBreakAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Create the agent's variable and set the initial values.
        self.shotsLeft = 15
        self.dead = False
        
        num = random.randint(1,20)
        #zombie 10%
        if num <= 2:
            self.isZombie = True
            self.isDoctor = False
        #doctor 5%
        elif num == 3:
            self.isZombie = False
            self.isDoctor = True
        #human 85%
        else:
            self.isZombie = False
            self.isDoctor = False

    def step(self):
        if not self.dead:
            self.move()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=1)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)


class OutBreakModel(mesa.Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height):
        super().__init__()
        self.num_agents = N
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.datacollector = mesa.DataCollector(
            model_reporters={"Gini": compute_gini}, agent_reporters={"isZombie": "isZombie"}
        )
        
        # Create agents
        for i in range(self.num_agents):
            a = OutBreakAgent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        self.checkPositions()
        
    # this method will check for duplicate positions  
    def checkPositions(self):
        positions = defaultdict(list)
        for agent in self.schedule.agents:
            x, y = agent.pos
            positions[(x, y)].append(agent)
            
        for pos, agents in positions.items():
            
            # if shared grid space with zombie and human
            if any(agent.isZombie and not agent.dead for agent in agents) and any(not agent.isZombie for agent in agents):
                zombie = [agent for agent in agents if agent.isZombie and not agent.dead]
                human = [agent for agent in agents if not agent.isZombie]
                notDoctor = [agent for agent in agents if not agent.isZombie and not agent.isDoctor]
                doctor = [agent for agent in agents if not agent.isZombie and agent.isDoctor]
                # if a human has ammo
                if any(h.shotsLeft > 0 for h in human):
                    # if shot fails
                    if random.random() < 0.5:
                        humanToConvert = random.choice(human)
                        humanToConvert.isZombie = True
                        humanToConvert.isDoctor = False
                        print("Human infected!")
                        
                    # if shot lands
                    else: 
                        #if there is a doctor, cure zombie back to human if they have shots left
                        if doctor:
                            for z in zombie:
                                for d in doctor:
                                    if d.shotsLeft > 0:
                                        d.shotsLeft -= 1
                                        print('Zombie Cured!')
                                        z.isZombie = False                                                                                                        
                        
                        #if no doctor, kill zombie
                        else:
                            # remove 1 from ammo
                            for h in notDoctor:
                                if h.shotsLeft > 0:
                                    h.shotsLeft -= 1
                            #50% chance to give ammo to players
                            if random.random() < 0.5:
                                # distribute ammo equally
                                ammo = sum(z.shotsLeft for z in zombie)
                                ammoPerHuman = ammo // len(notDoctor)
                                
                                for h in notDoctor:
                                    h.shotsLeft += ammoPerHuman
                                    
                            # kill zombie
                            for z in zombie:
                                print('Zombie killed!')
                                z.dead = True
                                z.shotsLeft = 0
                                
            # if doctor and human share space, doctor shares cure knowledge and turns human into doctor 25% of the time                  
            elif any(not agent.isZombie and not agent.isDoctor for agent in agents) and any(agent.isDoctor for agent in agents):
                notDoctor = [agent for agent in agents if not agent.isZombie and not agent.isDoctor]
                
                for h in notDoctor:
                    if random.random() <= 0.25:
                        print("Human trained by Doctor!")
                        h.isDoctor = True
                
                
                
                             
model_params = {
    "N": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
        "step": 1,
    },

    #changes size of model in web visualizer
    "width": 20,
    "height": 20,
}

#modify this function to change output on grid
def agent_portrayal(agent):
    size = 50
    color = "tab:blue"
    
    if agent.isDoctor == True:
        color = "tab:green"
    if agent.isZombie == True:
        size = 25
        color = "tab:red"
        if agent.dead == True:
            color = (0.5, 0, 0, 1) 
    return {"size": size, "color": color}

page = JupyterViz(
    OutBreakModel,
    model_params,
    measures=["Gini"],
    name="OutBreak Model",
    agent_portrayal=agent_portrayal,
)
# This is required to render the visualization in the Jupyter notebook
page