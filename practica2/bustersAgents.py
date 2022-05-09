from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import math
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters

class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())
        
        
    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move

    def printLineData(self, gameState):
        return "XXXXXXXXXX"

class QLearningAgent(BustersAgent):
    def registerInitialState(self, gameState):
        "Initialize Q-values"
        self.actions = {"North":0, "East":1, "South":2, "West":3}
        self.table_file = open("qtable.txt", "r+")
        self.q_table = self.readQtable()
        self.epsilon = 0.05
        self.alpha = 1
        self.discount = 0

        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()

        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item)+" ")
            self.table_file.write("\n")

    def getState(self, gameState):
        state = ["None", 0]
        pacmanPosition = gameState.getPacmanPosition()
        livingGhosts = gameState.getLivingGhosts()
        minDistance = gameState.data.layout.width * gameState.data.layout.height
        mini = -1

        # Calcular el index del fantasma más cercano
        for i in range(1, len(livingGhosts)): # Empezamos por el index 1 ya que el 0 en livingGhosts representa a Pac-Man
            if livingGhosts[i]:
                ghostPosition = gameState.getGhostPositions()[i-1]
                currentDistance = self.distancer.getDistance(pacmanPosition, ghostPosition)
                if currentDistance < minDistance:
                    minDistance = currentDistance
                    mini = i-1

        if mini == -1: # Si no hay fantasmas vivos return el estado final
            return state

        # Calcular la posición relativa del fantasma respecto a Pac-Man
        ghostPosition = gameState.getGhostPositions()[mini]
        xDistance = pacmanPosition[0] - ghostPosition[0]
        yDistance = pacmanPosition[1] - ghostPosition[1]

        # Asingnar la dirección relativa
        if (yDistance < 0 and xDistance == 0):
            state[0] = Directions.NORTH
        if (yDistance < 0 and xDistance < 0):
            state[0] = "Northeast"
        if (yDistance == 0 and xDistance < 0):
            state[0] = Directions.EAST
        if (yDistance > 0 and xDistance < 0):
            state[0] = "Southeast"
        if (yDistance > 0 and xDistance == 0):
            state[0] = Directions.SOUTH
        if (yDistance > 0 and xDistance > 0):
            state[0] = "Southwest"
        if (yDistance == 0 and xDistance > 0):
            state[0] = Directions.WEST
        if (yDistance < 0 and xDistance > 0):
            state[0] = "Northwest" 
        
        # Calcular en que rango del 1 al 10 está la distancia al fantasma, dependiendo de las dimensiones del mapa
        ghostDistance = self.distancer.getDistance(pacmanPosition, ghostPosition)
        rangeSize = ((gameState.data.layout.width-2)+(gameState.data.layout.height-4))/10.0
        state[1] = int(math.ceil(ghostDistance/rangeSize))
 
        return state

    def getReward(self, gameState, nextGameState):
        # Calcular el índice del fantasma más cercano
        pacmanPosition = gameState.getPacmanPosition()
        livingGhosts = gameState.getLivingGhosts()
        minDistance = gameState.data.layout.width * gameState.data.layout.height
        mini = -1

        for i in range(1, len(livingGhosts)):
            if livingGhosts[i]:
                ghostPosition = gameState.getGhostPositions()[i-1]
                currentDistance = self.distancer.getDistance(pacmanPosition, ghostPosition)
                if currentDistance < minDistance:
                    minDistance = currentDistance
                    mini = i-1

        ghostPosition = gameState.getGhostPositions()[mini]
        ghostDistance = self.distancer.getDistance(pacmanPosition, ghostPosition)

        # Si Pac-Man se ha comido al fantasma más cercano en el siguiente turno devolver un refuerzo de 200
        if not nextGameState.getLivingGhosts()[mini+1]:
            return 200

        # Calcular la distancia al fantasma en el siguiente turno
        nextGhostPosition = nextGameState.getGhostPositions()[mini]
        nextPacmanPosition = nextGameState.getPacmanPosition()
        nextGhostDistance = self.distancer.getDistance(nextPacmanPosition, nextGhostPosition)
       
        # Devolver la diferencia entre la distancia de este tick y el siguiente
        return ghostDistance - nextGhostDistance
        

    def computePosition(self, gameState):
        """
        Compute the row of the qtable for a given state.
        For instance, the state (3,1) is the row 7
        """
        state = self.getState(gameState)
        if state[0] == "None" and state[1] == 0:
            return 80 # Última fila para el estado final

        direction = 0
        if (state[0] == Directions.NORTH):
            direction = 0
        elif (state[0] == "Northeast"):
            direction = 1
        elif (state[0] == Directions.EAST):
            direction = 2
        elif (state[0] == "Southeast"):
            direction = 3
        elif (state[0] == Directions.SOUTH):
            direction = 4
        elif (state[0] == "Southwest"):
            direction = 5
        elif (state[0] == Directions.WEST):
            direction = 6
        elif (state[0] == "Northwest"):
            direction = 7

        # Primero se encuentran las filas de todas las direcciones con distancia 1 y luego van aumentando el valor de la distancia
        return direction+(state[1]-1)*8

    def getQValue(self, gameState, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(gameState)
        action_column = self.actions[action]

        return self.q_table[position][action_column]

        
    def computeValueFromQValues(self, gameState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = gameState.getLegalActions()
        legalActions.remove('Stop')
        return max(self.q_table[self.computePosition(gameState)])

    def computeActionFromQValues(self, gameState):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = gameState.getLegalActions()
        legalActions.remove('Stop')

        best_actions = [legalActions[0]]
        best_value = self.getQValue(gameState, legalActions[0])
        for action in legalActions:
            value = self.getQValue(gameState, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, gameState):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        # Pick Action
        legalActions = gameState.getLegalActions()
        legalActions.remove('Stop')
        action = 'Stop'

        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        return self.getPolicy(gameState)

    def update(self, gameState, action, nextGameState, reward):
        """
        Q-Learning update:
        if terminal_state:
            Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
            Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))
        """
        position = self.computePosition(gameState)
        action_column = self.actions[action]
        nextState = self.getState(nextGameState)
        nextAction = self.getValue(nextGameState)	

        if nextState[0] == "None" and nextState[1] == 0:
            self.q_table[position][action_column] = (1-self.alpha) * self.q_table[position][action_column] + self.alpha * reward
        else:
            self.q_table[position][action_column] = (1-self.alpha) * self.q_table[position][action_column] + self.alpha * (reward + self.discount * nextAction)

    def getPolicy(self, gameState):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(gameState)

    def getValue(self, gameState):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(gameState)

