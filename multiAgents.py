# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distanceToGhost = min(manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates)
        if distanceToGhost < 3:
            return distanceToGhost*-500
        if newFood.count() == 0:
            return 1000
        return 1000 - min(manhattanDistance(newPos, food) for food in newFood.asList()) - 25*newFood.count()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxNode:
    def __init__(self, game_state, agent_index):
        """
        :type game_state: pacman.GameState
        :type agent_index: int
        """
        self.game_state = game_state
        self.agent_index = agent_index

    def successors(self):
        next_agent_index = (self.agent_index + 1) % self.game_state.getNumAgents()
        for action in self.game_state.getLegalActions(self.agent_index):
            next_sate = self.game_state.generateSuccessor(self.agent_index, action)
            yield MinimaxNode(next_sate, next_agent_index)

    def maximize(self):
        return self.agent_index == 0


def is_terminal(game_state):
    """
        :type game_state: pacman.GameState
    """
    return game_state.isWin() or game_state.isLose()


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        n_agents = gameState.getNumAgents()
        depth = self.depth*n_agents - 1

        def action_value(action):
            state = gameState.generateSuccessor(0, action)
            node = MinimaxNode(state, 1)
            return self.minimax(node, depth)

        actions = gameState.getLegalActions(0)
        return max(actions, key=action_value)

    def minimax(self, node, depth):
        """
        :type node: MinimaxNode
        :type depth: int

        """
        if depth == 0 or is_terminal(node.game_state):
            return self.evaluationFunction(node.game_state)

        successors = node.successors()
        if node.maximize():
            return max(self.minimax(successor, depth - 1) for successor in successors)
        else:
            return min(self.minimax(successor, depth - 1) for successor in successors)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        n_agents = gameState.getNumAgents()
        depth = self.depth*n_agents - 1

        def action_value(action):
            state = gameState.generateSuccessor(0, action)
            node = MinimaxNode(state, 1)
            return self.alphabeta(node, float('-infinity'), float('+infinity'), depth)

        actions = gameState.getLegalActions(0)
        return max(actions, key=action_value)

    def alphabeta(self, node, alpha, beta, depth):
        if depth == 0 or is_terminal(node.game_state):
            return self.evaluationFunction(node.game_state)

        if node.maximize():
          v = float('-infinity')
          for successor in node.successors():
            v = max(v, self.alphabeta(successor, alpha, beta, depth-1))
            if v > beta:
              return v
            alpha = max(alpha, v)
        else:
          v = float('infinity')
          for successor in node.successors():
            v = min(v, self.alphabeta(successor, alpha, beta, depth-1))
            if v < alpha:
              return v
            beta = min(beta, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

