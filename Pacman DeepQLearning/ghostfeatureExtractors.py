# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Ghost Agent game states"

from game import Directions, Actions
import util

def ghostDistance(pacman_pos, ghost_pos, walls):
    fringe = [(pacman_pos[0], pacman_pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a capsule at this location then exit
        if (pos_x, pos_y) == (int(ghost_pos[0]), int(ghost_pos[1])):
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no capsule found
    return None


def pacmanDistance(pacman_pos, ghost_pos, walls):
    fringe = [(ghost_pos[0], ghost_pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a capsule at this location then exit
        if (pos_x, pos_y) == (int(pacman_pos[0]), int(pacman_pos[1])):
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no capsule found
    return None

def closestCapsule(pos, capsules, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a capsule at this location then exit
        if (pos_x, pos_y) in capsules:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no capsule found
    return None

def getDistribution( self, state ):
    #print('DirectionAgent',self.index,state.data)
    # Read variables from state
    ghostState = state.getGhostState( self.index )
    legalActions = state.getLegalActions( self.index )
    pos = state.getGhostPosition( self.index )
    isScared = ghostState.scaredTimer > 0

    speed = 1
    if isScared: speed = 0.5

    actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
    newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
    pacmanPosition = state.getPacmanPosition()

    # Select best actions given the state
    distancesToPacman = [util.manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
    if isScared:
        bestScore = max( distancesToPacman )
        bestProb = self.prob_scaredFlee
    else:
        bestScore = min( distancesToPacman )
        bestProb = self.prob_attack
    bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]

    # Construct distribution
    dist = util.Counter()
    for a in bestActions: dist[a] = bestProb / len(bestActions)
    for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
    dist.normalize()
    return dist

class GhostFeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class GhostIdentityExtractor(GhostFeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats


class GhostAdvancedExtractor(GhostFeatureExtractor):
    prob_attack = 0.8
    prob_scaredFlee = 0.8

    def getFeatures(self, state, action, agentIndex):
        if (agentIndex == 2):
            features = util.Counter()
            ghostPos1 = state.getGhostPosition(1)
            ghostPos2 = state.getGhostPosition(2)
            pacmanPos = state.getPacmanPosition()
            ghostState2 = state.getGhostState(2)
            pacmanState = state.getPacmanState()
            walls = state.getWalls()
            scaredTimer = ghostState2.scaredTimer

            dx, dy = Actions.directionToVector(action)
            if scaredTimer > 0:
                dx /= 2
                dy /= 2

            ghost2x = ghostPos2[0]+dx
            ghost2y = ghostPos2[1]+dy
            ghostPos2 = (ghost2x, ghost2y)
            ghostDistance2 = util.manhattanDistance(pacmanPos, ghostPos2)

            # print(state)
            # print(self.shortestDistAStar(walls, pacmanPos, ghostPos2))

            features["bias"] = 1.0
            features["stepsToOtherGhost"] = util.manhattanDistance(ghostPos1, ghostPos2)  / (walls.width + walls.height)

            if scaredTimer > 0:
                features["stepsFromScaredGhostToPacman"] = ghostDistance2 /  (walls.width + walls.height)
            else:
                # minDistanceToCapsule = 999999999
                # closestCapsuleLoc = []
                # for capsuleLoc in state.getCapsules():
                    # distance = util.manhattanDistance(pacmanPos, capsuleLoc)
                    # if(distance < minDistanceToCapsule):
                        # closestCapsuleLoc =  capsuleLoc
                        # minDistanceToCapsule =  distance

                # features["minDistanceBetweenGhostPacman"] = (11 - minDistanceToCapsule) / (walls.width + walls.height)

                min_dist = ghostDistance2
                features["stepsFromGhostToPacman"] =  min_dist / (walls.width + walls.height)
                features["danger"] = 0
                if (len(state.getCapsules()) > 0 and ghostDistance2 > 0):
                    closestCapsuleLoc = []
                    distances = []
                    for capsuleLoc in state.getCapsules():
                        distance = util.manhattanDistance(pacmanPos, capsuleLoc)
                        distances.append(distance)

                    distances.sort()
                    minDistanceOfPacmanToCapsule = distances[0]
                    if (minDistanceOfPacmanToCapsule > 0):
                        if min_dist < 11:
                            features["danger"] = 1 - min_dist/11

                if min_dist > 12 :
                    features["keepwithinrange"] = 1.0
                else:
                    features["keepwithinrange"] = 0

                if ((features["danger"] == 0) and (min_dist <= 12)):
                    features["trap"] = 1 - min_dist/99
                else:
                    features["trap"] = 0

            return features
        else:
            features = util.Counter()
            ghostPos1 = state.getGhostPosition(2)
            ghostPos2 = state.getGhostPosition(1)
            pacmanPos = state.getPacmanPosition()
            ghostState2 = state.getGhostState(2)
            pacmanState = state.getPacmanState()
            walls = state.getWalls()
            scaredTimer = ghostState2.scaredTimer

            dx, dy = Actions.directionToVector(action)
            if scaredTimer > 0:
                dx /= 2
                dy /= 2

            ghost2x = ghostPos2[0]+dx
            ghost2y = ghostPos2[1]+dy
            ghostPos2 = (ghost2x, ghost2y)
            ghostDistance2 = util.manhattanDistance(pacmanPos, ghostPos2)

            # print(state)
            # print(self.shortestDistAStar(walls, pacmanPos, ghostPos2))

            features["bias"] = 1.0
            features["stepsToOtherGhost"] = util.manhattanDistance(ghostPos1, ghostPos2)  / (walls.width + walls.height)

            if scaredTimer > 0:
                features["stepsFromScaredGhostToPacman"] = ghostDistance2 /  (walls.width + walls.height)
            else:
                # minDistanceToCapsule = 999999999
                # closestCapsuleLoc = []
                # for capsuleLoc in state.getCapsules():
                    # distance = util.manhattanDistance(pacmanPos, capsuleLoc)
                    # if(distance < minDistanceToCapsule):
                        # closestCapsuleLoc =  capsuleLoc
                        # minDistanceToCapsule =  distance

                # features["minDistanceBetweenGhostPacman"] = (11 - minDistanceToCapsule) / (walls.width + walls.height)

                min_dist = ghostDistance2
                features["stepsFromGhostToPacman"] =  min_dist / (walls.width + walls.height)
                features["danger"] = 0
                if (len(state.getCapsules()) > 0 and ghostDistance2 > 0):
                    closestCapsuleLoc = []
                    distances = []
                    for capsuleLoc in state.getCapsules():
                        distance = util.manhattanDistance(pacmanPos, capsuleLoc)
                        distances.append(distance)

                    distances.sort()
                    minDistanceOfPacmanToCapsule = distances[0]
                    if (minDistanceOfPacmanToCapsule > 0):
                        if min_dist < 11:
                            features["danger"] = 1 - min_dist/11

                if min_dist > 12 :
                    features["keepwithinrange"] = 1.0
                else:
                    features["keepwithinrange"] = 0

                if ((features["danger"] == 0) and (min_dist <= 12)):
                    features["trap"] = 1 - min_dist/99
                else:
                    features["trap"] = 0

            return features


    
