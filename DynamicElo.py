import math

class StochasticEloSet():
    def __init__(self, items, convergenceGameCount, variance):
        self.ratings = {}
        self.step = (3 * (items - 1)) / (2 * convergenceGameCount * variance)

    def add(self, id, rating):
        self.ratings[id] = rating

    def sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-z))

    def expectedOutcome(self, id1, id2):
        return self.sigmoid(self.ratings[id1] - self.ratings[id2])

    def updateRatings(self, id1, id2, outcome_A):
        ra = self.ratings[id1]
        rb = self.ratings[id2]

        expectedA = self.expectedOutcome(id1, id2)
        expectedB = self.expectedOutcome(id2, id1)

        outcome_B = 1 - outcome_A
        delta_A = self.step * (outcome_A - expectedA)
        delta_B = self.step * (outcome_B - expectedB)

        self.ratings[id1] += delta_A
        self.ratings[id2] += delta_B