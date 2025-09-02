import math

class EloRatingSet:
    def __init__(self):
        self.ratings = {}

    def add(self, id, rating):
        self.ratings[id] = rating

    def probability(self, id1, id2):
        return 1.0 / (1 + math.pow(10, (self.ratings[id1] - self.ratings[id2]) / 400.0))

    def updateRating(self, id1, id2, K, outcome):
        pa = self.probability(id1, id2)
        pb = self.probability(id2, id1)

        ra = self.ratings[id1] + K * (outcome - pa)
        rb = self.ratings[id2] + K * ((1 - outcome) - pb)

        self.ratings[id1] = ra
        self.ratings[id2] = rb