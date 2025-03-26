from vmas.scenarios import flocking

class Flocking(flocking.Scenario):

    def max_rewards(self):
        return 1

    def zero_grad(self):
        self.world.zero_grad()
