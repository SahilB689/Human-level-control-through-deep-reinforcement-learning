import random
import numpy as np
from utils.test_env import EnvTest


class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.nsteps = nsteps

    def update(self, t: int):
        """
        Updates epsilon

        Args:
            t: int
                frame number 
        self.epsilon grows linearly from self.eps_begin to 
			  self.eps_end as t goes from 0 to self.nsteps
			  For t > self.nsteps self.epsilon remains constant
        """ 
        
        if t > self.nsteps: 
            self.epsilon = self.eps_end 
        else: 
            self.epsilon = t/self.nsteps * (self.eps_end-self.eps_begin) + self.eps_begin
         

class LinearExploration(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: MinAtar-like environment
            eps_begin: float
                initial exploration rate
            eps_end: float
                final exploration rate
            nsteps: int
                number of steps taken to linearly decay eps_begin to eps_end
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)

    def get_action(self, best_action: int) -> int:
        """
        Returns a random action with prob epsilon, otherwise returns the best_action

        Args:
            best_action: int
                best action according some policy
        Returns:
            an action 
        
        With probability self.epsilon, this function will return a random action
                else, return best_action

                you can access the environment via self.env

                normally, you could use env.action_space.sample() to generate 
                a random action. MinAtar envs however use a different structure,
                so you may use random.randrange, and get the number of possible
                actions via env.num_actions()
        """ 
        if random.random() < self.epsilon:
            return random.randrange(self.env.num_actions())
        else:
            return best_action  


def test1():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)

    found_diff = False
    for i in range(10):
        rnd_act = exp_strat.get_action(0)
        if rnd_act != 0 and rnd_act is not None:
            found_diff = True

    assert found_diff, "Test 1 failed."
    print("Test1: ok")


def test2():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)
    exp_strat.update(5)
    assert exp_strat.epsilon == 0.5, "Test 2 failed"
    print("Test2: ok")


def test3():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0.5, 10)
    exp_strat.update(20)
    assert exp_strat.epsilon == 0.5, "Test 3 failed"
    print("Test3: ok")
 
if __name__ == "__main__":
    test1()
    test2()
    test3()
    your_test()

    