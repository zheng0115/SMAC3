'''
Created on Sep 1, 2017

@author: mrudolph
'''
import unittest
from smac.configspace.merge_config_spaces import merge_configurations_of_local_optimizations
import os
from smac.scenario.scenario import Scenario



class Test(unittest.TestCase):


    def test_merge_configurations_of_local_optimizations(self):
        dirname =os.path.dirname(os.path.abspath(__file__))
        scen = Scenario(scenario=os.path.join(dirname, "optimizationScenarios/test_160190/scenario.txt"))
        merge_configurations_of_local_optimizations(scen)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_merge_config_spaces_of_local_optimizations']
    unittest.main()