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
        merged_configs = merge_configurations_of_local_optimizations(scen, 3)
        self.assertEqual(3, len(merged_configs))
        
        for merged_config in merged_configs:
            for device_id in range(0,3):
                self.assertTrue(merged_config["device0_" + str(device_id)] is not None)
                self.assertTrue(merged_config["device1_" + str(device_id)] is not None)
                self.assertTrue(merged_config["device2_" + str(device_id)] is not None)
        
        merged_configs = merge_configurations_of_local_optimizations(scen, 7)
        self.assertEqual(7, len(merged_configs))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_merge_config_spaces_of_local_optimizations']
    unittest.main()