import unittest

from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown, \
    ChooserLinearCoolDown

class TestRandomCOnfigurationChooser(unittest.TestCase):

    def test_no_cool_down(self):
        c = ChooserNoCoolDown(modulus=3.0)
        self.assertFalse(c.check(1))
        self.assertFalse(c.check(2))
        self.assertTrue(c.check(3))
        self.assertFalse(c.check(4))
        self.assertFalse(c.check(5))
        self.assertTrue(c.check(6))
        self.assertTrue(c.check(30))
        c = ChooserNoCoolDown(modulus=1.0)
        self.assertTrue(c.check(1))
        self.assertTrue(c.check(2))
        self.assertTrue(c.check(3))

    def test_linear_cool_down(self):
        c = ChooserLinearCoolDown(2.0, 1.0, 4.0)
        self.assertFalse(c.check(1))
        self.assertTrue(c.check(2))
        self.assertFalse(c.check(3))
        self.assertFalse(c.check(4))
        self.assertTrue(c.check(5))
        self.assertFalse(c.check(6))
        self.assertFalse(c.check(7))
        self.assertFalse(c.check(8))
        self.assertTrue(c.check(9))
        self.assertFalse(c.check(10))
        self.assertFalse(c.check(11))
        self.assertFalse(c.check(12))
        self.assertTrue(c.check(13))
        self.assertFalse(c.check(14))


if __name__ == "__main__":
    unittest.main()
