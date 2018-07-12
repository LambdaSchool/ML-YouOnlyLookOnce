import unittest
from solution import contains_banana

class TestReward(unittest.TestCase):

    def test_always_passing_test(self):
        self.assertEqual(3 + 1, 4)

    def test_positive_1(self):
        self.assertGreater(contains_banana('./sample_data/positive_examples/example0.jpeg'), 0.0)

    def test_positive_2(self):
        self.assertGreater(contains_banana('./sample_data/positive_examples/example1.jpeg'), 0.0)

    def test_positive_3(self):
        self.assertGreater(contains_banana('./sample_data/positive_examples/example2.jpeg'), 0.0)

    def test_negative_1(self):
        self.assertEqual(contains_banana('./sample_data/negative_examples/example10.jpeg'), 0.0)

    def test_negative_2(self):
        self.assertEqual(contains_banana('./sample_data/negative_examples/example11.jpeg'), 0.0)

    def test_negative_3(self):
        self.assertEqual(contains_banana('./sample_data/negative_examples/example12.jpeg'), 0.0)


if __name__ == '__main__':
    unittest.main()
