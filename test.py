import numpy
import unittest
import libwmdrelax


class ApproxRelaxedTests(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(777)

    def test_no_cache(self):
        dist = numpy.random.rand(4, 4).astype(numpy.float32)
        w1 = numpy.ones(4, dtype=numpy.float32)
        w2 = numpy.ones(4, dtype=numpy.float32)
        r = libwmdrelax.approximate_relaxed(w1, w2, dist)
        self.assertAlmostEqual(r, 1.2120517)

    def test_with_cache(self):
        cache = numpy.zeros(4, dtype=numpy.int32)
        dist = numpy.random.rand(4, 4).astype(numpy.float32)
        w1 = numpy.ones(4, dtype=numpy.float32)
        w2 = numpy.ones(4, dtype=numpy.float32)
        r = libwmdrelax.approximate_relaxed(w1, w2, dist, cache)
        self.assertAlmostEqual(r, 1.2120517)
        r = libwmdrelax.approximate_relaxed(w1, w2, dist, cache=cache)
        self.assertAlmostEqual(r, 1.2120517)


if __name__ == "__main__":
    unittest.main()
