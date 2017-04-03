import numpy
import unittest
import libwmdrelax


def calc_w1_w2_dist():
    dist = numpy.random.rand(4, 4).astype(numpy.float32)
    w1 = numpy.ones(4, dtype=numpy.float32) / 4
    w2 = numpy.ones(4, dtype=numpy.float32) / 4
    return w1, w2, dist


class RelaxedTests(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(777)

    def test_no_cache(self):
        w1, w2, dist = calc_w1_w2_dist()
        r = libwmdrelax.emd_relaxed(w1, w2, dist)
        self.assertAlmostEqual(r, 0.3030129)

    def test_with_cache(self):
        cache = numpy.zeros(4, dtype=numpy.int32)
        w1, w2, dist = calc_w1_w2_dist()
        r = libwmdrelax.emd_relaxed(w1, w2, dist, cache)
        self.assertAlmostEqual(r, 0.3030129)
        r = libwmdrelax.emd_relaxed(w1, w2, dist, cache=cache)
        self.assertAlmostEqual(r, 0.3030129)

    def test_with_cache_alloc(self):
        cache = libwmdrelax.emd_relaxed_cache_init(4)
        w1, w2, dist = calc_w1_w2_dist()
        r = libwmdrelax.emd_relaxed(w1, w2, dist, cache)
        self.assertAlmostEqual(r, 0.3030129)
        r = libwmdrelax.emd_relaxed(w1, w2, dist, cache=cache)
        self.assertAlmostEqual(r, 0.3030129)
        libwmdrelax.emd_relaxed_cache_fini(cache)


class ExactTests(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(777)

    def test_no_cache(self):
        w1, w2, dist = calc_w1_w2_dist()
        r = libwmdrelax.emd(w1, w2, dist)
        self.assertAlmostEqual(r, 0.3030129)

    def test_with_cache(self):
        cache = libwmdrelax.emd_cache_init(4)
        w1, w2, dist = calc_w1_w2_dist()
        r = libwmdrelax.emd(w1, w2, dist, cache)
        self.assertAlmostEqual(r, 0.3030129)
        r = libwmdrelax.emd(w1, w2, dist, cache=cache)
        self.assertAlmostEqual(r, 0.3030129)
        libwmdrelax.emd_cache_fini(cache)


if __name__ == "__main__":
    unittest.main()
