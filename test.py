import numpy
import unittest
import libwmdrelax


class Base(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(777)

    def _get_w1_w2_dist_0(self):
        dist = numpy.random.rand(4, 4).astype(numpy.float32)
        w1 = numpy.ones(4, dtype=numpy.float32) / 4
        w2 = numpy.ones(4, dtype=numpy.float32) / 4
        return w1, w2, dist

    def _get_w1_w2_dist(self):
        dist = numpy.random.rand(4, 4).astype(numpy.float32)
        w1 = numpy.ones(4, dtype=numpy.float32) / 2
        w1[0] = w1[1] = 0
        w2 = numpy.ones(4, dtype=numpy.float32) / 2
        w2[2] = w2[3] = 0
        return w1, w2, dist


class RelaxedTests(Base):
    def test_no_cache_0(self):
        w1, w2, dist = self._get_w1_w2_dist_0()
        r = libwmdrelax.emd_relaxed(w1, w2, dist)
        self.assertAlmostEqual(r, 0)

    def test_no_cache(self):
        w1, w2, dist = self._get_w1_w2_dist()
        r = libwmdrelax.emd_relaxed(w1, w2, dist)
        self.assertAlmostEqual(r, 0.3945127)

    def test_with_cache(self):
        cache = numpy.zeros(4, dtype=numpy.int32)
        w1, w2, dist = self._get_w1_w2_dist()
        r = libwmdrelax.emd_relaxed(w1, w2, dist, cache)
        self.assertAlmostEqual(r, 0.3945127)
        r = libwmdrelax.emd_relaxed(w1, w2, dist, cache=cache)
        self.assertAlmostEqual(r, 0.3945127)

    def test_with_cache_alloc(self):
        cache = libwmdrelax.emd_relaxed_cache_init(4)
        w1, w2, dist = self._get_w1_w2_dist()
        r = libwmdrelax.emd_relaxed(w1, w2, dist, cache)
        self.assertAlmostEqual(r, 0.3945127)
        r = libwmdrelax.emd_relaxed(w1, w2, dist, cache=cache)
        self.assertAlmostEqual(r, 0.3945127)
        libwmdrelax.emd_relaxed_cache_fini(cache)


class ExactTests(Base):
    def test_no_cache_0(self):
        w1, w2, dist = self._get_w1_w2_dist_0()
        r = libwmdrelax.emd(w1, w2, dist)
        self.assertAlmostEqual(r, 0)

    def test_no_cache(self):
        w1, w2, dist = self._get_w1_w2_dist()
        r = libwmdrelax.emd(w1, w2, dist)
        self.assertAlmostEqual(r, 0.3062730)

    def test_with_cache(self):
        cache = libwmdrelax.emd_cache_init(4)
        w1, w2, dist = self._get_w1_w2_dist()
        r = libwmdrelax.emd(w1, w2, dist, cache)
        self.assertAlmostEqual(r, 0.3062730)
        r = libwmdrelax.emd(w1, w2, dist, cache=cache)
        self.assertAlmostEqual(r, 0.3062730)
        libwmdrelax.emd_cache_fini(cache)


if __name__ == "__main__":
    unittest.main()
