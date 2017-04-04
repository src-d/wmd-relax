import unittest
import numpy
import scipy.spatial.distance
import libwmdrelax


class Base(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(777)

    def _get_w1_w2_dist_0(self):
        vecs = numpy.random.rand(4, 4)
        dist = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(vecs)).astype(numpy.float32)
        w1 = numpy.ones(4, dtype=numpy.float32) / 4
        w2 = numpy.ones(4, dtype=numpy.float32) / 4
        return w1, w2, dist

    def _get_w1_w2_dist(self):
        vecs = numpy.random.rand(4, 4)
        dist = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(vecs)).astype(numpy.float32)
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
        self.assertAlmostEqual(r, 0.6125112)

    def test_with_cache(self):
        cache = numpy.zeros(4, dtype=numpy.int32)
        w1, w2, dist = self._get_w1_w2_dist()
        r = libwmdrelax.emd_relaxed(w1, w2, dist, cache)
        self.assertAlmostEqual(r, 0.6125112)
        r = libwmdrelax.emd_relaxed(w1, w2, dist, cache=cache)
        self.assertAlmostEqual(r, 0.6125112)

    def test_with_cache_alloc(self):
        cache = libwmdrelax.emd_relaxed_cache_init(4)
        w1, w2, dist = self._get_w1_w2_dist()
        r = libwmdrelax.emd_relaxed(w1, w2, dist, cache)
        self.assertAlmostEqual(r, 0.6125112)
        r = libwmdrelax.emd_relaxed(w1, w2, dist, cache=cache)
        self.assertAlmostEqual(r, 0.6125112)
        libwmdrelax.emd_relaxed_cache_fini(cache)


class ExactTests(Base):
    def test_no_cache_0(self):
        w1, w2, dist = self._get_w1_w2_dist_0()
        r = libwmdrelax.emd(w1, w2, dist)
        self.assertAlmostEqual(r, 0)

    def test_no_cache(self):
        w1, w2, dist = self._get_w1_w2_dist()
        r = libwmdrelax.emd(w1, w2, dist)
        self.assertAlmostEqual(r, 0.6125115)

    def test_with_cache(self):
        cache = libwmdrelax.emd_cache_init(4)
        w1, w2, dist = self._get_w1_w2_dist()
        r = libwmdrelax.emd(w1, w2, dist, cache)
        self.assertAlmostEqual(r, 0.6125115)
        r = libwmdrelax.emd(w1, w2, dist, cache=cache)
        self.assertAlmostEqual(r, 0.6125115)
        libwmdrelax.emd_cache_fini(cache)


if __name__ == "__main__":
    unittest.main()
