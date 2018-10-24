import unittest
import numpy
import scipy.spatial.distance
import libwmdrelax
import wmd
from numbers import Number


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


class TailVocabularyOptimizerTests(Base):
    def ndarray_almost_equals(self, a, b, msg=None):
        """Compares two 1D numpy arrays approximately."""
        if len(a) != len(b):
            if msg is None:
                msg = ("Length of arrays are not equal: {} and {}"
                       .format(len(a), len(b)))
            raise self.failureException(msg)
        for i, (x, y) in enumerate(zip(a, b)):
            try:
                self.assertAlmostEqual(x, y)
            except AssertionError as err:
                if msg is None:
                    msg = ("Arrays differ at index {}: {}" .format(i, err))
                raise self.failureException(msg)

    def setUp(self):
        self.tvo = wmd.TailVocabularyOptimizer()
        self.addTypeEqualityFunc(numpy.ndarray, self.ndarray_almost_equals)

    def test_trigger_ratio_getter_type(self):
        trigger_ratio = self.tvo.trigger_ratio
        self.assertIsInstance(trigger_ratio, Number)

    def test_trigger_ratio_constructor(self):
        tvo = wmd.TailVocabularyOptimizer(0.123)
        self.assertAlmostEqual(tvo.trigger_ratio, 0.123)

    def test_trigger_ratio_setter(self):
        self.tvo.trigger_ratio = 0.456
        self.assertAlmostEqual(self.tvo.trigger_ratio, 0.456)

    def test_trigger_ratio_too_low(self):
        with self.assertRaises(Exception):
            self.tvo.trigger_ratio = -0.5

    def test_trigger_ratio_too_high(self):
        with self.assertRaises(Exception):
            self.tvo.trigger_ratio = 1.5

    def test_call_below_trigger(self):
        tvo = wmd.TailVocabularyOptimizer(0.5)
        words = numpy.array([1, 2, 3], dtype=int)
        weights = numpy.array([0.5, 0.2, 0.3], dtype=numpy.float32)
        vocabulary_max = 10
        ret_words, ret_weights = tvo(words, weights, vocabulary_max)
        self.assertEqual(words, ret_words)
        self.assertEqual(weights, ret_weights)

    def test_call_too_many_words(self):
        tvo = wmd.TailVocabularyOptimizer(0.5)
        words = numpy.array([11, 22, 33, 44, 55, 66, 77], dtype=int)
        weights = numpy.array([0.5, 0.1, 0.4, 0.8, 0.6, 0.2, 0.7], dtype=numpy.float32)
        vocabulary_max = 2
        ret_words, ret_weights = tvo(words, weights, vocabulary_max)
        self.assertEqual(len(ret_words), vocabulary_max)
        self.assertEqual(len(ret_weights), vocabulary_max)
        sorter = numpy.argsort(ret_words)
        self.assertEqual(ret_words[sorter], numpy.array([44, 77]))
        self.assertEqual(ret_weights[sorter], numpy.array([0.8, 0.7]))

    def test_call(self):
        tvo = wmd.TailVocabularyOptimizer(0.5)
        words = numpy.array([11, 22, 33, 44, 55, 66, 77], dtype=int)
        weights = numpy.array([0.5, 0.1, 0.4, 0.8, 0.6, 0.2, 0.7], dtype=numpy.float32)
        vocabulary_max = 6
        ret_words, ret_weights = tvo(words, weights, vocabulary_max)
        self.assertEqual(len(ret_words), len(ret_weights))
        self.assertLessEqual(len(ret_words), vocabulary_max)
        sorter = numpy.argsort(ret_words)
        self.assertEqual(ret_words[sorter], numpy.array([11, 33, 44, 55, 77]))
        self.assertEqual(ret_weights[sorter], numpy.array([0.5, 0.4, 0.8, 0.6, 0.7]))


if __name__ == "__main__":
    unittest.main()
