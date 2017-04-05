import logging
import heapq
import os
import sys
from time import time
import numpy

sys.path.insert(0, os.path.dirname(__file__))
import libwmdrelax
del sys.path[0]

__version__ = (1, 1, 2)


class TailVocabularyOptimizer(object):
    def __init__(self, trigger_ratio=0.75):
        self._trigger_ratio = trigger_ratio

    @property
    def trigger_ratio(self):
        return self._trigger_ratio

    @trigger_ratio.setter
    def trigger_ratio(self, value):
        if value <= 0 or value > 1:
            raise ValueError("trigger_ratio must lie in (0, 1]")
        self._trigger_ratio = value

    def __call__(self, words, weights, vocabulary_max):
        if len(words) < vocabulary_max * self.trigger_ratio:
            return words, weights

        # Tail optimization does not help with very large vocabularies
        if len(words) > vocabulary_max * 2:
            indices = numpy.argpartition(weights, len(weights) - vocabulary_max)
            indices = indices[-vocabulary_max:]
            words = words[indices]
            weights = weights[indices]
            return words, weights

        # Vocabulary typically consists of these three parts:
        # 1) the core - we found it's border - `core_end` - 15%
        # 2) the body - 70%
        # 3) the minor tail - 15%
        # (1) and (3) are roughly the same size
        # (3) can be safely discarded, (2) can be discarded with care,
        # (1) shall never be discarded.

        sorter = numpy.argsort(weights)[::-1]
        weights = weights[sorter]
        trend_start = int(len(weights) * 0.2)
        trend_finish = int(len(weights) * 0.8)
        z = numpy.polyfit(numpy.arange(trend_start, trend_finish),
                          numpy.log(weights[trend_start:trend_finish]),
                          1)
        exp_z = numpy.exp(z[1] + z[0] * numpy.arange(len(weights)))
        avg_error = numpy.abs(weights[trend_start:trend_finish]
                              - exp_z[trend_start:trend_finish]).mean()
        tail_size = numpy.argmax((numpy.abs(weights - exp_z) < avg_error)[::-1])
        weights = weights[:-tail_size][:vocabulary_max]
        words = words[sorter[:-tail_size]][:vocabulary_max]

        return words, weights


class WMD(object):
    def __init__(self, embeddings, nbow, vocabulary_min=50, vocabulary_max=500,
                 vocabulary_optimizer=TailVocabularyOptimizer(),
                 verbosity=logging.INFO):
        self._relax_cache = None
        self._exact_cache = None
        self._centroid_cache = None
        self.embeddings = embeddings
        self.nbow = nbow
        self.vocabulary_min = vocabulary_min
        self.vocabulary_max = vocabulary_max
        self.vocabulary_optimizer = vocabulary_optimizer
        self._log = logging.getLogger("WMD")
        self._log.level = logging.Logger("", verbosity).level

    def __del__(self):
        if self._relax_cache is not None:
            libwmdrelax.emd_relaxed_cache_fini(self._relax_cache)
        if self._exact_cache is not None:
            libwmdrelax.emd_cache_fini(self._exact_cache)

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value):
        if not hasattr(value, "__getitem__"):
            raise TypeError("embeddings must support [] indexing")
        self._embeddings = value
        self._reset_caches()

    @property
    def nbow(self):
        return self._nbow

    @nbow.setter
    def nbow(self, value):
        if not hasattr(value, "__iter__") or not hasattr(value, "__getitem__"):
            raise TypeError("nbow must be iterable and support [] indexing")
        self._nbow = value
        self._reset_caches()

    @property
    def vocabulary_min(self):
        return self._vocabulary_min

    @vocabulary_min.setter
    def vocabulary_min(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError("vocabulary_min must be > 0 (got %d)" % value)
        self._vocabulary_min = value
        self._reset_caches()

    @property
    def vocabulary_max(self):
        return self._vocabulary_max

    @vocabulary_max.setter
    def vocabulary_max(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError("vocabulary_max must be > 0 (got %d)" % value)
        self._vocabulary_max = value
        if self._relax_cache is not None:
            libwmdrelax.emd_relaxed_cache_fini(self._relax_cache)
        self._relax_cache = libwmdrelax.emd_relaxed_cache_init(value * 2)
        if self._exact_cache is not None:
            libwmdrelax.emd_cache_fini(self._exact_cache)
        self._exact_cache = libwmdrelax.emd_cache_init(value * 2)
        self._reset_caches()

    @property
    def vocabulary_optimizer(self):
        return self._vocabulary_optimizer

    @vocabulary_optimizer.setter
    def vocabulary_optimizer(self, value):
        if not callable(value) and value is not None:
            raise ValueError("vocabulary_optimizer must be a callable")
        self._vocabulary_optimizer = value
        self._reset_caches()

    def _reset_caches(self):
        self._centroid_cache = None

    def _get_vocabulary(self, index):
        _, words, weights = self.nbow[index]
        if self.vocabulary_optimizer is not None:
            words, weights = self.vocabulary_optimizer(
                words, weights, self.vocabulary_max)
        else:
            words = words[:self.vocabulary_max]
            weights = weights[:self.vocabulary_max]
        return words, weights

    def _common_vocabulary_batch(self, words1, weights1, i2):
        words2, weights2 = self._get_vocabulary(i2)
        joint, index = numpy.unique(numpy.concatenate((words1, words2)),
                                    return_index=True)
        nw1 = numpy.zeros(len(joint), dtype=numpy.float32)
        cmp = index < len(words1)
        nw1[numpy.nonzero(cmp)] = weights1[index[cmp]]
        nw2 = numpy.zeros(len(joint), dtype=numpy.float32)
        nw2[numpy.searchsorted(joint, words2)] = weights2
        return joint, nw1, nw2

    def _get_centroid(self, words, weights):
        n = weights.sum()
        if n <= 0 or len(words) < self.vocabulary_min:
            return None
        wsum = (self.embeddings[words] * weights[:, numpy.newaxis]).sum(axis=0)
        return wsum / n

    def _get_centroid_by_index(self, index):
        words, weights = self._get_vocabulary(index)
        return self._get_centroid(words, weights)

    def _estimate_WMD_centroid_batch(self, avg1, i2):
        avg2 = self._get_centroid_by_index(i2)
        if avg2 is None:
            return avg2
        return numpy.linalg.norm(avg1 - avg2)

    def _estimate_WMD_relaxation_batch(self, words1, weights1, i2):
        joint, w1, w2 = self._common_vocabulary_batch(words1, weights1, i2)
        w1 /= w1.sum()
        w2 /= w2.sum()
        evec = self.embeddings[joint]
        evec_sqr = (evec * evec).sum(axis=1)
        dists = evec_sqr - 2 * evec.dot(evec.T) + evec_sqr[:, numpy.newaxis]
        dists[dists < 0] = 0
        dists = numpy.sqrt(dists)
        return libwmdrelax.emd_relaxed(w1, w2, dists, self._relax_cache), \
               w1, w2, dists

    def _WMD_batch(self, words1, weights1, i2):
        joint, w1, w2 = self._common_vocabulary_batch(words1, weights1, i2)
        w1 /= w1.sum()
        w2 /= w2.sum()
        evec = self.embeddings[joint]
        evec_sqr = (evec * evec).sum(axis=1)
        dists = evec_sqr - 2 * evec.dot(evec.T) + evec_sqr[:, numpy.newaxis]
        dists[dists < 0] = 0
        dists = numpy.sqrt(dists)
        return libwmdrelax.emd(w1, w2, dists, self._exact_cache)

    def cache_centroids(self):
        keys = []
        _, words, _ = self.nbow[next(iter(self.nbow))]
        centroids = numpy.zeros(
            (sum(1 for _ in self.nbow), self.embeddings[words[0]].shape[0]),
            dtype=numpy.float32)
        for i, key in enumerate(self.nbow):
            centroid = self._get_centroid_by_index(key)
            if centroid is not None:
                centroids[i] = centroid
            else:
                key = None
            keys.append(key)
        keys = numpy.array(keys)
        self._centroid_cache = (keys, centroids)

    def nearest_neighbors(self, origin, k=10, early_stop=0.5, max_time=3600):
        if isinstance(origin, (tuple, list)):
            words, weights = origin
            index = None
            avg = self._get_centroid(words, weights)
        else:
            index = origin
            words, weights = self._get_vocabulary(index)
            avg = self._get_centroid_by_index(index)
        if avg is None:
            raise ValueError(
                "Too little vocabulary for %d: %d" % (index, len(words)))
        self._log.info("Vocabulary size: %d %d",
                       len(words), self.vocabulary_max)
        self._log.info("WCD")
        ts = time()
        if self._centroid_cache is None:
            queue = []
            for i2 in self.nbow:
                if i2 == index:
                    continue
                d = self._estimate_WMD_centroid_batch(avg, i2)
                if d is not None:
                    queue.append((d, i2))
            queue.sort()
        else:
            keys, centroids = self._centroid_cache
            dists = numpy.linalg.norm(centroids - avg, axis=-1)
            queue = [(None, k) for k in keys[numpy.argsort(dists)]
                     if k is not None]
        self._log.info("%.1f", time() - ts)
        self._log.info("First K WMD")
        ts = time()
        neighbors = [(-self._WMD_batch(words, weights, i2), i2)
                     for (_, i2) in queue[:k]]
        heapq.heapify(neighbors)
        self._log.info("%s", neighbors)
        self._log.info("%.1f", time() - ts)
        self._log.info("P&P")
        skipped = estimated_d = 0
        ppts = time()
        for progress, (_, i2) in enumerate(queue[k:int(len(queue) * early_stop)]):
            if progress % 10 == 0 and time() - ppts > 60:
                self._log.info(
                    "%s %s %s %s %s", progress, skipped / max(progress, 1),
                    estimated_d, neighbors[:3],
                    [self.nbow[n[1]][0] for n in neighbors[-3:]])
                ppts = time()
                if ppts - ts > max_time:
                    break
            estimated_d, w1, w2, dists = self._estimate_WMD_relaxation_batch(
                words, weights, i2)
            farthest = -neighbors[0][0]
            if farthest == 0:
                break
            if estimated_d >= farthest:
                skipped += 1
                continue
            d = libwmdrelax.emd(w1, w2, dists, self._exact_cache)
            if d < farthest:
                heapq.heapreplace(neighbors, (-d, i2))
        neighbors = [(-n[0], n[1]) for n in neighbors]
        neighbors.sort()
        return [(n[1], n[0]) for n in neighbors]
