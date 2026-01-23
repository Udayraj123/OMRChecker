import type {
  double,
  InputArray,
  InputOutputArray,
  int,
  OutputArray,
  TermCriteria,
  _EqPredicate,
} from "./_types";
/*
 * # Clustering
 *
 */
/**
 * The function kmeans implements a k-means algorithm that finds the centers of cluster_count clusters
 * and groups the input samples around the clusters. As an output, `$\\texttt{bestLabels}_i$` contains
 * a 0-based cluster index for the sample stored in the `$i^{th}$` row of the samples matrix.
 *
 * (Python) An example on K-means clustering can be found at
 * opencv_source_code/samples/python/kmeans.py
 *
 * The function returns the compactness measure that is computed as `\\[\\sum _i \\| \\texttt{samples}
 * _i - \\texttt{centers} _{ \\texttt{labels} _i} \\| ^2\\]` after every attempt. The best (minimum)
 * value is chosen and the corresponding labels and the compactness value are returned by the function.
 * Basically, you can use only the core of the function, set the number of attempts to 1, initialize
 * labels each time using a custom algorithm, pass them with the ( flags = [KMEANS_USE_INITIAL_LABELS]
 * ) flag, and then choose the best (most-compact) clustering.
 *
 * @param data Data for clustering. An array of N-Dimensional points with float coordinates is needed.
 * Examples of this array can be:
 * Mat points(count, 2, CV_32F);Mat points(count, 1, CV_32FC2);Mat points(1, count,
 * CV_32FC2);std::vector<cv::Point2f> points(sampleCount);
 *
 * @param K Number of clusters to split the set by.
 *
 * @param bestLabels Input/output integer array that stores the cluster indices for every sample.
 *
 * @param criteria The algorithm termination criteria, that is, the maximum number of iterations and/or
 * the desired accuracy. The accuracy is specified as criteria.epsilon. As soon as each of the cluster
 * centers moves by less than criteria.epsilon on some iteration, the algorithm stops.
 *
 * @param attempts Flag to specify the number of times the algorithm is executed using different
 * initial labellings. The algorithm returns the labels that yield the best compactness (see the last
 * function parameter).
 *
 * @param flags Flag that can take values of cv::KmeansFlags
 *
 * @param centers Output matrix of the cluster centers, one row per each cluster center.
 */
export declare function kmeans(
  data: InputArray,
  K: int,
  bestLabels: InputOutputArray,
  criteria: TermCriteria,
  attempts: int,
  flags: int,
  centers?: OutputArray,
): double;

/**
 * The generic function partition implements an `$O(N^2)$` algorithm for splitting a set of `$N$`
 * elements into one or more equivalency classes, as described in  . The function returns the number of
 * equivalency classes.
 *
 * @param _vec Set of elements stored as a vector.
 *
 * @param labels Output vector of labels. It contains as many elements as vec. Each label labels[i] is
 * a 0-based cluster index of vec[i].
 *
 * @param predicate Equivalence predicate (pointer to a boolean function of two arguments or an
 * instance of the class that has the method bool operator()(const _Tp& a, const _Tp& b) ). The
 * predicate returns true when the elements are certainly in the same class, and returns false if they
 * may or may not be in the same class.
 */
export declare function partition(
  arg119: any,
  arg120: any,
  _vec: any,
  labels: any,
  predicate?: _EqPredicate,
): any;
