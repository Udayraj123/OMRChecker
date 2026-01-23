import type {
  double,
  FileNode,
  FileStorage,
  InputArray,
  int,
  Mat,
  OutputArray,
} from "./_types";

/**
 * The class is used to calculate a special basis for a set of vectors. The basis will consist of
 * eigenvectors of the covariance matrix calculated from the input set of vectors. The class PCA can
 * also transform vectors to/from the new coordinate space defined by the basis. Usually, in this new
 * coordinate system, each vector from the original set (and any linear combination of such vectors)
 * can be quite accurately approximated by taking its first few components, corresponding to the
 * eigenvectors of the largest eigenvalues of the covariance matrix. Geometrically it means that you
 * calculate a projection of the vector to a subspace formed by a few eigenvectors corresponding to the
 * dominant eigenvalues of the covariance matrix. And usually such a projection is very close to the
 * original vector. So, you can represent the original vector from a high-dimensional space with a much
 * shorter vector consisting of the projected vector's coordinates in the subspace. Such a
 * transformation is also known as Karhunen-Loeve Transform, or KLT. See
 *
 * The sample below is the function that takes two matrices. The first function stores a set of vectors
 * (a row per vector) that is used to calculate [PCA](#d3/d8d/classcv_1_1PCA}). The second function
 * stores another "test" set of vectors (a row per vector). First, these vectors are compressed with
 * [PCA](#d3/d8d/classcv_1_1PCA}), then reconstructed back, and then the reconstruction error norm is
 * computed and printed for each vector. :
 *
 * ```cpp
 * using namespace cv;
 *
 * PCA compressPCA(const Mat& pcaset, int maxComponents,
 *                 const Mat& testset, Mat& compressed)
 * {
 *     PCA pca(pcaset, // pass the data
 *             Mat(), // we do not have a pre-computed mean vector,
 *                    // so let the PCA engine to compute it
 *             PCA::DATA_AS_ROW, // indicate that the vectors
 *                                 // are stored as matrix rows
 *                                 // (use PCA::DATA_AS_COL if the vectors are
 *                                 // the matrix columns)
 *             maxComponents // specify, how many principal components to retain
 *             );
 *     // if there is no test data, just return the computed basis, ready-to-use
 *     if( !testset.data )
 *         return pca;
 *     CV_Assert( testset.cols == pcaset.cols );
 *
 *     compressed.create(testset.rows, maxComponents, testset.type());
 *
 *     Mat reconstructed;
 *     for( int i = 0; i < testset.rows; i++ )
 *     {
 *         Mat vec = testset.row(i), coeffs = compressed.row(i), reconstructed;
 *         // compress the vector, the result will be stored
 *         // in the i-th row of the output matrix
 *         pca.project(vec, coeffs);
 *         // and then reconstruct it
 *         pca.backProject(coeffs, reconstructed);
 *         // and measure the error
 *         printf("%d. diff = %g\\n", i, norm(vec, reconstructed, NORM_L2));
 *     }
 *     return pca;
 * }
 * ```
 *
 * [calcCovarMatrix](#d2/de8/group__core__array_1gae6ffa9354633f984246945d52823165d}),
 * [mulTransposed](#d2/de8/group__core__array_1gadc4e49f8f7a155044e3be1b9e3b270ab}),
 * [SVD](#df/df7/classcv_1_1SVD}),
 * [dft](#d2/de8/group__core__array_1gadd6cf9baf2b8b704a11b5f04aaf4f39d}),
 * [dct](#d2/de8/group__core__array_1ga85aad4d668c01fbd64825f589e3696d4})
 *
 * Source:
 * [opencv2/core.hpp](https://github.com/opencv/opencv/tree/master/modules/core/include/opencv2/core.hpp#L2393).
 *
 */
export declare class PCA {
  public eigenvalues: Mat;

  public eigenvectors: Mat;

  public mean: Mat;

  /**
   *   The default constructor initializes an empty PCA structure. The other constructors initialize the
   * structure and call [PCA::operator()()].
   */
  public constructor();

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param data input samples stored as matrix rows or matrix columns.
   *
   * @param mean optional mean value; if the matrix is empty (noArray()), the mean is computed from the
   * data.
   *
   * @param flags operation flags; currently the parameter is only used to specify the data layout
   * (PCA::Flags)
   *
   * @param maxComponents maximum number of components that PCA should retain; by default, all the
   * components are retained.
   */
  public constructor(
    data: InputArray,
    mean: InputArray,
    flags: int,
    maxComponents?: int,
  );

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param data input samples stored as matrix rows or matrix columns.
   *
   * @param mean optional mean value; if the matrix is empty (noArray()), the mean is computed from the
   * data.
   *
   * @param flags operation flags; currently the parameter is only used to specify the data layout
   * (PCA::Flags)
   *
   * @param retainedVariance Percentage of variance that PCA should retain. Using this parameter will
   * let the PCA decided how many components to retain but it will always keep at least 2.
   */
  public constructor(
    data: InputArray,
    mean: InputArray,
    flags: int,
    retainedVariance: double,
  );

  /**
   *   The methods are inverse operations to [PCA::project]. They take PC coordinates of projected
   * vectors and reconstruct the original vectors. Unless all the principal components have been
   * retained, the reconstructed vectors are different from the originals. But typically, the difference
   * is small if the number of components is large enough (but still much smaller than the original
   * vector dimensionality). As a result, [PCA] is used.
   *
   * @param vec coordinates of the vectors in the principal component subspace, the layout and size are
   * the same as of PCA::project output vectors.
   */
  public backProject(vec: InputArray): Mat;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param vec coordinates of the vectors in the principal component subspace, the layout and size are
   * the same as of PCA::project output vectors.
   *
   * @param result reconstructed vectors; the layout and size are the same as of PCA::project input
   * vectors.
   */
  public backProject(vec: InputArray, result: OutputArray): InputArray;

  /**
   *   The methods project one or more vectors to the principal component subspace, where each vector
   * projection is represented by coefficients in the principal component basis. The first form of the
   * method returns the matrix that the second form writes to the result. So the first form can be used
   * as a part of expression while the second form can be more efficient in a processing loop.
   *
   * @param vec input vector(s); must have the same dimensionality and the same layout as the input
   * data used at PCA phase, that is, if DATA_AS_ROW are specified, then vec.cols==data.cols (vector
   * dimensionality) and vec.rows is the number of vectors to project, and the same is true for the
   * PCA::DATA_AS_COL case.
   */
  public project(vec: InputArray): Mat;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param vec input vector(s); must have the same dimensionality and the same layout as the input
   * data used at PCA phase, that is, if DATA_AS_ROW are specified, then vec.cols==data.cols (vector
   * dimensionality) and vec.rows is the number of vectors to project, and the same is true for the
   * PCA::DATA_AS_COL case.
   *
   * @param result output vectors; in case of PCA::DATA_AS_COL, the output matrix has as many columns
   * as the number of input vectors, this means that result.cols==vec.cols and the number of rows match
   * the number of principal components (for example, maxComponents parameter passed to the constructor).
   */
  public project(vec: InputArray, result: OutputArray): InputArray;

  /**
   *   Loads [eigenvalues] [eigenvectors] and [mean] from specified [FileNode]
   */
  public read(fn: FileNode): FileNode;

  /**
   *   Writes [eigenvalues] [eigenvectors] and [mean] to specified [FileStorage]
   */
  public write(fs: FileStorage): FileStorage;
}

export declare const DATA_AS_ROW: Flags; // initializer: = 0

export declare const DATA_AS_COL: Flags; // initializer: = 1

export declare const USE_AVG: Flags; // initializer: = 2

export type Flags = any;
