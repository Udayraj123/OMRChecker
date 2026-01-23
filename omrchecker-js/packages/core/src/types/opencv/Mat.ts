import type {
  AccessFlag,
  bool,
  double,
  EmscriptenEmbindInstance,
  InputArray,
  int,
  MatAllocator,
  MatCommaInitializer_,
  MatConstIterator_,
  MatExpr,
  MatIterator_,
  MatSize,
  MatStep,
  Matx,
  OutputArray,
  Point,
  Point3_,
  Point_,
  Rect,
  Scalar,
  Size,
  size_t,
  typename,
  uchar,
  UMat,
  UMatData,
  UMatUsageFlags,
  Vec,
} from "./_types";

/**
 * <a name="d3/d63/classcv_1_1Mat_1CVMat_Details"></a> The class [Mat](#d3/d63/classcv_1_1Mat})
 * represents an n-dimensional dense numerical single-channel or multi-channel array. It can be used to
 * store real or complex-valued vectors and matrices, grayscale or color images, voxel volumes, vector
 * fields, point clouds, tensors, histograms (though, very high-dimensional histograms may be better
 * stored in a [SparseMat](#dd/da9/classcv_1_1SparseMat}) ). The data layout of the array `M` is
 * defined by the array `M.step[]`, so that the address of element `$(i_0,...,i_{M.dims-1})$`, where
 * `$0\\leq i_k<M.size[k]$`, is computed as: `\\[addr(M_{i_0,...,i_{M.dims-1}}) = M.data +
 * M.step[0]*i_0 + M.step[1]*i_1 + ... + M.step[M.dims-1]*i_{M.dims-1}\\]` In case of a 2-dimensional
 * array, the above formula is reduced to: `\\[addr(M_{i,j}) = M.data + M.step[0]*i + M.step[1]*j\\]`
 * Note that `M.step[i] >= M.step[i+1]` (in fact, `M.step[i] >= M.step[i+1]*M.size[i+1]` ). This means
 * that 2-dimensional matrices are stored row-by-row, 3-dimensional matrices are stored plane-by-plane,
 * and so on. M.step[M.dims-1] is minimal and always equal to the element size M.elemSize() .
 *
 * So, the data layout in [Mat](#d3/d63/classcv_1_1Mat}) is compatible with the majority of dense array
 * types from the standard toolkits and SDKs, such as Numpy (ndarray), Win32 (independent device
 * bitmaps), and others, that is, with any array that uses *steps* (or *strides*) to compute the
 * position of a pixel. Due to this compatibility, it is possible to make a
 * [Mat](#d3/d63/classcv_1_1Mat}) header for user-allocated data and process it in-place using OpenCV
 * functions.
 *
 * There are many different ways to create a [Mat](#d3/d63/classcv_1_1Mat}) object. The most popular
 * options are listed below:
 *
 * Use the create(nrows, ncols, type) method or the similar Mat(nrows, ncols, type[, fillValue])
 * constructor. A new array of the specified size and type is allocated. type has the same meaning as
 * in the cvCreateMat method. For example, CV_8UC1 means a 8-bit single-channel array, CV_32FC2 means a
 * 2-channel (complex) floating-point array, and so on.
 *
 * ```cpp
 * // make a 7x7 complex matrix filled with 1+3j.
 * Mat M(7,7,CV_32FC2,Scalar(1,3));
 * // and now turn M to a 100x60 15-channel 8-bit matrix.
 * // The old content will be deallocated
 * M.create(100,60,CV_8UC(15));
 * ```
 *
 *  As noted in the introduction to this chapter,
 * [create()](#d3/d63/classcv_1_1Mat_1a55ced2c8d844d683ea9a725c60037ad0}) allocates only a new array
 * when the shape or type of the current array are different from the specified ones.
 * Create a multi-dimensional array:
 *
 * ```cpp
 * // create a 100x100x100 8-bit array
 * int sz[] = {100, 100, 100};
 * Mat bigCube(3, sz, CV_8U, Scalar::all(0));
 * ```
 *
 *  It passes the number of dimensions =1 to the [Mat](#d3/d63/classcv_1_1Mat}) constructor but the
 * created array will be 2-dimensional with the number of columns set to 1. So,
 * [Mat::dims](#d3/d63/classcv_1_1Mat_1a39cf614aa52567e9a945cd2609bd767b}) is always >= 2 (can also be
 * 0 when the array is empty).
 * Use a copy constructor or assignment operator where there can be an array or expression on the right
 * side (see below). As noted in the introduction, the array assignment is an O(1) operation because it
 * only copies the header and increases the reference counter. The
 * [Mat::clone()](#d3/d63/classcv_1_1Mat_1adff2ea98da45eae0833e73582dd4a660}) method can be used to get
 * a full (deep) copy of the array when you need it.
 * Construct a header for a part of another array. It can be a single row, single column, several rows,
 * several columns, rectangular region in the array (called a *minor* in algebra) or a diagonal. Such
 * operations are also O(1) because the new header references the same data. You can actually modify a
 * part of the array using this feature, for example:
 *
 * ```cpp
 * // add the 5-th row, multiplied by 3 to the 3rd row
 * M.row(3) = M.row(3) + M.row(5)*3;
 * // now copy the 7-th column to the 1-st column
 * // M.col(1) = M.col(7); // this will not work
 * Mat M1 = M.col(1);
 * M.col(7).copyTo(M1);
 * // create a new 320x240 image
 * Mat img(Size(320,240),CV_8UC3);
 * // select a ROI
 * Mat roi(img, Rect(10,10,100,100));
 * // fill the ROI with (0,255,0) (which is green in RGB space);
 * // the original 320x240 image will be modified
 * roi = Scalar(0,255,0);
 * ```
 *
 *  Due to the additional datastart and dataend members, it is possible to compute a relative sub-array
 * position in the main *container* array using
 * [locateROI()](#d3/d63/classcv_1_1Mat_1a40b5b3371a9c2a4b2b8ce0c8068d7c96}):
 *
 * ```cpp
 * Mat A = Mat::eye(10, 10, CV_32S);
 * // extracts A columns, 1 (inclusive) to 3 (exclusive).
 * Mat B = A(Range::all(), Range(1, 3));
 * // extracts B rows, 5 (inclusive) to 9 (exclusive).
 * // that is, C \\~ A(Range(5, 9), Range(1, 3))
 * Mat C = B(Range(5, 9), Range::all());
 * Size size; Point ofs;
 * C.locateROI(size, ofs);
 * // size will be (width=10,height=10) and the ofs will be (x=1, y=5)
 * ```
 *
 *  As in case of whole matrices, if you need a deep copy, use the
 * `[clone()](#d3/d63/classcv_1_1Mat_1adff2ea98da45eae0833e73582dd4a660})` method of the extracted
 * sub-matrices.
 * Make a header for user-allocated data. It can be useful to do the following:
 *
 * Process "foreign" data using OpenCV (for example, when you implement a DirectShow* filter or a
 * processing module for gstreamer, and so on). For example:
 *
 * ```cpp
 * void process_video_frame(const unsigned char* pixels,
 *                          int width, int height, int step)
 * {
 *     Mat img(height, width, CV_8UC3, pixels, step);
 *     GaussianBlur(img, img, Size(7,7), 1.5, 1.5);
 * }
 * ```
 *
 * Quickly initialize small matrices and/or get a super-fast element access.
 *
 * ```cpp
 * double m[3][3] = {{a, b, c}, {d, e, f}, {g, h, i}};
 * Mat M = Mat(3, 3, CV_64F, m).inv();
 * ```
 *
 * Use MATLAB-style array initializers,
 * [zeros()](#d3/d63/classcv_1_1Mat_1a0b57b6a326c8876d944d188a46e0f556}),
 * [ones()](#d3/d63/classcv_1_1Mat_1a69ae0402d116fc9c71908d8508dc2f09}),
 * [eye()](#d3/d63/classcv_1_1Mat_1a2cf9b9acde7a9852542bbc20ef851ed2}), for example:
 *
 * ```cpp
 * // create a double-precision identity matrix and add it to M.
 * M += Mat::eye(M.rows, M.cols, CV_64F);
 * ```
 *
 * Use a comma-separated initializer:
 *
 * ```cpp
 * // create a 3x3 double-precision identity matrix
 * Mat M = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
 * ```
 *
 *  With this approach, you first call a constructor of the [Mat](#d3/d63/classcv_1_1Mat}) class with
 * the proper parameters, and then you just put `<< operator` followed by comma-separated values that
 * can be constants, variables, expressions, and so on. Also, note the extra parentheses required to
 * avoid compilation errors.
 *
 * Once the array is created, it is automatically managed via a reference-counting mechanism. If the
 * array header is built on top of user-allocated data, you should handle the data by yourself. The
 * array data is deallocated when no one points to it. If you want to release the data pointed by a
 * array header before the array destructor is called, use
 * [Mat::release()](#d3/d63/classcv_1_1Mat_1ae48d4913285518e2c21a3457017e716e}).
 *
 * The next important thing to learn about the array class is element access. This manual already
 * described how to compute an address of each array element. Normally, you are not required to use the
 * formula directly in the code. If you know the array element type (which can be retrieved using the
 * method [Mat::type()](#d3/d63/classcv_1_1Mat_1af2d2652e552d7de635988f18a84b53e5}) ), you can access
 * the element `$M_{ij}$` of a 2-dimensional array as:
 *
 * ```cpp
 * M.at<double>(i,j) += 1.f;
 * ```
 *
 *  assuming that `M` is a double-precision floating-point array. There are several variants of the
 * method at for a different number of dimensions.
 *
 * If you need to process a whole row of a 2D array, the most efficient way is to get the pointer to
 * the row first, and then just use the plain C operator [] :
 *
 * ```cpp
 * // compute sum of positive matrix elements
 * // (assuming that M is a double-precision matrix)
 * double sum=0;
 * for(int i = 0; i < M.rows; i++)
 * {
 *     const double* Mi = M.ptr<double>(i);
 *     for(int j = 0; j < M.cols; j++)
 *         sum += std::max(Mi[j], 0.);
 * }
 * ```
 *
 *  Some operations, like the one above, do not actually depend on the array shape. They just process
 * elements of an array one by one (or elements from multiple arrays that have the same coordinates,
 * for example, array addition). Such operations are called *element-wise*. It makes sense to check
 * whether all the input/output arrays are continuous, namely, have no gaps at the end of each row. If
 * yes, process them as a long single row:
 *
 * ```cpp
 * // compute the sum of positive matrix elements, optimized variant
 * double sum=0;
 * int cols = M.cols, rows = M.rows;
 * if(M.isContinuous())
 * {
 *     cols *= rows;
 *     rows = 1;
 * }
 * for(int i = 0; i < rows; i++)
 * {
 *     const double* Mi = M.ptr<double>(i);
 *     for(int j = 0; j < cols; j++)
 *         sum += std::max(Mi[j], 0.);
 * }
 * ```
 *
 *  In case of the continuous matrix, the outer loop body is executed just once. So, the overhead is
 * smaller, which is especially noticeable in case of small matrices.
 *
 * Finally, there are STL-style iterators that are smart enough to skip gaps between successive rows:
 *
 * ```cpp
 * // compute sum of positive matrix elements, iterator-based variant
 * double sum=0;
 * MatConstIterator_<double> it = M.begin<double>(), it_end = M.end<double>();
 * for(; it != it_end; ++it)
 *     sum += std::max(*it, 0.);
 * ```
 *
 *  The matrix iterators are random-access iterators, so they can be passed to any STL algorithm,
 * including [std::sort()](#d2/de8/group__core__array_1ga45dd56da289494ce874be2324856898f}).
 *
 * Matrix Expressions and arithmetic see [MatExpr](#d1/d10/classcv_1_1MatExpr})
 *
 * Source:
 * [opencv2/core/mat.hpp](https://github.com/opencv/opencv/tree/master/modules/core/include/opencv2/core/mat.hpp#L2073).
 *
 */
export declare class Mat extends EmscriptenEmbindInstance {
  public allocator: MatAllocator;

  public cols: int;

  public data: Uint8Array;
  public data8S: Int8Array;
  public data8U: Uint8Array;
  public data16U: Uint16Array;
  public data16S: Int16Array;
  public data32U: Uint32Array;
  public data32S: Int32Array;
  public data32F: Float32Array;
  public data64F: Float64Array;

  public dataend: uchar;

  public datalimit: uchar;

  public datastart: uchar;

  public dims: int;

  /**
   *   includes several bit-fields:
   *
   * the magic signature
   * continuity flag
   * depth
   * number of channels
   *
   */
  public flags: int;

  public rows: int;

  public size: MatSize;

  public step: MatStep;

  public u: UMatData;

  /**
   *   These are various constructors that form a matrix. As noted in the AutomaticAllocation, often the
   * default constructor is enough, and the proper matrix will be allocated by an OpenCV function. The
   * constructed matrix can further be assigned to another matrix or matrix expression or can be
   * allocated with [Mat::create] . In the former case, the old content is de-referenced.
   */
  public constructor();

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param rows Number of rows in a 2D array.
   *
   * @param cols Number of columns in a 2D array.
   *
   * @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n),
   * ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
   */
  public constructor(rows: int, cols: int, type: int);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and
   * the number of columns go in the reverse order.
   *
   * @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n),
   * ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
   */
  public constructor(size: Size, type: int);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param rows Number of rows in a 2D array.
   *
   * @param cols Number of columns in a 2D array.
   *
   * @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n),
   * ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
   *
   * @param s An optional value to initialize each matrix element with. To set all the matrix elements
   * to the particular value after the construction, use the assignment operator Mat::operator=(const
   * Scalar& value) .
   */
  public constructor(rows: int, cols: int, type: int, s: Scalar);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and
   * the number of columns go in the reverse order.
   *
   * @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n),
   * ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
   *
   * @param s An optional value to initialize each matrix element with. To set all the matrix elements
   * to the particular value after the construction, use the assignment operator Mat::operator=(const
   * Scalar& value) .
   */
  public constructor(size: Size, type: int, s: Scalar);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param ndims Array dimensionality.
   *
   * @param sizes Array of integers specifying an n-dimensional array shape.
   *
   * @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n),
   * ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
   */
  public constructor(ndims: int, sizes: any, type: int);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param sizes Array of integers specifying an n-dimensional array shape.
   *
   * @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n),
   * ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
   */
  public constructor(sizes: any, type: int);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param ndims Array dimensionality.
   *
   * @param sizes Array of integers specifying an n-dimensional array shape.
   *
   * @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n),
   * ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
   *
   * @param s An optional value to initialize each matrix element with. To set all the matrix elements
   * to the particular value after the construction, use the assignment operator Mat::operator=(const
   * Scalar& value) .
   */
  public constructor(ndims: int, sizes: any, type: int, s: Scalar);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param sizes Array of integers specifying an n-dimensional array shape.
   *
   * @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n),
   * ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
   *
   * @param s An optional value to initialize each matrix element with. To set all the matrix elements
   * to the particular value after the construction, use the assignment operator Mat::operator=(const
   * Scalar& value) .
   */
  public constructor(sizes: any, type: int, s: Scalar);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is
   * copied by these constructors. Instead, the header pointing to m data or its sub-array is constructed
   * and associated with it. The reference counter, if any, is incremented. So, when you modify the
   * matrix formed using such a constructor, you also modify the corresponding elements of m . If you
   * want to have an independent copy of the sub-array, use Mat::clone() .
   */
  public constructor(m: Mat);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param rows Number of rows in a 2D array.
   *
   * @param cols Number of columns in a 2D array.
   *
   * @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n),
   * ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
   *
   * @param data Pointer to the user data. Matrix constructors that take data and step parameters do
   * not allocate matrix data. Instead, they just initialize the matrix header that points to the
   * specified data, which means that no data is copied. This operation is very efficient and can be used
   * to process external data using OpenCV functions. The external data is not automatically deallocated,
   * so you should take care of it.
   *
   * @param step Number of bytes each matrix row occupies. The value should include the padding bytes
   * at the end of each row, if any. If the parameter is missing (set to AUTO_STEP ), no padding is
   * assumed and the actual step is calculated as cols*elemSize(). See Mat::elemSize.
   */
  public constructor(rows: int, cols: int, type: int, data: any, step?: size_t);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and
   * the number of columns go in the reverse order.
   *
   * @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n),
   * ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
   *
   * @param data Pointer to the user data. Matrix constructors that take data and step parameters do
   * not allocate matrix data. Instead, they just initialize the matrix header that points to the
   * specified data, which means that no data is copied. This operation is very efficient and can be used
   * to process external data using OpenCV functions. The external data is not automatically deallocated,
   * so you should take care of it.
   *
   * @param step Number of bytes each matrix row occupies. The value should include the padding bytes
   * at the end of each row, if any. If the parameter is missing (set to AUTO_STEP ), no padding is
   * assumed and the actual step is calculated as cols*elemSize(). See Mat::elemSize.
   */
  public constructor(size: Size, type: int, data: any, step?: size_t);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param ndims Array dimensionality.
   *
   * @param sizes Array of integers specifying an n-dimensional array shape.
   *
   * @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n),
   * ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
   *
   * @param data Pointer to the user data. Matrix constructors that take data and step parameters do
   * not allocate matrix data. Instead, they just initialize the matrix header that points to the
   * specified data, which means that no data is copied. This operation is very efficient and can be used
   * to process external data using OpenCV functions. The external data is not automatically deallocated,
   * so you should take care of it.
   *
   * @param steps Array of ndims-1 steps in case of a multi-dimensional array (the last step is always
   * set to the element size). If not specified, the matrix is assumed to be continuous.
   */
  public constructor(ndims: int, sizes: any, type: int, data: any, steps?: any);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param sizes Array of integers specifying an n-dimensional array shape.
   *
   * @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or CV_8UC(n),
   * ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
   *
   * @param data Pointer to the user data. Matrix constructors that take data and step parameters do
   * not allocate matrix data. Instead, they just initialize the matrix header that points to the
   * specified data, which means that no data is copied. This operation is very efficient and can be used
   * to process external data using OpenCV functions. The external data is not automatically deallocated,
   * so you should take care of it.
   *
   * @param steps Array of ndims-1 steps in case of a multi-dimensional array (the last step is always
   * set to the element size). If not specified, the matrix is assumed to be continuous.
   */
  public constructor(sizes: any, type: int, data: any, steps?: any);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is
   * copied by these constructors. Instead, the header pointing to m data or its sub-array is constructed
   * and associated with it. The reference counter, if any, is incremented. So, when you modify the
   * matrix formed using such a constructor, you also modify the corresponding elements of m . If you
   * want to have an independent copy of the sub-array, use Mat::clone() .
   *
   * @param rowRange Range of the m rows to take. As usual, the range start is inclusive and the range
   * end is exclusive. Use Range::all() to take all the rows.
   *
   * @param colRange Range of the m columns to take. Use Range::all() to take all the columns.
   */
  public constructor(m: Mat, rowRange: Range, colRange?: Range);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is
   * copied by these constructors. Instead, the header pointing to m data or its sub-array is constructed
   * and associated with it. The reference counter, if any, is incremented. So, when you modify the
   * matrix formed using such a constructor, you also modify the corresponding elements of m . If you
   * want to have an independent copy of the sub-array, use Mat::clone() .
   *
   * @param roi Region of interest.
   */
  public constructor(m: Mat, roi: Rect);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is
   * copied by these constructors. Instead, the header pointing to m data or its sub-array is constructed
   * and associated with it. The reference counter, if any, is incremented. So, when you modify the
   * matrix formed using such a constructor, you also modify the corresponding elements of m . If you
   * want to have an independent copy of the sub-array, use Mat::clone() .
   *
   * @param ranges Array of selected ranges of m along each dimensionality.
   */
  public constructor(m: Mat, ranges: Range);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is
   * copied by these constructors. Instead, the header pointing to m data or its sub-array is constructed
   * and associated with it. The reference counter, if any, is incremented. So, when you modify the
   * matrix formed using such a constructor, you also modify the corresponding elements of m . If you
   * want to have an independent copy of the sub-array, use Mat::clone() .
   *
   * @param ranges Array of selected ranges of m along each dimensionality.
   */
  public constructor(m: Mat, ranges: Range);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param vec STL vector whose elements form the matrix. The matrix has a single column and the
   * number of rows equal to the number of vector elements. Type of the matrix matches the type of vector
   * elements. The constructor can handle arbitrary types, for which there is a properly declared
   * DataType . This means that the vector elements must be primitive numbers or uni-type numerical
   * tuples of numbers. Mixed-type structures are not supported. The corresponding constructor is
   * explicit. Since STL vectors are not automatically converted to Mat instances, you should write
   * Mat(vec) explicitly. Unless you copy the data into the matrix ( copyData=true ), no new elements
   * will be added to the vector because it can potentially yield vector data reallocation, and, thus,
   * the matrix data pointer will be invalid.
   *
   * @param copyData Flag to specify whether the underlying data of the STL vector should be copied to
   * (true) or shared with (false) the newly constructed matrix. When the data is copied, the allocated
   * buffer is managed using Mat reference counting mechanism. While the data is shared, the reference
   * counter is NULL, and you should not deallocate the data until the matrix is not destructed.
   */
  public constructor(arg3: any, vec: any, copyData?: bool);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public constructor(arg4: any, arg5?: typename, list?: any);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public constructor(arg6: any, sizes: any, list: any);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public constructor(arg7: any, _Nm: size_t, arr: any, copyData?: bool);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public constructor(arg8: any, n: int, vec: Vec, copyData?: bool);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public constructor(arg9: any, m: int, n: int, mtx: Matx, copyData?: bool);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public constructor(arg10: any, pt: Point_, copyData?: bool);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public constructor(arg11: any, pt: Point3_, copyData?: bool);

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public constructor(arg12: any, commaInitializer: MatCommaInitializer_);

  public constructor(m: any);

  public constructor(m: Mat);

  /**
   *   The method increments the reference counter associated with the matrix data. If the matrix header
   * points to an external data set (see [Mat::Mat] ), the reference counter is NULL, and the method has
   * no effect in this case. Normally, to avoid memory leaks, the method should not be called explicitly.
   * It is called implicitly by the matrix assignment operator. The reference counter increment is an
   * atomic operation on the platforms that support it. Thus, it is safe to operate on the same matrices
   * asynchronously in different threads.
   */
  public addref(): void;

  /**
   *   The method is complimentary to [Mat::locateROI] . The typical use of these functions is to
   * determine the submatrix position within the parent matrix and then shift the position somehow.
   * Typically, it can be required for filtering operations when pixels outside of the ROI should be
   * taken into account. When all the method parameters are positive, the ROI needs to grow in all
   * directions by the specified amount, for example:
   *
   *   ```cpp
   *   A.adjustROI(2, 2, 2, 2);
   *   ```
   *
   *    In this example, the matrix size is increased by 4 elements in each direction. The matrix is
   * shifted by 2 elements to the left and 2 elements up, which brings in all the necessary pixels for
   * the filtering with the 5x5 kernel.
   *
   *   adjustROI forces the adjusted ROI to be inside of the parent matrix that is boundaries of the
   * adjusted ROI are constrained by boundaries of the parent matrix. For example, if the submatrix A is
   * located in the first row of a parent matrix and you called A.adjustROI(2, 2, 2, 2) then A will not
   * be increased in the upward direction.
   *
   *   The function is used internally by the OpenCV filtering functions, like filter2D , morphological
   * operations, and so on.
   *
   *   [copyMakeBorder]
   *
   * @param dtop Shift of the top submatrix boundary upwards.
   *
   * @param dbottom Shift of the bottom submatrix boundary downwards.
   *
   * @param dleft Shift of the left submatrix boundary to the left.
   *
   * @param dright Shift of the right submatrix boundary to the right.
   */
  public adjustROI(dtop: int, dbottom: int, dleft: int, dright: int): Mat;
  /**
   *   The methods return the matrix read-only or read-write iterators. The use of matrix iterators is
   * very similar to the use of bi-directional STL iterators. In the example below, the alpha blending
   * function is rewritten using the matrix iterators:
   *
   *   ```cpp
   *   template<typename T>
   *   void alphaBlendRGBA(const Mat& src1, const Mat& src2, Mat& dst)
   *   {
   *       typedef Vec<T, 4> VT;
   *
   *       const float alpha_scale = (float)std::numeric_limits<T>::max(),
   *                   inv_scale = 1.f/alpha_scale;
   *
   *       CV_Assert( src1.type() == src2.type() &&
   *                  src1.type() == traits::Type<VT>::value &&
   *                  src1.size() == src2.size());
   *       Size size = src1.size();
   *       dst.create(size, src1.type());
   *
   *       MatConstIterator_<VT> it1 = src1.begin<VT>(), it1_end = src1.end<VT>();
   *       MatConstIterator_<VT> it2 = src2.begin<VT>();
   *       MatIterator_<VT> dst_it = dst.begin<VT>();
   *
   *       for( ; it1 != it1_end; ++it1, ++it2, ++dst_it )
   *       {
   *           VT pix1 = *it1, pix2 = *it2;
   *           float alpha = pix1[3]*inv_scale, beta = pix2[3]*inv_scale;
   * dst_it = VT(saturate_cast<T>(pix1[0]*alpha + pix2[0]*beta),
   *                        saturate_cast<T>(pix1[1]*alpha + pix2[1]*beta),
   *                        saturate_cast<T>(pix1[2]*alpha + pix2[2]*beta),
   *                        saturate_cast<T>((1 - (1-alpha)*(1-beta))*alpha_scale));
   *       }
   *   }
   *   ```
   */
  public begin(arg25: any): MatIterator_;

  public begin(arg26: any): MatConstIterator_;

  /**
   *   The method returns the number of matrix channels.
   */
  public channels(): int;

  /**
   *   -1 if the requirement is not satisfied. Otherwise, it returns the number of elements in the
   * matrix. Note that an element may have multiple channels.
   *   The following code demonstrates its usage for a 2-d matrix:
   *
   *   ```cpp
   *       cv::Mat mat(20, 1, CV_32FC2);
   *       int n = mat.checkVector(2);
   *       CV_Assert(n == 20); // mat has 20 elements
   *
   *       mat.create(20, 2, CV_32FC1);
   *       n = mat.checkVector(1);
   *       CV_Assert(n == -1); // mat is neither a column nor a row vector
   *
   *       n = mat.checkVector(2);
   *       CV_Assert(n == 20); // the 2 columns are considered as 1 element
   *   ```
   *
   *    The following code demonstrates its usage for a 3-d matrix:
   *
   *   ```cpp
   *       int dims[] = {1, 3, 5}; // 1 plane, every plane has 3 rows and 5 columns
   *       mat.create(3, dims, CV_32FC1); // for 3-d mat, it MUST have only 1 channel
   *       n = mat.checkVector(5); // the 5 columns are considered as 1 element
   *       CV_Assert(n == 3);
   *
   *       int dims2[] = {3, 1, 5}; // 3 planes, every plane has 1 row and 5 columns
   *       mat.create(3, dims2, CV_32FC1);
   *       n = mat.checkVector(5); // the 5 columns are considered as 1 element
   *       CV_Assert(n == 3);
   *   ```
   *
   * @param elemChannels Number of channels or number of columns the matrix should have. For a 2-D
   * matrix, when the matrix has only 1 column, then it should have elemChannels channels; When the
   * matrix has only 1 channel, then it should have elemChannels columns. For a 3-D matrix, it should
   * have only one channel. Furthermore, if the number of planes is not one, then the number of rows
   * within every plane has to be 1; if the number of rows within every plane is not 1, then the number
   * of planes has to be 1.
   *
   * @param depth The depth the matrix should have. Set it to -1 when any depth is fine.
   *
   * @param requireContinuous Set it to true to require the matrix to be continuous
   */
  public checkVector(
    elemChannels: int,
    depth?: int,
    requireContinuous?: bool,
  ): int;

  /**
   *   The method creates a full copy of the array. The original step[] is not taken into account. So,
   * the array copy is a continuous array occupying [total()]*elemSize() bytes.
   */
  public clone(): Mat;

  /**
   *   The method makes a new header for the specified matrix column and returns it. This is an O(1)
   * operation, regardless of the matrix size. The underlying data of the new matrix is shared with the
   * original matrix. See also the [Mat::row] description.
   *
   * @param x A 0-based column index.
   */
  public col(x: int): Mat;

  /**
   *   The method makes a new header for the specified column span of the matrix. Similarly to [Mat::row]
   * and [Mat::col] , this is an O(1) operation.
   *
   * @param startcol An inclusive 0-based start index of the column span.
   *
   * @param endcol An exclusive 0-based ending index of the column span.
   */
  public colRange(startcol: int, endcol: int): Mat;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param r Range structure containing both the start and the end indices.
   */
  public colRange(r: Range): Mat;

  /**
   *   The method converts source pixel values to the target data type. saturate_cast<> is applied at the
   * end to avoid possible overflows:
   *
   *   `\\[m(x,y) = saturate \\_ cast<rType>( \\alpha (*this)(x,y) + \\beta )\\]`
   *
   * @param m output matrix; if it does not have a proper size or type before the operation, it is
   * reallocated.
   *
   * @param rtype desired output matrix type or, rather, the depth since the number of channels are the
   * same as the input has; if rtype is negative, the output matrix will have the same type as the input.
   *
   * @param alpha optional scale factor.
   *
   * @param beta optional delta added to the scaled values.
   */
  public convertTo(
    m: OutputArray,
    rtype: int,
    alpha?: double,
    beta?: double,
  ): OutputArray;

  public copySize(m: Mat): Mat;

  /**
   *   The method copies the matrix data to another matrix. Before copying the data, the method invokes :
   *
   *
   *   ```cpp
   *   m.create(this->size(), this->type());
   *   ```
   *
   *    so that the destination matrix is reallocated if needed. While m.copyTo(m); works flawlessly, the
   * function does not handle the case of a partial overlap between the source and the destination
   * matrices.
   *
   *   When the operation mask is specified, if the [Mat::create] call shown above reallocates the
   * matrix, the newly allocated matrix is initialized with all zeros before copying the data.
   *
   * @param m Destination matrix. If it does not have a proper size or type before the operation, it is
   * reallocated.
   */
  public copyTo(m: OutputArray): OutputArray;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param m Destination matrix. If it does not have a proper size or type before the operation, it is
   * reallocated.
   *
   * @param mask Operation mask of the same size as *this. Its non-zero elements indicate which matrix
   * elements need to be copied. The mask has to be of type CV_8U and can have 1 or multiple channels.
   */
  public copyTo(m: OutputArray, mask: InputArray): OutputArray;

  /**
   *   This is one of the key [Mat] methods. Most new-style OpenCV functions and methods that produce
   * arrays call this method for each output array. The method uses the following algorithm:
   *
   * If the current array shape and the type match the new ones, return immediately. Otherwise,
   * de-reference the previous data by calling [Mat::release].
   * Initialize the new header.
   * Allocate the new data of [total()]*elemSize() bytes.
   * Allocate the new, associated with the data, reference counter and set it to 1.
   *
   *   Such a scheme makes the memory management robust and efficient at the same time and helps avoid
   * extra typing for you. This means that usually there is no need to explicitly allocate output arrays.
   * That is, instead of writing:
   *
   *   ```cpp
   *   Mat color;
   *   ...
   *   Mat gray(color.rows, color.cols, color.depth());
   *   cvtColor(color, gray, COLOR_BGR2GRAY);
   *   ```
   *
   *    you can simply write:
   *
   *   ```cpp
   *   Mat color;
   *   ...
   *   Mat gray;
   *   cvtColor(color, gray, COLOR_BGR2GRAY);
   *   ```
   *
   *    because cvtColor, as well as the most of OpenCV functions, calls [Mat::create()] for the output
   * array internally.
   *
   * @param rows New number of rows.
   *
   * @param cols New number of columns.
   *
   * @param type New matrix type.
   */
  public create(rows: int, cols: int, type: int): void;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param size Alternative new matrix size specification: Size(cols, rows)
   *
   * @param type New matrix type.
   */
  public create(size: Size, type: int): Size;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param ndims New array dimensionality.
   *
   * @param sizes Array of integers specifying a new array shape.
   *
   * @param type New matrix type.
   */
  public create(ndims: int, sizes: any, type: int): void;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param sizes Array of integers specifying a new array shape.
   *
   * @param type New matrix type.
   */
  public create(sizes: any, type: int): void;

  /**
   *   The method computes a cross-product of two 3-element vectors. The vectors must be 3-element
   * floating-point vectors of the same shape and size. The result is another 3-element vector of the
   * same shape and type as operands.
   *
   * @param m Another cross-product operand.
   */
  public cross(m: InputArray): Mat;

  public deallocate(): void;

  /**
   *   The method returns the identifier of the matrix element depth (the type of each individual
   * channel). For example, for a 16-bit signed element array, the method returns CV_16S . A complete
   * list of matrix types contains the following values:
   *
   * CV_8U - 8-bit unsigned integers ( 0..255 )
   * CV_8S - 8-bit signed integers ( -128..127 )
   * CV_16U - 16-bit unsigned integers ( 0..65535 )
   * CV_16S - 16-bit signed integers ( -32768..32767 )
   * CV_32S - 32-bit signed integers ( -2147483648..2147483647 )
   * CV_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
   * CV_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )
   */
  public depth(): int;

  /**
   *   The method makes a new header for the specified matrix diagonal. The new matrix is represented as
   * a single-column matrix. Similarly to [Mat::row] and [Mat::col], this is an O(1) operation.
   *
   * @param d index of the diagonal, with the following values:
   *   d=0 is the main diagonal.d<0 is a diagonal from the lower half. For example, d=-1 means the
   * diagonal is set immediately below the main one.d>0 is a diagonal from the upper half. For example,
   * d=1 means the diagonal is set immediately above the main one. For example: Matm=(Mat_<int>(3,3)<<
   *   1,2,3,
   *   4,5,6,
   *   7,8,9);
   *   Matd0=m.diag(0);
   *   Matd1=m.diag(1);
   *   Matd_1=m.diag(-1);
   *    The resulting matrices are d0=
   *   [1;
   *   5;
   *   9]
   *   d1=
   *   [2;
   *   6]
   *   d_1=
   *   [4;
   *   8]
   */
  public diag(d?: int): Mat;

  /**
   *   The method computes a dot-product of two matrices. If the matrices are not single-column or
   * single-row vectors, the top-to-bottom left-to-right scan ordering is used to treat them as 1D
   * vectors. The vectors must have the same size and type. If the matrices have more than one channel,
   * the dot products from all the channels are summed together.
   *
   * @param m another dot-product operand.
   */
  public dot(m: InputArray): InputArray;

  /**
   *   The method returns the matrix element size in bytes. For example, if the matrix type is CV_16SC3 ,
   * the method returns 3*sizeof(short) or 6.
   */
  public elemSize(): size_t;

  /**
   *   The method returns the matrix element channel size in bytes, that is, it ignores the number of
   * channels. For example, if the matrix type is CV_16SC3 , the method returns sizeof(short) or 2.
   */
  public elemSize1(): size_t;

  /**
   *   The method returns true if [Mat::total()] is 0 or if [Mat::data] is NULL. Because of [pop_back()]
   * and [resize()] methods `[M.total()] == 0` does not imply that `M.data == NULL`.
   */
  public empty(): bool;

  /**
   *   The methods return the matrix read-only or read-write iterators, set to the point following the
   * last matrix element.
   */
  public end(arg27: any): MatIterator_;

  public end(arg28: any): MatConstIterator_;

  /**
   *   The operation passed as argument has to be a function pointer, a function object or a
   * lambda(C++11).
   *
   *   Example 1. All of the operations below put 0xFF the first channel of all matrix elements:
   *
   *   ```cpp
   *   Mat image(1920, 1080, CV_8UC3);
   *   typedef cv::Point3_<uint8_t> Pixel;
   *
   *   // first. raw pointer access.
   *   for (int r = 0; r < image.rows; ++r) {
   *       Pixel* ptr = image.ptr<Pixel>(r, 0);
   *       const Pixel* ptr_end = ptr + image.cols;
   *       for (; ptr != ptr_end; ++ptr) {
   *           ptr->x = 255;
   *       }
   *   }
   *
   *   // Using MatIterator. (Simple but there are a Iterator's overhead)
   *   for (Pixel &p : cv::Mat_<Pixel>(image)) {
   *       p.x = 255;
   *   }
   *
   *   // Parallel execution with function object.
   *   struct Operator {
   *       void operator ()(Pixel &pixel, const int * position) {
   *           pixel.x = 255;
   *       }
   *   };
   *   image.forEach<Pixel>(Operator());
   *
   *   // Parallel execution using C++11 lambda.
   *   image.forEach<Pixel>([](Pixel &p, const int * position) -> void {
   *       p.x = 255;
   *   });
   *   ```
   *
   *    Example 2. Using the pixel's position:
   *
   *   ```cpp
   *   // Creating 3D matrix (255 x 255 x 255) typed uint8_t
   *   // and initialize all elements by the value which equals elements position.
   *   // i.e. pixels (x,y,z) = (1,2,3) is (b,g,r) = (1,2,3).
   *
   *   int sizes[] = { 255, 255, 255 };
   *   typedef cv::Point3_<uint8_t> Pixel;
   *
   *   Mat_<Pixel> image = Mat::zeros(3, sizes, CV_8UC3);
   *
   *   image.forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
   *       pixel.x = position[0];
   *       pixel.y = position[1];
   *       pixel.z = position[2];
   *   });
   *   ```
   */
  public forEach(arg29: any, arg30: any, operation: any): any;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public forEach(arg31: any, arg32: any, operation: any): any;

  public getUMat(accessFlags: AccessFlag, usageFlags?: UMatUsageFlags): UMat;

  /**
   *   The method performs a matrix inversion by means of matrix expressions. This means that a temporary
   * matrix inversion object is returned by the method and can be used further as a part of more complex
   * matrix expressions or can be assigned to a matrix.
   *
   * @param method Matrix inversion method. One of cv::DecompTypes
   */
  public inv(method?: int): MatExpr;

  /**
   *   The method returns true if the matrix elements are stored continuously without gaps at the end of
   * each row. Otherwise, it returns false. Obviously, 1x1 or 1xN matrices are always continuous.
   * Matrices created with [Mat::create] are always continuous. But if you extract a part of the matrix
   * using [Mat::col], [Mat::diag], and so on, or constructed a matrix header for externally allocated
   * data, such matrices may no longer have this property.
   *
   *   The continuity flag is stored as a bit in the [Mat::flags] field and is computed automatically
   * when you construct a matrix header. Thus, the continuity check is a very fast operation, though
   * theoretically it could be done as follows:
   *
   *   ```cpp
   *   // alternative implementation of Mat::isContinuous()
   *   bool myCheckMatContinuity(const Mat& m)
   *   {
   *       //return (m.flags & Mat::CONTINUOUS_FLAG) != 0;
   *       return m.rows == 1 || m.step == m.cols*m.elemSize();
   *   }
   *   ```
   *
   *    The method is used in quite a few of OpenCV functions. The point is that element-wise operations
   * (such as arithmetic and logical operations, math functions, alpha blending, color space
   * transformations, and others) do not depend on the image geometry. Thus, if all the input and output
   * arrays are continuous, the functions can process them as very long single-row vectors. The example
   * below illustrates how an alpha-blending function can be implemented:
   *
   *   ```cpp
   *   template<typename T>
   *   void alphaBlendRGBA(const Mat& src1, const Mat& src2, Mat& dst)
   *   {
   *       const float alpha_scale = (float)std::numeric_limits<T>::max(),
   *                   inv_scale = 1.f/alpha_scale;
   *
   *       CV_Assert( src1.type() == src2.type() &&
   *                  src1.type() == CV_MAKETYPE(traits::Depth<T>::value, 4) &&
   *                  src1.size() == src2.size());
   *       Size size = src1.size();
   *       dst.create(size, src1.type());
   *
   *       // here is the idiom: check the arrays for continuity and,
   *       // if this is the case,
   *       // treat the arrays as 1D vectors
   *       if( src1.isContinuous() && src2.isContinuous() && dst.isContinuous() )
   *       {
   *           size.width *= size.height;
   *           size.height = 1;
   *       }
   *       size.width *= 4;
   *
   *       for( int i = 0; i < size.height; i++ )
   *       {
   *           // when the arrays are continuous,
   *           // the outer loop is executed only once
   *           const T* ptr1 = src1.ptr<T>(i);
   *           const T* ptr2 = src2.ptr<T>(i);
   *           T* dptr = dst.ptr<T>(i);
   *
   *           for( int j = 0; j < size.width; j += 4 )
   *           {
   *               float alpha = ptr1[j+3]*inv_scale, beta = ptr2[j+3]*inv_scale;
   *               dptr[j] = saturate_cast<T>(ptr1[j]*alpha + ptr2[j]*beta);
   *               dptr[j+1] = saturate_cast<T>(ptr1[j+1]*alpha + ptr2[j+1]*beta);
   *               dptr[j+2] = saturate_cast<T>(ptr1[j+2]*alpha + ptr2[j+2]*beta);
   *               dptr[j+3] = saturate_cast<T>((1 - (1-alpha)*(1-beta))*alpha_scale);
   *           }
   *       }
   *   }
   *   ```
   *
   *    This approach, while being very simple, can boost the performance of a simple element-operation
   * by 10-20 percents, especially if the image is rather small and the operation is quite simple.
   *
   *   Another OpenCV idiom in this function, a call of [Mat::create] for the destination array, that
   * allocates the destination array unless it already has the proper size and type. And while the newly
   * allocated arrays are always continuous, you still need to check the destination array because
   * [Mat::create] does not always allocate a new matrix.
   */
  public isContinuous(): bool;
  /**
   *   After you extracted a submatrix from a matrix using [Mat::row], [Mat::col], [Mat::rowRange],
   * [Mat::colRange], and others, the resultant submatrix points just to the part of the original big
   * matrix. However, each submatrix contains information (represented by datastart and dataend fields)
   * that helps reconstruct the original matrix size and the position of the extracted submatrix within
   * the original matrix. The method locateROI does exactly that.
   *
   * @param wholeSize Output parameter that contains the size of the whole matrix containing this as a
   * part.
   *
   * @param ofs Output parameter that contains an offset of this inside the whole matrix.
   */
  public locateROI(wholeSize: Size, ofs: Point): Size;

  /**
   *   The method returns a temporary object encoding per-element array multiplication, with optional
   * scale. Note that this is not a matrix multiplication that corresponds to a simpler "\\*" operator.
   *
   *   Example:
   *
   *   ```cpp
   *   Mat C = A.mul(5/B); // equivalent to divide(A, B, C, 5)
   *   ```
   *
   * @param m Another array of the same type and the same size as *this, or a matrix expression.
   *
   * @param scale Optional scale factor.
   */
  public mul(m: InputArray, scale?: double): MatExpr;

  /**
   *   The method removes one or more rows from the bottom of the matrix.
   *
   * @param nelems Number of removed rows. If it is greater than the total number of rows, an exception
   * is thrown.
   */
  public pop_back(nelems?: size_t): void;

  /**
   *   The methods return `uchar*` or typed pointer to the specified matrix row. See the sample in
   * [Mat::isContinuous] to know how to use these methods.
   *
   * @param i0 A 0-based row index.
   */
  public ptr(i0?: int): uchar;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(i0?: int): uchar;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param row Index along the dimension 0
   *
   * @param col Index along the dimension 1
   */
  public ptr(row: int, col: int): uchar;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param row Index along the dimension 0
   *
   * @param col Index along the dimension 1
   */
  public ptr(row: int, col: int): uchar;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(i0: int, i1: int, i2: int): uchar;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(i0: int, i1: int, i2: int): uchar;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(idx: any): uchar;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(idx: any): uchar;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(n: int, idx: Vec): uchar;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(n: int, idx: Vec): uchar;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(arg37: any, i0?: int): any;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(arg38: any, i0?: int): any;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param row Index along the dimension 0
   *
   * @param col Index along the dimension 1
   */
  public ptr(arg39: any, row: int, col: int): any;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param row Index along the dimension 0
   *
   * @param col Index along the dimension 1
   */
  public ptr(arg40: any, row: int, col: int): any;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(arg41: any, i0: int, i1: int, i2: int): any;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(arg42: any, i0: int, i1: int, i2: int): any;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(arg43: any, idx: any): any;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(arg44: any, idx: any): any;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(arg45: any, n: int, idx: Vec): Vec;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public ptr(arg46: any, n: int, idx: Vec): Vec;

  /**
   *   The methods add one or more elements to the bottom of the matrix. They emulate the corresponding
   * method of the STL vector class. When elem is [Mat] , its type and the number of columns must be the
   * same as in the container matrix.
   *
   * @param elem Added element(s).
   */
  public push_back(arg47: any, elem: any): any;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param elem Added element(s).
   */
  public push_back(arg48: any, elem: Mat): Mat;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param elem Added element(s).
   */
  public push_back(arg49: any, elem: any): any;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param m Added line(s).
   */
  public push_back(m: Mat): Mat;

  public push_back_(elem: any): void;

  /**
   *   The method decrements the reference counter associated with the matrix data. When the reference
   * counter reaches 0, the matrix data is deallocated and the data and the reference counter pointers
   * are set to NULL's. If the matrix header points to an external data set (see [Mat::Mat] ), the
   * reference counter is NULL, and the method has no effect in this case.
   *
   *   This method can be called manually to force the matrix data deallocation. But since this method is
   * automatically called in the destructor, or by any other method that changes the data pointer, it is
   * usually not needed. The reference counter decrement and check for 0 is an atomic operation on the
   * platforms that support it. Thus, it is safe to operate on the same matrices asynchronously in
   * different threads.
   */
  public release(): void;

  /**
   *   The method reserves space for sz rows. If the matrix already has enough space to store sz rows,
   * nothing happens. If the matrix is reallocated, the first [Mat::rows] rows are preserved. The method
   * emulates the corresponding method of the STL vector class.
   *
   * @param sz Number of rows.
   */
  public reserve(sz: size_t): void;

  /**
   *   The method reserves space for sz bytes. If the matrix already has enough space to store sz bytes,
   * nothing happens. If matrix has to be reallocated its previous content could be lost.
   *
   * @param sz Number of bytes.
   */
  public reserveBuffer(sz: size_t): void;

  /**
   *   The method makes a new matrix header for *this elements. The new matrix may have a different size
   * and/or different number of channels. Any combination is possible if:
   *
   * No extra elements are included into the new matrix and no elements are excluded. Consequently, the
   * product rows*cols*channels() must stay the same after the transformation.
   * No data is copied. That is, this is an O(1) operation. Consequently, if you change the number of
   * rows, or the operation changes the indices of elements row in some other way, the matrix must be
   * continuous. See [Mat::isContinuous] .
   *
   *   For example, if there is a set of 3D points stored as an STL vector, and you want to represent the
   * points as a 3xN matrix, do the following:
   *
   *   ```cpp
   *   std::vector<Point3f> vec;
   *   ...
   *   Mat pointMat = Mat(vec). // convert vector to Mat, O(1) operation
   *                     reshape(1). // make Nx3 1-channel matrix out of Nx1 3-channel.
   *                                 // Also, an O(1) operation
   *                        t(); // finally, transpose the Nx3 matrix.
   *                             // This involves copying all the elements
   *   ```
   *
   * @param cn New number of channels. If the parameter is 0, the number of channels remains the same.
   *
   * @param rows New number of rows. If the parameter is 0, the number of rows remains the same.
   */
  public reshape(cn: int, rows?: int): Mat;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public reshape(cn: int, newndims: int, newsz: any): Mat;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   */
  public reshape(cn: int, newshape: any): Mat;

  /**
   *   The methods change the number of matrix rows. If the matrix is reallocated, the first
   * min(Mat::rows, sz) rows are preserved. The methods emulate the corresponding methods of the STL
   * vector class.
   *
   * @param sz New number of rows.
   */
  public resize(sz: size_t): void;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param sz New number of rows.
   *
   * @param s Value assigned to the newly added elements.
   */
  public resize(sz: size_t, s: Scalar): Scalar;

  /**
   *   The method makes a new header for the specified matrix row and returns it. This is an O(1)
   * operation, regardless of the matrix size. The underlying data of the new matrix is shared with the
   * original matrix. Here is the example of one of the classical basic matrix processing operations,
   * axpy, used by LU and many other algorithms:
   *
   *   ```cpp
   *   inline void matrix_axpy(Mat& A, int i, int j, double alpha)
   *   {
   *       A.row(i) += A.row(j)*alpha;
   *   }
   *   ```
   *
   *   In the current implementation, the following code does not work as expected:
   *
   *   ```cpp
   *   Mat A;
   *   ...
   *   A.row(i) = A.row(j); // will not work
   *   ```
   *
   *    This happens because A.row(i) forms a temporary header that is further assigned to another
   * header. Remember that each of these operations is O(1), that is, no data is copied. Thus, the above
   * assignment is not true if you may have expected the j-th row to be copied to the i-th row. To
   * achieve that, you should either turn this simple assignment into an expression or use the
   * [Mat::copyTo] method:
   *
   *   ```cpp
   *   Mat A;
   *   ...
   *   // works, but looks a bit obscure.
   *   A.row(i) = A.row(j) + 0;
   *   // this is a bit longer, but the recommended method.
   *   A.row(j).copyTo(A.row(i));
   *   ```
   *
   * @param y A 0-based row index.
   */
  public row(y: int): Mat;

  /**
   *   The method makes a new header for the specified row span of the matrix. Similarly to [Mat::row]
   * and [Mat::col] , this is an O(1) operation.
   *
   * @param startrow An inclusive 0-based start index of the row span.
   *
   * @param endrow An exclusive 0-based ending index of the row span.
   */
  public rowRange(startrow: int, endrow: int): Mat;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param r Range structure containing both the start and the end indices.
   */
  public rowRange(r: Range): Mat;

  /**
   *   This is an advanced variant of the [Mat::operator=(const Scalar& s)] operator.
   *
   * @param value Assigned scalar converted to the actual array type.
   *
   * @param mask Operation mask of the same size as *this. Its non-zero elements indicate which matrix
   * elements need to be copied. The mask has to be of type CV_8U and can have 1 or multiple channels
   */
  public setTo(value: InputArray | Scalar, mask?: InputArray): Mat;

  /**
   *   The method returns a matrix step divided by [Mat::elemSize1()] . It can be useful to quickly
   * access an arbitrary matrix element.
   */
  public step1(i?: int): size_t;

  /**
   *   The method performs matrix transposition by means of matrix expressions. It does not perform the
   * actual transposition but returns a temporary matrix transposition object that can be further used as
   * a part of more complex matrix expressions or can be assigned to a matrix:
   *
   *   ```cpp
   *   Mat A1 = A + Mat::eye(A.size(), A.type())*lambda;
   *   Mat C = A1.t()*A1; // compute (A + lambda*I)^t * (A + lamda*I)
   *   ```
   */
  public t(): MatExpr;

  /**
   *   The method returns the number of array elements (a number of pixels if the array represents an
   * image).
   */
  public total(): size_t;

  /**
   *   The method returns the number of elements within a certain sub-array slice with startDim <= dim <
   * endDim
   */
  public total(startDim: int, endDim?: int): size_t;

  /**
   *   The method returns a matrix element type. This is an identifier compatible with the CvMat type
   * system, like CV_16SC3 or 16-bit signed 3-channel array, and so on.
   */
  public type(): int;

  public updateContinuityFlag(): void;

  public ucharPtr(i: any, j?: any): any;
  public charPtr(i: any, j: any): any;
  public shortPtr(i: any, j: any): any;
  public ushortPtr(i: any, j: any): any;
  public intPtr(i: any, j: any): any;
  public floatPtr(i: any, j: any): any;
  public doublePtr(i: any, j: any): any;
  public intPtr(i: any, j: any): any;

  public charAt(i0: number): number;
  public charAt(i0: number, i1: number): number;
  public charAt(i0: number, i1: number, i2: number): number;

  public ucharAt(i0: number): number;
  public ucharAt(i0: number, i1: number): number;
  public ucharAt(i0: number, i1: number, i2: number): number;

  public shortAt(i0: number): number;
  public shortAt(i0: number, i1: number): number;
  public shortAt(i0: number, i1: number, i2: number): number;

  public ushortAt(i0: number): number;
  public ushortAt(i0: number, i1: number): number;
  public ushortAt(i0: number, i1: number, i2: number): number;

  public intAt(i0: number): number;
  public intAt(i0: number, i1: number): number;
  public intAt(i0: number, i1: number, i2: number): number;

  public floatAt(i0: number): number;
  public floatAt(i0: number, i1: number): number;
  public floatAt(i0: number, i1: number, i2: number): number;

  public doubleAt(i0: number): number;
  public doubleAt(i0: number, i1: number): number;
  public doubleAt(i0: number, i1: number, i2: number): number;

  public setTo(value: Mat | Scalar, mask?: Mat): Mat;
  /**
   * Sometimes, you will have to play with certain region of images.
   * For eye detection in images, first face detection is done all
   * over the image and when face is obtained, we select the face region alone and search for eyes inside it instead of searching whole image.
   * It improves accuracy (because eyes are always on faces) and performance (because we search for a small area).
   *
   * Heads up : in JS seems only one argument is expected.
   */
  public roi(expr: Rect | Mat): Mat;

  /**
   *   The method creates a square diagonal matrix from specified main diagonal.
   *
   * @param d One-dimensional matrix that represents the main diagonal.
   */
  public static diag(d: Mat): Mat;

  /**
   *   The method returns a Matlab-style identity matrix initializer, similarly to [Mat::zeros].
   * Similarly to [Mat::ones], you can use a scale operation to create a scaled identity matrix
   * efficiently:
   *
   *   ```cpp
   *   // make a 4x4 diagonal matrix with 0.1's on the diagonal.
   *   Mat A = Mat::eye(4, 4, CV_32F)*0.1;
   *   ```
   *
   *   In case of multi-channels type, identity matrix will be initialized only for the first channel,
   * the others will be set to 0's
   *
   * @param rows Number of rows.
   *
   * @param cols Number of columns.
   *
   * @param type Created matrix type.
   */
  public static eye(rows: int, cols: int, type: int): MatExpr;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param size Alternative matrix size specification as Size(cols, rows) .
   *
   * @param type Created matrix type.
   */
  public static eye(size: Size, type: int): MatExpr;

  public static getDefaultAllocator(): MatAllocator;

  public static getStdAllocator(): MatAllocator;

  /**
   *   The method returns a Matlab-style 1's array initializer, similarly to [Mat::zeros]. Note that
   * using this method you can initialize an array with an arbitrary value, using the following Matlab
   * idiom:
   *
   *   ```cpp
   *   Mat A = Mat::ones(100, 100, CV_8U)*3; // make 100x100 matrix filled with 3.
   *   ```
   *
   *    The above operation does not form a 100x100 matrix of 1's and then multiply it by 3. Instead, it
   * just remembers the scale factor (3 in this case) and use it when actually invoking the matrix
   * initializer.
   *
   *   In case of multi-channels type, only the first channel will be initialized with 1's, the others
   * will be set to 0's.
   *
   * @param rows Number of rows.
   *
   * @param cols Number of columns.
   *
   * @param type Created matrix type.
   */
  public static ones(rows: int, cols: int, type: int): MatExpr;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param size Alternative to the matrix size specification Size(cols, rows) .
   *
   * @param type Created matrix type.
   */
  public static ones(size: Size, type: int): MatExpr;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param ndims Array dimensionality.
   *
   * @param sz Array of integers specifying the array shape.
   *
   * @param type Created matrix type.
   */
  public static ones(ndims: int, sz: any, type: int): MatExpr;

  public static setDefaultAllocator(allocator: MatAllocator): MatAllocator;

  /**
   *   The method returns a Matlab-style zero array initializer. It can be used to quickly form a
   * constant array as a function parameter, part of a matrix expression, or as a matrix initializer:
   *
   *   ```cpp
   *   Mat A;
   *   A = Mat::zeros(3, 3, CV_32F);
   *   ```
   *
   *    In the example above, a new matrix is allocated only if A is not a 3x3 floating-point matrix.
   * Otherwise, the existing matrix A is filled with zeros.
   *
   * @param rows Number of rows.
   *
   * @param cols Number of columns.
   *
   * @param type Created matrix type.
   */
  public static zeros(rows: int, cols: int, type: int): MatExpr;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param size Alternative to the matrix size specification Size(cols, rows) .
   *
   * @param type Created matrix type.
   */
  public static zeros(size: Size, type: int): MatExpr;

  /**
   *   This is an overloaded member function, provided for convenience. It differs from the above
   * function only in what argument(s) it accepts.
   *
   * @param ndims Array dimensionality.
   *
   * @param sz Array of integers specifying the array shape.
   *
   * @param type Created matrix type.
   */
  public static zeros(ndims: int, sz: any, type: int): MatExpr;
}

export declare const MAGIC_VAL: any; // initializer: = 0x42FF0000

export declare const AUTO_STEP: any; // initializer: = 0

export declare const CONTINUOUS_FLAG: any; // initializer: = CV_MAT_CONT_FLAG

export declare const SUBMATRIX_FLAG: any; // initializer: = CV_SUBMAT_FLAG

export declare const MAGIC_MASK: any; // initializer: = 0xFFFF0000

export declare const TYPE_MASK: any; // initializer: = 0x00000FFF

export declare const DEPTH_MASK: any; // initializer: = 7
