import type {
  bool,
  double,
  InputArray,
  int,
  Mat,
  OutputArray,
  OutputArrayOfArrays,
  Point,
  Scalar,
  Size,
  TermCriteria,
} from "./_types";
/*
 * # Image Filtering
 * Functions and classes described in this section are used to perform various linear or non-linear filtering operations on 2D images (represented as [Mat]'s). It means that for each pixel location `$(x,y)$` in the source image (normally, rectangular), its neighborhood is considered and used to compute the response. In case of a linear filter, it is a weighted sum of pixel values. In case of morphological operations, it is the minimum or maximum values, and so on. The computed response is stored in the destination image at the same location `$(x,y)$`. It means that the output image will be of the same size as the input image. Normally, the functions support multi-channel arrays, in which case every channel is processed independently. Therefore, the output image will also have the same number of channels as the input one.
 *
 * Another common feature of the functions and classes described in this section is that, unlike simple arithmetic functions, they need to extrapolate values of some non-existing pixels. For example, if you want to smooth an image using a Gaussian `$3 \times 3$` filter, then, when processing the left-most pixels in each row, you need pixels to the left of them, that is, outside of the image. You can let these pixels be the same as the left-most image pixels ("replicated
 * border" extrapolation method), or assume that all the non-existing pixels are zeros ("constant
 * border" extrapolation method), and so on. OpenCV enables you to specify the extrapolation method. For details, see [BorderTypes]
 *
 * <a name="d4/d86/group__imgproc__filter_1filter_depths"></a>
 *
 * ## Depth combinations
 *
 *
 *
 *
 *
 * when ddepth=-1, the output image will have the same depth as the source.
 */
/**
 * The function applies bilateral filtering to the input image, as described in  bilateralFilter can
 * reduce unwanted noise very well while keeping edges fairly sharp. However, it is very slow compared
 * to most filters.
 *
 * Sigma values*: For simplicity, you can set the 2 sigma values to be the same. If they are small (<
 * 10), the filter will not have much effect, whereas if they are large (> 150), they will have a very
 * strong effect, making the image look "cartoonish".
 *
 * Filter size*: Large filters (d > 5) are very slow, so it is recommended to use d=5 for real-time
 * applications, and perhaps d=9 for offline applications that need heavy noise filtering.
 *
 * This filter does not work inplace.
 *
 * @param src Source 8-bit or floating-point, 1-channel or 3-channel image.
 *
 * @param dst Destination image of the same size and type as src .
 *
 * @param d Diameter of each pixel neighborhood that is used during filtering. If it is non-positive,
 * it is computed from sigmaSpace.
 *
 * @param sigmaColor Filter sigma in the color space. A larger value of the parameter means that
 * farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in
 * larger areas of semi-equal color.
 *
 * @param sigmaSpace Filter sigma in the coordinate space. A larger value of the parameter means that
 * farther pixels will influence each other as long as their colors are close enough (see sigmaColor ).
 * When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional
 * to sigmaSpace.
 *
 * @param borderType border mode used to extrapolate pixels outside of the image, see BorderTypes
 */
export declare function bilateralFilter(
  src: InputArray,
  dst: OutputArray,
  d: int,
  sigmaColor: double,
  sigmaSpace: double,
  borderType?: int,
): void;

/**
 * The function smooths an image using the kernel:
 *
 * `\\[\\texttt{K} = \\frac{1}{\\texttt{ksize.width*ksize.height}} \\begin{bmatrix} 1 & 1 & 1 & \\cdots
 * & 1 & 1 \\\\ 1 & 1 & 1 & \\cdots & 1 & 1 \\\\ \\hdotsfor{6} \\\\ 1 & 1 & 1 & \\cdots & 1 & 1 \\\\
 * \\end{bmatrix}\\]`
 *
 * The call `blur(src, dst, ksize, anchor, borderType)` is equivalent to `boxFilter(src, dst,
 * src.type(), anchor, true, borderType)`.
 *
 * [boxFilter], [bilateralFilter], [GaussianBlur], [medianBlur]
 *
 * @param src input image; it can have any number of channels, which are processed independently, but
 * the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
 *
 * @param dst output image of the same size and type as src.
 *
 * @param ksize blurring kernel size.
 *
 * @param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel
 * center.
 *
 * @param borderType border mode used to extrapolate pixels outside of the image, see BorderTypes
 */
export declare function blur(
  src: InputArray,
  dst: OutputArray,
  ksize: Size,
  anchor?: Point,
  borderType?: int,
): void;

/**
 * The function smooths an image using the kernel:
 *
 * `\\[\\texttt{K} = \\alpha \\begin{bmatrix} 1 & 1 & 1 & \\cdots & 1 & 1 \\\\ 1 & 1 & 1 & \\cdots & 1
 * & 1 \\\\ \\hdotsfor{6} \\\\ 1 & 1 & 1 & \\cdots & 1 & 1 \\end{bmatrix}\\]`
 *
 * where
 *
 * `\\[\\alpha = \\fork{\\frac{1}{\\texttt{ksize.width*ksize.height}}}{when
 * \\texttt{normalize=true}}{1}{otherwise}\\]`
 *
 * Unnormalized box filter is useful for computing various integral characteristics over each pixel
 * neighborhood, such as covariance matrices of image derivatives (used in dense optical flow
 * algorithms, and so on). If you need to compute pixel sums over variable-size windows, use
 * [integral].
 *
 * [blur], [bilateralFilter], [GaussianBlur], [medianBlur], [integral]
 *
 * @param src input image.
 *
 * @param dst output image of the same size and type as src.
 *
 * @param ddepth the output image depth (-1 to use src.depth()).
 *
 * @param ksize blurring kernel size.
 *
 * @param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel
 * center.
 *
 * @param normalize flag, specifying whether the kernel is normalized by its area or not.
 *
 * @param borderType border mode used to extrapolate pixels outside of the image, see BorderTypes
 */
export declare function boxFilter(
  src: InputArray,
  dst: OutputArray,
  ddepth: int,
  ksize: Size,
  anchor?: Point,
  normalize?: bool,
  borderType?: int,
): void;

/**
 * The function constructs a vector of images and builds the Gaussian pyramid by recursively applying
 * pyrDown to the previously built pyramid layers, starting from `dst[0]==src`.
 *
 * @param src Source image. Check pyrDown for the list of supported types.
 *
 * @param dst Destination vector of maxlevel+1 images of the same type as src. dst[0] will be the same
 * as src. dst[1] is the next pyramid layer, a smoothed and down-sized src, and so on.
 *
 * @param maxlevel 0-based index of the last (the smallest) pyramid layer. It must be non-negative.
 *
 * @param borderType Pixel extrapolation method, see BorderTypes (BORDER_CONSTANT isn't supported)
 */
export declare function buildPyramid(
  src: InputArray,
  dst: OutputArrayOfArrays,
  maxlevel: int,
  borderType?: int,
): void;

/**
 * The function dilates the source image using the specified structuring element that determines the
 * shape of a pixel neighborhood over which the maximum is taken: `\\[\\texttt{dst} (x,y) = \\max
 * _{(x',y'): \\, \\texttt{element} (x',y') \\ne0 } \\texttt{src} (x+x',y+y')\\]`
 *
 * The function supports the in-place mode. Dilation can be applied several ( iterations ) times. In
 * case of multi-channel images, each channel is processed independently.
 *
 * [erode], [morphologyEx], [getStructuringElement]
 *
 * @param src input image; the number of channels can be arbitrary, but the depth should be one of
 * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
 *
 * @param dst output image of the same size and type as src.
 *
 * @param kernel structuring element used for dilation; if elemenat=Mat(), a 3 x 3 rectangular
 * structuring element is used. Kernel can be created using getStructuringElement
 *
 * @param anchor position of the anchor within the element; default value (-1, -1) means that the
 * anchor is at the element center.
 *
 * @param iterations number of times dilation is applied.
 *
 * @param borderType pixel extrapolation method, see BorderTypes
 *
 * @param borderValue border value in case of a constant border
 */
export declare function dilate(
  src: InputArray,
  dst: OutputArray,
  kernel: InputArray,
  anchor?: Point,
  iterations?: int,
  borderType?: int,
  borderValue?: any,
): void;

/**
 * The function erodes the source image using the specified structuring element that determines the
 * shape of a pixel neighborhood over which the minimum is taken:
 *
 * `\\[\\texttt{dst} (x,y) = \\min _{(x',y'): \\, \\texttt{element} (x',y') \\ne0 } \\texttt{src}
 * (x+x',y+y')\\]`
 *
 * The function supports the in-place mode. Erosion can be applied several ( iterations ) times. In
 * case of multi-channel images, each channel is processed independently.
 *
 * [dilate], [morphologyEx], [getStructuringElement]
 *
 * @param src input image; the number of channels can be arbitrary, but the depth should be one of
 * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
 *
 * @param dst output image of the same size and type as src.
 *
 * @param kernel structuring element used for erosion; if element=Mat(), a 3 x 3 rectangular
 * structuring element is used. Kernel can be created using getStructuringElement.
 *
 * @param anchor position of the anchor within the element; default value (-1, -1) means that the
 * anchor is at the element center.
 *
 * @param iterations number of times erosion is applied.
 *
 * @param borderType pixel extrapolation method, see BorderTypes
 *
 * @param borderValue border value in case of a constant border
 */
export declare function erode(
  src: InputArray,
  dst: OutputArray,
  kernel: InputArray,
  anchor?: Point,
  iterations?: int,
  borderType?: int,
  borderValue?: any,
): void;

/**
 * The function applies an arbitrary linear filter to an image. In-place operation is supported. When
 * the aperture is partially outside the image, the function interpolates outlier pixel values
 * according to the specified border mode.
 *
 * The function does actually compute correlation, not the convolution:
 *
 * `\\[\\texttt{dst} (x,y) = \\sum _{ \\stackrel{0\\leq x' < \\texttt{kernel.cols},}{0\\leq y' <
 * \\texttt{kernel.rows}} } \\texttt{kernel} (x',y')* \\texttt{src} (x+x'- \\texttt{anchor.x} ,y+y'-
 * \\texttt{anchor.y} )\\]`
 *
 * That is, the kernel is not mirrored around the anchor point. If you need a real convolution, flip
 * the kernel using [flip] and set the new anchor to `(kernel.cols - anchor.x - 1, kernel.rows -
 * anchor.y - 1)`.
 *
 * The function uses the DFT-based algorithm in case of sufficiently large kernels (~`11 x 11` or
 * larger) and the direct algorithm for small kernels.
 *
 * [sepFilter2D], [dft], [matchTemplate]
 *
 * @param src input image.
 *
 * @param dst output image of the same size and the same number of channels as src.
 *
 * @param ddepth desired depth of the destination image, see combinations
 *
 * @param kernel convolution kernel (or rather a correlation kernel), a single-channel floating point
 * matrix; if you want to apply different kernels to different channels, split the image into separate
 * color planes using split and process them individually.
 *
 * @param anchor anchor of the kernel that indicates the relative position of a filtered point within
 * the kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor is
 * at the kernel center.
 *
 * @param delta optional value added to the filtered pixels before storing them in dst.
 *
 * @param borderType pixel extrapolation method, see BorderTypes
 */
export declare function filter2D(
  src: InputArray,
  dst: OutputArray,
  ddepth: int,
  kernel: InputArray,
  anchor?: Point,
  delta?: double,
  borderType?: int,
): void;

/**
 * The function convolves the source image with the specified Gaussian kernel. In-place filtering is
 * supported.
 *
 * [sepFilter2D], [filter2D], [blur], [boxFilter], [bilateralFilter], [medianBlur]
 *
 * @param src input image; the image can have any number of channels, which are processed
 * independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
 *
 * @param dst output image of the same size and type as src.
 *
 * @param ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both must be
 * positive and odd. Or, they can be zero's and then they are computed from sigma.
 *
 * @param sigmaX Gaussian kernel standard deviation in X direction.
 *
 * @param sigmaY Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be
 * equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height,
 * respectively (see getGaussianKernel for details); to fully control the result regardless of possible
 * future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and
 * sigmaY.
 *
 * @param borderType pixel extrapolation method, see BorderTypes
 */
export declare function GaussianBlur(
  src: InputArray,
  dst: OutputArray,
  ksize: Size,
  sigmaX: double,
  sigmaY?: double,
  borderType?: int,
): void;

/**
 * The function computes and returns the filter coefficients for spatial image derivatives. When
 * `ksize=FILTER_SCHARR`, the Scharr `$3 \\times 3$` kernels are generated (see [Scharr]). Otherwise,
 * Sobel kernels are generated (see [Sobel]). The filters are normally passed to [sepFilter2D] or to
 *
 * @param kx Output matrix of row filter coefficients. It has the type ktype .
 *
 * @param ky Output matrix of column filter coefficients. It has the type ktype .
 *
 * @param dx Derivative order in respect of x.
 *
 * @param dy Derivative order in respect of y.
 *
 * @param ksize Aperture size. It can be FILTER_SCHARR, 1, 3, 5, or 7.
 *
 * @param normalize Flag indicating whether to normalize (scale down) the filter coefficients or not.
 * Theoretically, the coefficients should have the denominator $=2^{ksize*2-dx-dy-2}$. If you are going
 * to filter floating-point images, you are likely to use the normalized kernels. But if you compute
 * derivatives of an 8-bit image, store the results in a 16-bit image, and wish to preserve all the
 * fractional bits, you may want to set normalize=false .
 *
 * @param ktype Type of filter coefficients. It can be CV_32f or CV_64F .
 */
export declare function getDerivKernels(
  kx: OutputArray,
  ky: OutputArray,
  dx: int,
  dy: int,
  ksize: int,
  normalize?: bool,
  ktype?: int,
): void;

/**
 * For more details about gabor filter equations and parameters, see: .
 *
 * @param ksize Size of the filter returned.
 *
 * @param sigma Standard deviation of the gaussian envelope.
 *
 * @param theta Orientation of the normal to the parallel stripes of a Gabor function.
 *
 * @param lambd Wavelength of the sinusoidal factor.
 *
 * @param gamma Spatial aspect ratio.
 *
 * @param psi Phase offset.
 *
 * @param ktype Type of filter coefficients. It can be CV_32F or CV_64F .
 */
export declare function getGaborKernel(
  ksize: Size,
  sigma: double,
  theta: double,
  lambd: double,
  gamma: double,
  psi?: double,
  ktype?: int,
): Mat;

/**
 * The function computes and returns the `$\\texttt{ksize} \\times 1$` matrix of Gaussian filter
 * coefficients:
 *
 * `\\[G_i= \\alpha *e^{-(i-( \\texttt{ksize} -1)/2)^2/(2* \\texttt{sigma}^2)},\\]`
 *
 * where `$i=0..\\texttt{ksize}-1$` and `$\\alpha$` is the scale factor chosen so that `$\\sum_i
 * G_i=1$`.
 *
 * Two of such generated kernels can be passed to sepFilter2D. Those functions automatically recognize
 * smoothing kernels (a symmetrical kernel with sum of weights equal to 1) and handle them accordingly.
 * You may also use the higher-level GaussianBlur.
 *
 * [sepFilter2D], [getDerivKernels], [getStructuringElement], [GaussianBlur]
 *
 * @param ksize Aperture size. It should be odd ( $\texttt{ksize} \mod 2 = 1$ ) and positive.
 *
 * @param sigma Gaussian standard deviation. If it is non-positive, it is computed from ksize as sigma
 * = 0.3*((ksize-1)*0.5 - 1) + 0.8.
 *
 * @param ktype Type of filter coefficients. It can be CV_32F or CV_64F .
 */
export declare function getGaussianKernel(
  ksize: int,
  sigma: double,
  ktype?: int,
): Mat;

/**
 * The function constructs and returns the structuring element that can be further passed to [erode],
 * [dilate] or [morphologyEx]. But you can also construct an arbitrary binary mask yourself and use it
 * as the structuring element.
 *
 * @param shape Element shape that could be one of MorphShapes
 *
 * @param ksize Size of the structuring element.
 *
 * @param anchor Anchor position within the element. The default value $(-1, -1)$ means that the anchor
 * is at the center. Note that only the shape of a cross-shaped element depends on the anchor position.
 * In other cases the anchor just regulates how much the result of the morphological operation is
 * shifted.
 */
export declare function getStructuringElement(
  shape: int,
  ksize: Size,
  anchor?: Point,
): Mat;

/**
 * The function calculates the Laplacian of the source image by adding up the second x and y
 * derivatives calculated using the Sobel operator:
 *
 * `\\[\\texttt{dst} = \\Delta \\texttt{src} = \\frac{\\partial^2 \\texttt{src}}{\\partial x^2} +
 * \\frac{\\partial^2 \\texttt{src}}{\\partial y^2}\\]`
 *
 * This is done when `ksize > 1`. When `ksize == 1`, the Laplacian is computed by filtering the image
 * with the following `$3 \\times 3$` aperture:
 *
 * `\\[\\vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}\\]`
 *
 * [Sobel], [Scharr]
 *
 * @param src Source image.
 *
 * @param dst Destination image of the same size and the same number of channels as src .
 *
 * @param ddepth Desired depth of the destination image.
 *
 * @param ksize Aperture size used to compute the second-derivative filters. See getDerivKernels for
 * details. The size must be positive and odd.
 *
 * @param scale Optional scale factor for the computed Laplacian values. By default, no scaling is
 * applied. See getDerivKernels for details.
 *
 * @param delta Optional delta value that is added to the results prior to storing them in dst .
 *
 * @param borderType Pixel extrapolation method, see BorderTypes
 */
export declare function Laplacian(
  src: InputArray,
  dst: OutputArray,
  ddepth: int,
  ksize?: int,
  scale?: double,
  delta?: double,
  borderType?: int,
): void;

/**
 * The function smoothes an image using the median filter with the `$\\texttt{ksize} \\times
 * \\texttt{ksize}$` aperture. Each channel of a multi-channel image is processed independently.
 * In-place operation is supported.
 *
 * The median filter uses [BORDER_REPLICATE] internally to cope with border pixels, see [BorderTypes]
 *
 * [bilateralFilter], [blur], [boxFilter], [GaussianBlur]
 *
 * @param src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U,
 * CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
 *
 * @param dst destination array of the same size and type as src.
 *
 * @param ksize aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...
 */
export declare function medianBlur(
  src: InputArray,
  dst: OutputArray,
  ksize: int,
): void;

export declare function morphologyDefaultBorderValue(): Scalar;

/**
 * The function [cv::morphologyEx] can perform advanced morphological transformations using an erosion
 * and dilation as basic operations.
 *
 * Any of the operations can be done in-place. In case of multi-channel images, each channel is
 * processed independently.
 *
 * [dilate], [erode], [getStructuringElement]
 *
 * The number of iterations is the number of times erosion or dilatation operation will be applied. For
 * instance, an opening operation ([MORPH_OPEN]) with two iterations is equivalent to apply
 * successively: erode -> erode -> dilate -> dilate (and not erode -> dilate -> erode -> dilate).
 *
 * @param src Source image. The number of channels can be arbitrary. The depth should be one of CV_8U,
 * CV_16U, CV_16S, CV_32F or CV_64F.
 *
 * @param dst Destination image of the same size and type as source image.
 *
 * @param op Type of a morphological operation, see MorphTypes
 *
 * @param kernel Structuring element. It can be created using getStructuringElement.
 *
 * @param anchor Anchor position with the kernel. Negative values mean that the anchor is at the kernel
 * center.
 *
 * @param iterations Number of times erosion and dilation are applied.
 *
 * @param borderType Pixel extrapolation method, see BorderTypes
 *
 * @param borderValue Border value in case of a constant border. The default value has a special
 * meaning.
 */
export declare function morphologyEx(
  src: InputArray,
  dst: OutputArray,
  op: int | MorphTypes,
  kernel: InputArray,
  anchor?: Point,
  iterations?: int,
  borderType?: int,
  borderValue?: any,
): void;

/**
 * By default, size of the output image is computed as `Size((src.cols+1)/2, (src.rows+1)/2)`, but in
 * any case, the following conditions should be satisfied:
 *
 * `\\[\\begin{array}{l} | \\texttt{dstsize.width} *2-src.cols| \\leq 2 \\\\ | \\texttt{dstsize.height}
 * *2-src.rows| \\leq 2 \\end{array}\\]`
 *
 * The function performs the downsampling step of the Gaussian pyramid construction. First, it
 * convolves the source image with the kernel:
 *
 * `\\[\\frac{1}{256} \\begin{bmatrix} 1 & 4 & 6 & 4 & 1 \\\\ 4 & 16 & 24 & 16 & 4 \\\\ 6 & 24 & 36 &
 * 24 & 6 \\\\ 4 & 16 & 24 & 16 & 4 \\\\ 1 & 4 & 6 & 4 & 1 \\end{bmatrix}\\]`
 *
 * Then, it downsamples the image by rejecting even rows and columns.
 *
 * @param src input image.
 *
 * @param dst output image; it has the specified size and the same type as src.
 *
 * @param dstsize size of the output image.
 *
 * @param borderType Pixel extrapolation method, see BorderTypes (BORDER_CONSTANT isn't supported)
 */
export declare function pyrDown(
  src: InputArray,
  dst: OutputArray,
  dstsize?: any,
  borderType?: int,
): void;

/**
 * The function implements the filtering stage of meanshift segmentation, that is, the output of the
 * function is the filtered "posterized" image with color gradients and fine-grain texture flattened.
 * At every pixel (X,Y) of the input image (or down-sized input image, see below) the function executes
 * meanshift iterations, that is, the pixel (X,Y) neighborhood in the joint space-color hyperspace is
 * considered:
 *
 * `\\[(x,y): X- \\texttt{sp} \\le x \\le X+ \\texttt{sp} , Y- \\texttt{sp} \\le y \\le Y+ \\texttt{sp}
 * , ||(R,G,B)-(r,g,b)|| \\le \\texttt{sr}\\]`
 *
 * where (R,G,B) and (r,g,b) are the vectors of color components at (X,Y) and (x,y), respectively
 * (though, the algorithm does not depend on the color space used, so any 3-component color space can
 * be used instead). Over the neighborhood the average spatial value (X',Y') and average color vector
 * (R',G',B') are found and they act as the neighborhood center on the next iteration:
 *
 * `\\[(X,Y)~(X',Y'), (R,G,B)~(R',G',B').\\]`
 *
 * After the iterations over, the color components of the initial pixel (that is, the pixel from where
 * the iterations started) are set to the final value (average color at the last iteration):
 *
 * `\\[I(X,Y) <- (R*,G*,B*)\\]`
 *
 * When maxLevel > 0, the gaussian pyramid of maxLevel+1 levels is built, and the above procedure is
 * run on the smallest layer first. After that, the results are propagated to the larger layer and the
 * iterations are run again only on those pixels where the layer colors differ by more than sr from the
 * lower-resolution layer of the pyramid. That makes boundaries of color regions sharper. Note that the
 * results will be actually different from the ones obtained by running the meanshift procedure on the
 * whole original image (i.e. when maxLevel==0).
 *
 * @param src The source 8-bit, 3-channel image.
 *
 * @param dst The destination image of the same format and the same size as the source.
 *
 * @param sp The spatial window radius.
 *
 * @param sr The color window radius.
 *
 * @param maxLevel Maximum level of the pyramid for the segmentation.
 *
 * @param termcrit Termination criteria: when to stop meanshift iterations.
 */
export declare function pyrMeanShiftFiltering(
  src: InputArray,
  dst: OutputArray,
  sp: double,
  sr: double,
  maxLevel?: int,
  termcrit?: TermCriteria,
): void;

/**
 * By default, size of the output image is computed as `Size(src.cols\\*2, (src.rows\\*2)`, but in any
 * case, the following conditions should be satisfied:
 *
 * `\\[\\begin{array}{l} | \\texttt{dstsize.width} -src.cols*2| \\leq ( \\texttt{dstsize.width} \\mod
 * 2) \\\\ | \\texttt{dstsize.height} -src.rows*2| \\leq ( \\texttt{dstsize.height} \\mod 2)
 * \\end{array}\\]`
 *
 * The function performs the upsampling step of the Gaussian pyramid construction, though it can
 * actually be used to construct the Laplacian pyramid. First, it upsamples the source image by
 * injecting even zero rows and columns and then convolves the result with the same kernel as in
 * pyrDown multiplied by 4.
 *
 * @param src input image.
 *
 * @param dst output image. It has the specified size and the same type as src .
 *
 * @param dstsize size of the output image.
 *
 * @param borderType Pixel extrapolation method, see BorderTypes (only BORDER_DEFAULT is supported)
 */
export declare function pyrUp(
  src: InputArray,
  dst: OutputArray,
  dstsize?: any,
  borderType?: int,
): void;

/**
 * The function computes the first x- or y- spatial image derivative using the Scharr operator. The
 * call
 *
 * `\\[\\texttt{Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)}\\]`
 *
 * is equivalent to
 *
 * `\\[\\texttt{Sobel(src, dst, ddepth, dx, dy, FILTER_SCHARR, scale, delta, borderType)} .\\]`
 *
 * [cartToPolar]
 *
 * @param src input image.
 *
 * @param dst output image of the same size and the same number of channels as src.
 *
 * @param ddepth output image depth, see combinations
 *
 * @param dx order of the derivative x.
 *
 * @param dy order of the derivative y.
 *
 * @param scale optional scale factor for the computed derivative values; by default, no scaling is
 * applied (see getDerivKernels for details).
 *
 * @param delta optional delta value that is added to the results prior to storing them in dst.
 *
 * @param borderType pixel extrapolation method, see BorderTypes
 */
export declare function Scharr(
  src: InputArray,
  dst: OutputArray,
  ddepth: int,
  dx: int,
  dy: int,
  scale?: double,
  delta?: double,
  borderType?: int,
): void;

/**
 * The function applies a separable linear filter to the image. That is, first, every row of src is
 * filtered with the 1D kernel kernelX. Then, every column of the result is filtered with the 1D kernel
 * kernelY. The final result shifted by delta is stored in dst .
 *
 * [filter2D], [Sobel], [GaussianBlur], [boxFilter], [blur]
 *
 * @param src Source image.
 *
 * @param dst Destination image of the same size and the same number of channels as src .
 *
 * @param ddepth Destination image depth, see combinations
 *
 * @param kernelX Coefficients for filtering each row.
 *
 * @param kernelY Coefficients for filtering each column.
 *
 * @param anchor Anchor position within the kernel. The default value $(-1,-1)$ means that the anchor
 * is at the kernel center.
 *
 * @param delta Value added to the filtered results before storing them.
 *
 * @param borderType Pixel extrapolation method, see BorderTypes
 */
export declare function sepFilter2D(
  src: InputArray,
  dst: OutputArray,
  ddepth: int,
  kernelX: InputArray,
  kernelY: InputArray,
  anchor?: Point,
  delta?: double,
  borderType?: int,
): void;

/**
 * In all cases except one, the `$\\texttt{ksize} \\times \\texttt{ksize}$` separable kernel is used to
 * calculate the derivative. When `$\\texttt{ksize = 1}$`, the `$3 \\times 1$` or `$1 \\times 3$`
 * kernel is used (that is, no Gaussian smoothing is done). `ksize = 1` can only be used for the first
 * or the second x- or y- derivatives.
 *
 * There is also the special value `ksize = [FILTER_SCHARR] (-1)` that corresponds to the `$3\\times3$`
 * Scharr filter that may give more accurate results than the `$3\\times3$` Sobel. The Scharr aperture
 * is
 *
 * `\\[\\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}\\]`
 *
 * for the x-derivative, or transposed for the y-derivative.
 *
 * The function calculates an image derivative by convolving the image with the appropriate kernel:
 *
 * `\\[\\texttt{dst} = \\frac{\\partial^{xorder+yorder} \\texttt{src}}{\\partial x^{xorder} \\partial
 * y^{yorder}}\\]`
 *
 * The Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less
 * resistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)
 * or ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first
 * case corresponds to a kernel of:
 *
 * `\\[\\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}\\]`
 *
 * The second case corresponds to a kernel of:
 *
 * `\\[\\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}\\]`
 *
 * [Scharr], [Laplacian], [sepFilter2D], [filter2D], [GaussianBlur], [cartToPolar]
 *
 * @param src input image.
 *
 * @param dst output image of the same size and the same number of channels as src .
 *
 * @param ddepth output image depth, see combinations; in the case of 8-bit input images it will result
 * in truncated derivatives.
 *
 * @param dx order of the derivative x.
 *
 * @param dy order of the derivative y.
 *
 * @param ksize size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
 *
 * @param scale optional scale factor for the computed derivative values; by default, no scaling is
 * applied (see getDerivKernels for details).
 *
 * @param delta optional delta value that is added to the results prior to storing them in dst.
 *
 * @param borderType pixel extrapolation method, see BorderTypes
 */
export declare function Sobel(
  src: InputArray,
  dst: OutputArray,
  ddepth: int,
  dx: int,
  dy: int,
  ksize?: int,
  scale?: double,
  delta?: double,
  borderType?: int,
): void;

/**
 * Equivalent to calling:
 *
 * ```cpp
 * Sobel( src, dx, CV_16SC1, 1, 0, 3 );
 * Sobel( src, dy, CV_16SC1, 0, 1, 3 );
 * ```
 *
 * [Sobel]
 *
 * @param src input image.
 *
 * @param dx output image with first-order derivative in x.
 *
 * @param dy output image with first-order derivative in y.
 *
 * @param ksize size of Sobel kernel. It must be 3.
 *
 * @param borderType pixel extrapolation method, see BorderTypes
 */
export declare function spatialGradient(
  src: InputArray,
  dx: OutputArray,
  dy: OutputArray,
  ksize?: int,
  borderType?: int,
): void;

/**
 * For every pixel `$ (x, y) $` in the source image, the function calculates the sum of squares of
 * those neighboring pixel values which overlap the filter placed over the pixel `$ (x, y) $`.
 *
 * The unnormalized square box filter can be useful in computing local image statistics such as the the
 * local variance and standard deviation around the neighborhood of a pixel.
 *
 * [boxFilter]
 *
 * @param src input image
 *
 * @param dst output image of the same size and type as _src
 *
 * @param ddepth the output image depth (-1 to use src.depth())
 *
 * @param ksize kernel size
 *
 * @param anchor kernel anchor point. The default value of Point(-1, -1) denotes that the anchor is at
 * the kernel center.
 *
 * @param normalize flag, specifying whether the kernel is to be normalized by it's area or not.
 *
 * @param borderType border mode used to extrapolate pixels outside of the image, see BorderTypes
 */
export declare function sqrBoxFilter(
  src: InputArray,
  dst: OutputArray,
  ddepth: int,
  ksize: Size,
  anchor?: Point,
  normalize?: bool,
  borderType?: int,
): void;

export declare const MORPH_RECT: MorphShapes; // initializer: = 0

/**
 * a cross-shaped structuring element: `\\[E_{ij} = \\fork{1}{if i=\\texttt{anchor.y} or
 * j=\\texttt{anchor.x}}{0}{otherwise}\\]`
 *
 */
export declare const MORPH_CROSS: MorphShapes; // initializer: = 1

/**
 * an elliptic structuring element, that is, a filled ellipse inscribed into the rectangle Rect(0, 0,
 * esize.width, 0.esize.height)
 *
 */
export declare const MORPH_ELLIPSE: MorphShapes; // initializer: = 2

export declare const MORPH_ERODE: MorphTypes; // initializer: = 0

export declare const MORPH_DILATE: MorphTypes; // initializer: = 1

/**
 * an opening operation `\\[\\texttt{dst} = \\mathrm{open} ( \\texttt{src} , \\texttt{element} )=
 * \\mathrm{dilate} ( \\mathrm{erode} ( \\texttt{src} , \\texttt{element} ))\\]`
 *
 */
export declare const MORPH_OPEN: MorphTypes; // initializer: = 2

/**
 * a closing operation `\\[\\texttt{dst} = \\mathrm{close} ( \\texttt{src} , \\texttt{element} )=
 * \\mathrm{erode} ( \\mathrm{dilate} ( \\texttt{src} , \\texttt{element} ))\\]`
 *
 */
export declare const MORPH_CLOSE: MorphTypes; // initializer: = 3

/**
 * a morphological gradient `\\[\\texttt{dst} = \\mathrm{morph\\_grad} ( \\texttt{src} ,
 * \\texttt{element} )= \\mathrm{dilate} ( \\texttt{src} , \\texttt{element} )- \\mathrm{erode} (
 * \\texttt{src} , \\texttt{element} )\\]`
 *
 */
export declare const MORPH_GRADIENT: MorphTypes; // initializer: = 4

/**
 * "top hat" `\\[\\texttt{dst} = \\mathrm{tophat} ( \\texttt{src} , \\texttt{element} )= \\texttt{src}
 * - \\mathrm{open} ( \\texttt{src} , \\texttt{element} )\\]`
 *
 */
export declare const MORPH_TOPHAT: MorphTypes; // initializer: = 5

/**
 * "black hat" `\\[\\texttt{dst} = \\mathrm{blackhat} ( \\texttt{src} , \\texttt{element} )=
 * \\mathrm{close} ( \\texttt{src} , \\texttt{element} )- \\texttt{src}\\]`
 *
 */
export declare const MORPH_BLACKHAT: MorphTypes; // initializer: = 6

/**
 * "hit or miss" .- Only supported for CV_8UC1 binary images. A tutorial can be found in the
 * documentation
 *
 */
export declare const MORPH_HITMISS: MorphTypes; // initializer: = 7

export declare const FILTER_SCHARR: SpecialFilter; // initializer: = -1

export type MorphShapes = any;

export type MorphTypes = any;

export type SpecialFilter = any;
