import type {
  AsyncArray,
  bool,
  double,
  ErrorCallback,
  float,
  float16_t,
  InputArray,
  InputArrayOfArrays,
  InputOutputArray,
  InputOutputArrayOfArrays,
  int,
  int64,
  schar,
  short,
  size_t,
  uchar,
  uint64,
  unsigned,
  ushort,
  _Tp,
} from "./_types";
/*
 * # Utility and system functions and macros
 *
 */
/**
 * The function returns the aligned pointer of the same type as the input pointer:
 * `\\[\\texttt{(_Tp*)(((size_t)ptr + n-1) & -n)}\\]`
 *
 * @param ptr Aligned pointer.
 *
 * @param n Alignment size that must be a power of two.
 */
export declare function alignPtr(arg92: any, ptr: any, n?: int): any;

/**
 * The function returns the minimum number that is greater than or equal to sz and is divisible by n :
 * `\\[\\texttt{(sz + n-1) & -n}\\]`
 *
 * @param sz Buffer size to align.
 *
 * @param n Alignment size that must be a power of two.
 */
export declare function alignSize(sz: size_t, n: int): size_t;

/**
 * The function returns true if the host hardware supports the specified feature. When user calls
 * setUseOptimized(false), the subsequent calls to [checkHardwareSupport()] will return false until
 * setUseOptimized(true) is called. This way user can dynamically switch on and off the optimized code
 * in OpenCV.
 *
 * @param feature The feature of interest, one of cv::CpuFeatures
 */
export declare function checkHardwareSupport(feature: int): bool;

/**
 * proxy for hal::Cholesky
 */
export declare function Cholesky(
  A: any,
  astep: size_t,
  m: int,
  b: any,
  bstep: size_t,
  n: int,
): bool;

/**
 * proxy for hal::Cholesky
 */
export declare function Cholesky(
  A: any,
  astep: size_t,
  m: int,
  b: any,
  bstep: size_t,
  n: int,
): bool;

/**
 * The function cubeRoot computes `$\\sqrt[3]{\\texttt{val}}$`. Negative arguments are handled
 * correctly. NaN and Inf are not handled. The accuracy approaches the maximum possible accuracy for
 * single-precision data.
 *
 * @param val A function argument.
 */
export declare function cubeRoot(val: float): float;

export declare function cv_abs(arg93: any, x: _Tp): any;

export declare function cv_abs(x: uchar): uchar;

export declare function cv_abs(x: schar): schar;

export declare function cv_abs(x: ushort): ushort;

export declare function cv_abs(x: short): int;

export declare function CV_XADD(addr: any, delta: int): any;

/**
 * The function computes an integer i such that: `\\[i \\le \\texttt{value} < i+1\\]`
 *
 * @param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the result
 * is not defined.
 */
export declare function cvCeil(value: double): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function cvCeil(value: float): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function cvCeil(value: int): int;

/**
 * The function computes an integer i such that: `\\[i \\le \\texttt{value} < i+1\\]`
 *
 * @param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the result
 * is not defined.
 */
export declare function cvFloor(value: double): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function cvFloor(value: float): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function cvFloor(value: int): int;

/**
 * The function returns 1 if the argument is a plus or minus infinity (as defined by IEEE754 standard)
 * and 0 otherwise.
 *
 * @param value The input floating-point value
 */
export declare function cvIsInf(value: double): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function cvIsInf(value: float): int;

/**
 * The function returns 1 if the argument is Not A Number (as defined by IEEE754 standard), 0
 * otherwise.
 *
 * @param value The input floating-point value
 */
export declare function cvIsNaN(value: double): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function cvIsNaN(value: float): int;

/**
 * @param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the result
 * is not defined.
 */
export declare function cvRound(value: double): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function cvRound(value: float): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function cvRound(value: int): int;

/**
 * Use this function instead of `ceil((float)a / b)` expressions.
 *
 * [alignSize]
 */
export declare function divUp(a: int, b: any): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function divUp(a: size_t, b: any): size_t;

export declare function dumpInputArray(argument: InputArray): String;

export declare function dumpInputArrayOfArrays(
  argument: InputArrayOfArrays,
): String;

export declare function dumpInputOutputArray(
  argument: InputOutputArray,
): String;

export declare function dumpInputOutputArrayOfArrays(
  argument: InputOutputArrayOfArrays,
): String;

/**
 * By default the function prints information about the error to stderr, then it either stops if
 * [cv::setBreakOnError()] had been called before or raises the exception. It is possible to alternate
 * error processing by using [redirectError()].
 *
 * @param exc the exception raisen.
 */
export declare function error(exc: any): void;

/**
 * By default the function prints information about the error to stderr, then it either stops if
 * [setBreakOnError()] had been called before or raises the exception. It is possible to alternate
 * error processing by using [redirectError()].
 *
 * [CV_Error], [CV_Error_], [CV_Assert], [CV_DbgAssert]
 *
 * @param _code - error code (Error::Code)
 *
 * @param _err - error description
 *
 * @param _func - function name. Available only when the compiler supports getting it
 *
 * @param _file - source file name where the error has occurred
 *
 * @param _line - line number in the source file where the error has occurred
 */
export declare function error(
  _code: int,
  _err: any,
  _func: any,
  _file: any,
  _line: int,
): void;

/**
 * The function fastAtan2 calculates the full-range angle of an input 2D vector. The angle is measured
 * in degrees and varies from 0 to 360 degrees. The accuracy is about 0.3 degrees.
 *
 * @param y y-coordinate of the vector.
 *
 * @param x x-coordinate of the vector.
 */
export declare function fastAtan2(y: float, x: float): float;

/**
 * The function deallocates the buffer allocated with fastMalloc . If NULL pointer is passed, the
 * function does nothing. C version of the function clears the pointer *pptr* to avoid problems with
 * double memory deallocation.
 *
 * @param ptr Pointer to the allocated buffer.
 */
export declare function fastFree(ptr: any): void;

/**
 * The function allocates the buffer of the specified size and returns it. When the buffer size is 16
 * bytes or more, the returned buffer is aligned to 16 bytes.
 *
 * @param bufSize Allocated buffer size.
 */
export declare function fastMalloc(bufSize: size_t): any;

export declare function forEach_impl(
  arg94: any,
  arg95: any,
  operation: any,
): any;

/**
 * Returned value is raw cmake output including version control system revision, compiler version,
 * compiler flags, enabled modules and third party libraries, etc. Output format depends on target
 * architecture.
 */
export declare function getBuildInformation(): any;

/**
 * Returned value is a string containing space separated list of CPU features with following markers:
 *
 * no markers - baseline features
 * prefix `*` - features enabled in dispatcher
 * suffix `?` - features enabled but not available in HW
 *
 * Example: `SSE SSE2 SSE3 *SSE4.1 *SSE4.2 *FP16 *AVX *AVX2 *AVX512-SKX?`
 */
export declare function getCPUFeaturesLine(): any;

/**
 * The function returns the current number of CPU ticks on some architectures (such as x86, x64,
 * PowerPC). On other platforms the function is equivalent to getTickCount. It can also be used for
 * very accurate time measurements, as well as for [RNG] initialization. Note that in case of multi-CPU
 * systems a thread, from which getCPUTickCount is called, can be suspended and resumed at another CPU
 * with its own counter. So, theoretically (and practically) the subsequent calls to the function do
 * not necessary return the monotonously increasing values. Also, since a modern CPU varies the CPU
 * frequency depending on the load, the number of CPU clocks spent in some code cannot be directly
 * converted to time units. Therefore, getTickCount is generally a preferable solution for measuring
 * execution time.
 */
export declare function getCPUTickCount(): int64;

export declare function getElemSize(type: int): size_t;

/**
 * Returns empty string if feature is not defined
 */
export declare function getHardwareFeatureName(feature: int): String;

export declare function getNumberOfCPUs(): int;

/**
 * Always returns 1 if OpenCV is built without threading support.
 *
 * The exact meaning of return value depends on the threading framework used by OpenCV library:
 *
 * `TBB` - The number of threads, that OpenCV will try to use for parallel regions. If there is any
 * tbb::thread_scheduler_init in user code conflicting with OpenCV, then function returns default
 * number of threads used by TBB library.
 * `OpenMP` - An upper bound on the number of threads that could be used to form a new team.
 * `Concurrency` - The number of threads, that OpenCV will try to use for parallel regions.
 * `GCD` - Unsupported; returns the GCD thread pool limit (512) for compatibility.
 * `C=` - The number of threads, that OpenCV will try to use for parallel regions, if before called
 * setNumThreads with threads > 0, otherwise returns the number of logical CPUs, available for the
 * process.
 *
 * [setNumThreads], [getThreadNum]
 */
export declare function getNumThreads(): int;

/**
 * The exact meaning of the return value depends on the threading framework used by OpenCV library:
 *
 * `TBB` - Unsupported with current 4.1 TBB release. Maybe will be supported in future.
 * `OpenMP` - The thread number, within the current team, of the calling thread.
 * `Concurrency` - An ID for the virtual processor that the current context is executing on (0 for
 * master thread and unique number for others, but not necessary 1,2,3,...).
 * `GCD` - System calling thread's ID. Never returns 0 inside parallel region.
 * `C=` - The index of the current parallel task.
 *
 * [setNumThreads], [getNumThreads]
 */
export declare function getThreadNum(): int;

/**
 * The function returns the number of ticks after the certain event (for example, when the machine was
 * turned on). It can be used to initialize [RNG] or to measure a function execution time by reading
 * the tick count before and after the function call.
 *
 * [getTickFrequency], [TickMeter]
 */
export declare function getTickCount(): int64;

/**
 * The function returns the number of ticks per second. That is, the following code computes the
 * execution time in seconds:
 *
 * ```cpp
 * double t = (double)getTickCount();
 * // do something ...
 * t = ((double)getTickCount() - t)/getTickFrequency();
 * ```
 *
 * [getTickCount], [TickMeter]
 */
export declare function getTickFrequency(): double;

export declare function getVersionMajor(): int;

export declare function getVersionMinor(): int;

export declare function getVersionRevision(): int;

/**
 * For example "3.4.1-dev".
 *
 * getMajorVersion, getMinorVersion, getRevisionVersion
 */
export declare function getVersionString(): String;

export declare function glob(
  pattern: String,
  result: any,
  recursive?: bool,
): void;

/**
 * proxy for hal::LU
 */
export declare function LU(
  A: any,
  astep: size_t,
  m: int,
  b: any,
  bstep: size_t,
  n: int,
): int;

/**
 * proxy for hal::LU
 */
export declare function LU(
  A: any,
  astep: size_t,
  m: int,
  b: any,
  bstep: size_t,
  n: int,
): int;

export declare function normInf(arg96: any, arg97: any, a: any, n: int): any;

export declare function normInf(
  arg98: any,
  arg99: any,
  a: any,
  b: any,
  n: int,
): any;

export declare function normL1(arg100: any, arg101: any, a: any, n: int): any;

export declare function normL1(
  arg102: any,
  arg103: any,
  a: any,
  b: any,
  n: int,
): any;

export declare function normL1(a: any, b: any, n: int): float;

export declare function normL1(a: uchar, b: uchar, n: int): uchar;

export declare function normL2Sqr(
  arg104: any,
  arg105: any,
  a: any,
  n: int,
): any;

export declare function normL2Sqr(
  arg106: any,
  arg107: any,
  a: any,
  b: any,
  n: int,
): any;

export declare function normL2Sqr(a: any, b: any, n: int): float;

export declare function parallel_for_(
  range: any,
  body: any,
  nstripes?: double,
): void;

export declare function parallel_for_(
  range: any,
  functor: any,
  nstripes?: double,
): void;

/**
 * The function sets the new error handler, called from [cv::error()].
 *
 * the previous error handler
 *
 * @param errCallback the new error handler. If NULL, the default error handler is used.
 *
 * @param userdata the optional user data pointer, passed to the callback.
 *
 * @param prevUserdata the optional output parameter where the previous user data pointer is stored
 */
export declare function redirectError(
  errCallback: ErrorCallback,
  userdata?: any,
  prevUserdata?: any,
): ErrorCallback;

/**
 * Use this function instead of `ceil((float)a / b) * b` expressions.
 *
 * [divUp]
 */
export declare function roundUp(a: int, b: any): int;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function roundUp(a: size_t, b: any): size_t;

/**
 * The function saturate_cast resembles the standard C++ cast operations, such as static_cast<T>() and
 * others. It perform an efficient and accurate conversion from one primitive type to another (see the
 * introduction chapter). saturate in the name means that when the input value v is out of the range of
 * the target type, the result is not formed just by taking low bits of the input, but instead the
 * value is clipped. For example:
 *
 * ```cpp
 * uchar a = saturate_cast<uchar>(-100); // a = 0 (UCHAR_MIN)
 * short b = saturate_cast<short>(33333.33333); // b = 32767 (SHRT_MAX)
 * ```
 *
 *  Such clipping is done when the target type is unsigned char , signed char , unsigned short or
 * signed short . For 32-bit integers, no clipping is done.
 *
 * When the parameter is a floating-point value and the target type is an integer (8-, 16- or 32-bit),
 * the floating-point value is first rounded to the nearest integer and then clipped if needed (when
 * the target type is 8- or 16-bit).
 *
 * This operation is used in the simplest or most complex image processing functions in OpenCV.
 *
 * [add], [subtract], [multiply], [divide], [Mat::convertTo]
 *
 * @param v Function parameter.
 */
export declare function saturate_cast(arg108: any, v: uchar): uchar;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function saturate_cast(arg109: any, v: schar): schar;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function saturate_cast(arg110: any, v: ushort): ushort;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function saturate_cast(arg111: any, v: short): any;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function saturate_cast(arg112: any, v: unsigned): any;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function saturate_cast(arg113: any, v: int): any;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function saturate_cast(arg114: any, v: float): any;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function saturate_cast(arg115: any, v: double): any;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function saturate_cast(arg116: any, v: int64): int64;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function saturate_cast(arg117: any, v: uint64): uint64;

/**
 * This is an overloaded member function, provided for convenience. It differs from the above function
 * only in what argument(s) it accepts.
 */
export declare function saturate_cast(arg118: any, v: float16_t): any;

/**
 * When the break-on-error mode is set, the default error handler issues a hardware exception, which
 * can make debugging more convenient.
 *
 * the previous state
 */
export declare function setBreakOnError(flag: bool): bool;

/**
 * If threads == 0, OpenCV will disable threading optimizations and run all it's functions
 * sequentially. Passing threads < 0 will reset threads number to system default. This function must be
 * called outside of parallel region.
 *
 * OpenCV will try to run its functions with specified threads number, but some behaviour differs from
 * framework:
 *
 * `TBB` - User-defined parallel constructions will run with the same threads number, if another is not
 * specified. If later on user creates his own scheduler, OpenCV will use it.
 * `OpenMP` - No special defined behaviour.
 * `Concurrency` - If threads == 1, OpenCV will disable threading optimizations and run its functions
 * sequentially.
 * `GCD` - Supports only values <= 0.
 * `C=` - No special defined behaviour.
 *
 * [getNumThreads], [getThreadNum]
 *
 * @param nthreads Number of threads used by OpenCV.
 */
export declare function setNumThreads(nthreads: int): void;

/**
 * The function can be used to dynamically turn on and off optimized dispatched code (code that uses
 * SSE4.2, AVX/AVX2, and other instructions on the platforms that support it). It sets a global flag
 * that is further checked by OpenCV functions. Since the flag is not checked in the inner OpenCV
 * loops, it is only safe to call the function on the very top level in your application where you can
 * be sure that no other OpenCV function is currently executed.
 *
 * By default, the optimized code is enabled unless you disable it in CMake. The current status can be
 * retrieved using useOptimized.
 *
 * @param onoff The boolean flag specifying whether the optimized code should be used (onoff=true) or
 * not (onoff=false).
 */
export declare function setUseOptimized(onoff: bool): void;

export declare function tempfile(suffix?: any): String;

export declare function testAsyncArray(argument: InputArray): AsyncArray;

export declare function testAsyncException(): AsyncArray;

/**
 * The function returns true if the optimized code is enabled. Otherwise, it returns false.
 */
export declare function useOptimized(): bool;

export declare const CPU_MMX: CpuFeatures; // initializer: = 1

export declare const CPU_SSE: CpuFeatures; // initializer: = 2

export declare const CPU_SSE2: CpuFeatures; // initializer: = 3

export declare const CPU_SSE3: CpuFeatures; // initializer: = 4

export declare const CPU_SSSE3: CpuFeatures; // initializer: = 5

export declare const CPU_SSE4_1: CpuFeatures; // initializer: = 6

export declare const CPU_SSE4_2: CpuFeatures; // initializer: = 7

export declare const CPU_POPCNT: CpuFeatures; // initializer: = 8

export declare const CPU_FP16: CpuFeatures; // initializer: = 9

export declare const CPU_AVX: CpuFeatures; // initializer: = 10

export declare const CPU_AVX2: CpuFeatures; // initializer: = 11

export declare const CPU_FMA3: CpuFeatures; // initializer: = 12

export declare const CPU_AVX_512F: CpuFeatures; // initializer: = 13

export declare const CPU_AVX_512BW: CpuFeatures; // initializer: = 14

export declare const CPU_AVX_512CD: CpuFeatures; // initializer: = 15

export declare const CPU_AVX_512DQ: CpuFeatures; // initializer: = 16

export declare const CPU_AVX_512ER: CpuFeatures; // initializer: = 17

export declare const CPU_AVX_512IFMA512: CpuFeatures; // initializer: = 18

export declare const CPU_AVX_512IFMA: CpuFeatures; // initializer: = 18

export declare const CPU_AVX_512PF: CpuFeatures; // initializer: = 19

export declare const CPU_AVX_512VBMI: CpuFeatures; // initializer: = 20

export declare const CPU_AVX_512VL: CpuFeatures; // initializer: = 21

export declare const CPU_AVX_512VBMI2: CpuFeatures; // initializer: = 22

export declare const CPU_AVX_512VNNI: CpuFeatures; // initializer: = 23

export declare const CPU_AVX_512BITALG: CpuFeatures; // initializer: = 24

export declare const CPU_AVX_512VPOPCNTDQ: CpuFeatures; // initializer: = 25

export declare const CPU_AVX_5124VNNIW: CpuFeatures; // initializer: = 26

export declare const CPU_AVX_5124FMAPS: CpuFeatures; // initializer: = 27

export declare const CPU_NEON: CpuFeatures; // initializer: = 100

export declare const CPU_VSX: CpuFeatures; // initializer: = 200

export declare const CPU_VSX3: CpuFeatures; // initializer: = 201

export declare const CPU_AVX512_SKX: CpuFeatures; // initializer: = 256

export declare const CPU_AVX512_COMMON: CpuFeatures; // initializer: = 257

export declare const CPU_AVX512_KNL: CpuFeatures; // initializer: = 258

export declare const CPU_AVX512_KNM: CpuFeatures; // initializer: = 259

export declare const CPU_AVX512_CNL: CpuFeatures; // initializer: = 260

export declare const CPU_AVX512_CEL: CpuFeatures; // initializer: = 261

export declare const CPU_AVX512_ICL: CpuFeatures; // initializer: = 262

export declare const CPU_MAX_FEATURE: CpuFeatures; // initializer: = 512

export declare const SORT_EVERY_ROW: SortFlags; // initializer: = 0

/**
 * each matrix column is sorted independently; this flag and the previous one are mutually exclusive.
 *
 */
export declare const SORT_EVERY_COLUMN: SortFlags; // initializer: = 1

/**
 * each matrix row is sorted in the ascending order.
 *
 */
export declare const SORT_ASCENDING: SortFlags; // initializer: = 0

/**
 * each matrix row is sorted in the descending order; this flag and the previous one are also mutually
 * exclusive.
 *
 */
export declare const SORT_DESCENDING: SortFlags; // initializer: = 16

export type CpuFeatures = any;

export type SortFlags = any;
