import type {
  diag_type,
  int,
  Matx_AddOp,
  Matx_DivOp,
  Matx_MatMulOp,
  Matx_MulOp,
  Matx_ScaleOp,
  Matx_SubOp,
  Matx_TOp,
  Vec,
  _T2,
  _Tp,
} from "./_types";

/**
 * If you need a more flexible type, use [Mat](#d3/d63/classcv_1_1Mat}) . The elements of the matrix M
 * are accessible using the M(i,j) notation. Most of the common matrix operations (see also
 * [MatrixExpressions](#d1/d10/classcv_1_1MatExpr_1MatrixExpressions}) ) are available. To do an
 * operation on [Matx](#de/de1/classcv_1_1Matx}) that is not implemented, you can easily convert the
 * matrix to [Mat](#d3/d63/classcv_1_1Mat}) and backwards:
 *
 * ```cpp
 * Matx33f m(1, 2, 3,
 *           4, 5, 6,
 *           7, 8, 9);
 * cout << sum(Mat(m*m.t())) << endl;
 * ```
 *
 *  Except of the plain constructor which takes a list of elements, [Matx](#de/de1/classcv_1_1Matx})
 * can be initialized from a C-array:
 *
 * ```cpp
 * float values[] = { 1, 2, 3};
 * Matx31f m(values);
 * ```
 *
 *  In case if C++11 features are available, std::initializer_list can be also used to initialize
 * [Matx](#de/de1/classcv_1_1Matx}):
 *
 * ```cpp
 * Matx31f m = { 1, 2, 3};
 * ```
 *
 * Source:
 * [opencv2/core/matx.hpp](https://github.com/opencv/opencv/tree/master/modules/core/include/opencv2/core/matx.hpp#L1185).
 *
 */
export declare class Matx {
  public val: _Tp;

  public constructor();

  public constructor(v0: _Tp);

  public constructor(v0: _Tp, v1: _Tp);

  public constructor(v0: _Tp, v1: _Tp, v2: _Tp);

  public constructor(v0: _Tp, v1: _Tp, v2: _Tp, v3: _Tp);

  public constructor(v0: _Tp, v1: _Tp, v2: _Tp, v3: _Tp, v4: _Tp);

  public constructor(v0: _Tp, v1: _Tp, v2: _Tp, v3: _Tp, v4: _Tp, v5: _Tp);

  public constructor(
    v0: _Tp,
    v1: _Tp,
    v2: _Tp,
    v3: _Tp,
    v4: _Tp,
    v5: _Tp,
    v6: _Tp,
  );

  public constructor(
    v0: _Tp,
    v1: _Tp,
    v2: _Tp,
    v3: _Tp,
    v4: _Tp,
    v5: _Tp,
    v6: _Tp,
    v7: _Tp,
  );

  public constructor(
    v0: _Tp,
    v1: _Tp,
    v2: _Tp,
    v3: _Tp,
    v4: _Tp,
    v5: _Tp,
    v6: _Tp,
    v7: _Tp,
    v8: _Tp,
  );

  public constructor(
    v0: _Tp,
    v1: _Tp,
    v2: _Tp,
    v3: _Tp,
    v4: _Tp,
    v5: _Tp,
    v6: _Tp,
    v7: _Tp,
    v8: _Tp,
    v9: _Tp,
  );

  public constructor(
    v0: _Tp,
    v1: _Tp,
    v2: _Tp,
    v3: _Tp,
    v4: _Tp,
    v5: _Tp,
    v6: _Tp,
    v7: _Tp,
    v8: _Tp,
    v9: _Tp,
    v10: _Tp,
    v11: _Tp,
  );

  public constructor(
    v0: _Tp,
    v1: _Tp,
    v2: _Tp,
    v3: _Tp,
    v4: _Tp,
    v5: _Tp,
    v6: _Tp,
    v7: _Tp,
    v8: _Tp,
    v9: _Tp,
    v10: _Tp,
    v11: _Tp,
    v12: _Tp,
    v13: _Tp,
  );

  public constructor(
    v0: _Tp,
    v1: _Tp,
    v2: _Tp,
    v3: _Tp,
    v4: _Tp,
    v5: _Tp,
    v6: _Tp,
    v7: _Tp,
    v8: _Tp,
    v9: _Tp,
    v10: _Tp,
    v11: _Tp,
    v12: _Tp,
    v13: _Tp,
    v14: _Tp,
    v15: _Tp,
  );

  public constructor(vals: any);

  public constructor(arg334: any);

  public constructor(a: Matx, b: Matx, arg335: Matx_AddOp);

  public constructor(a: Matx, b: Matx, arg336: Matx_SubOp);

  public constructor(arg337: any, a: Matx, alpha: _T2, arg338: Matx_ScaleOp);

  public constructor(a: Matx, b: Matx, arg339: Matx_MulOp);

  public constructor(a: Matx, b: Matx, arg340: Matx_DivOp);

  public constructor(l: int, a: Matx, b: Matx, arg341: Matx_MatMulOp);

  public constructor(a: Matx, arg342: Matx_TOp);

  public col(i: int): Matx;

  public ddot(v: Matx): Matx;

  public diag(): diag_type;

  public div(a: Matx): Matx;

  public dot(v: Matx): Matx;

  public get_minor(m1: int, n1: int, base_row: int, base_col: int): Matx;

  public inv(method?: int, p_is_ok?: any): Matx;

  public mul(a: Matx): Matx;

  public reshape(m1: int, n1: int): Matx;

  public row(i: int): Matx;

  public solve(l: int, rhs: Matx, flags?: int): Matx;

  public solve(rhs: Vec, method: int): Vec;

  public t(): Matx;

  public static all(alpha: _Tp): Matx;

  public static diag(d: diag_type): Matx;

  public static eye(): Matx;

  public static ones(): Matx;

  public static randn(a: _Tp, b: _Tp): Matx;

  public static randu(a: _Tp, b: _Tp): Matx;

  public static zeros(): Matx;
}

export declare const rows: any; // initializer: = m

export declare const cols: any; // initializer: = n

export declare const channels: any; // initializer: = rows*cols

export declare const shortdim: any; // initializer: = (m < n ? m : n)
