import type { double, int, Mat, MatExpr, Scalar, Size } from "./_types";

export declare class MatOp {
  public constructor();

  public abs(expr: MatExpr, res: MatExpr): MatExpr;

  public add(expr1: MatExpr, expr2: MatExpr, res: MatExpr): MatExpr;

  public add(expr1: MatExpr, s: Scalar, res: MatExpr): MatExpr;

  public assign(expr: MatExpr, m: Mat, type?: int): MatExpr;

  public augAssignAdd(expr: MatExpr, m: Mat): MatExpr;

  public augAssignAnd(expr: MatExpr, m: Mat): MatExpr;

  public augAssignDivide(expr: MatExpr, m: Mat): MatExpr;

  public augAssignMultiply(expr: MatExpr, m: Mat): MatExpr;

  public augAssignOr(expr: MatExpr, m: Mat): MatExpr;

  public augAssignSubtract(expr: MatExpr, m: Mat): MatExpr;

  public augAssignXor(expr: MatExpr, m: Mat): MatExpr;

  public diag(expr: MatExpr, d: int, res: MatExpr): MatExpr;

  public divide(
    expr1: MatExpr,
    expr2: MatExpr,
    res: MatExpr,
    scale?: double,
  ): MatExpr;

  public divide(s: double, expr: MatExpr, res: MatExpr): MatExpr;

  public elementWise(expr: MatExpr): MatExpr;

  public invert(expr: MatExpr, method: int, res: MatExpr): MatExpr;

  public matmul(expr1: MatExpr, expr2: MatExpr, res: MatExpr): MatExpr;

  public multiply(
    expr1: MatExpr,
    expr2: MatExpr,
    res: MatExpr,
    scale?: double,
  ): MatExpr;

  public multiply(expr1: MatExpr, s: double, res: MatExpr): MatExpr;

  public roi(
    expr: MatExpr,
    rowRange: Range,
    colRange: Range,
    res: MatExpr,
  ): MatExpr;

  public size(expr: MatExpr): Size;

  public subtract(expr1: MatExpr, expr2: MatExpr, res: MatExpr): MatExpr;

  public subtract(s: Scalar, expr: MatExpr, res: MatExpr): Scalar;

  public transpose(expr: MatExpr, res: MatExpr): MatExpr;

  public type(expr: MatExpr): MatExpr;
}
