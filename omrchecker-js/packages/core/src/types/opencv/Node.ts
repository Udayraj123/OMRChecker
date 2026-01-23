import type { double, int } from "./_types";

export declare class Node {
  /**
   *   Class index normalized to 0..class_count-1 range and assigned to the node. It is used internally
   * in classification trees and tree ensembles.
   *
   */
  public classIdx: int;

  /**
   *   Default direction where to go (-1: left or +1: right). It helps in the case of missing values.
   *
   */
  public defaultDir: int;

  public left: int;

  public parent: int;

  public right: int;

  public split: int;

  /**
   *   Value at the node: a class label in case of classification or estimated function value in case of
   * regression.
   *
   */
  public value: double;

  public constructor();
}
