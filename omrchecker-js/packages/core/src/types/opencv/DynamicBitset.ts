import type { bool, size_t } from "./_types";

/**
 * Class re-implementing the boost version of it This helps not depending on boost, it also does not do
 * the bound checks and has a way to reset a block for speed
 *
 * Source:
 * [opencv2/flann/dynamic_bitset.h](https://github.com/opencv/opencv/tree/master/modules/core/include/opencv2/flann/dynamic_bitset.h#L150).
 *
 */
export declare class DynamicBitset {
  /**
   *   default constructor
   */
  public constructor();

  /**
   *   only constructor we use in our code
   *
   * @param sz the size of the bitset (in bits)
   */
  public constructor(sz: size_t);

  /**
   *   Sets all the bits to 0
   */
  public clear(): void;

  /**
   *   true if the bitset is empty
   */
  public empty(): bool;

  /**
   *   set all the bits to 0
   */
  public reset(): void;

  public reset(index: size_t): void;

  public reset_block(index: size_t): void;

  /**
   *   resize the bitset so that it contains at least sz bits
   */
  public resize(sz: size_t): void;

  /**
   *   set a bit to true
   *
   * @param index the index of the bit to set to 1
   */
  public set(index: size_t): void;

  /**
   *   gives the number of contained bits
   */
  public size(): size_t;

  /**
   *   check if a bit is set
   *
   *   true if the bit is set
   *
   * @param index the index of the bit to check
   */
  public test(index: size_t): bool;
}
