import type { bool, int, int32_t, int64_t, uint32_t, uint64_t } from "./_types";

export declare class softfloat {
  public v: uint32_t;

  public constructor();

  public constructor(c: softfloat);

  public constructor(arg174: uint32_t);

  public constructor(arg175: uint64_t);

  public constructor(arg176: int32_t);

  public constructor(arg177: int64_t);

  public constructor(a: any);

  public getExp(): int;

  /**
   *   Returns a number 1 <= x < 2 with the same significand
   */
  public getFrac(): softfloat;

  public getSign(): bool;

  public isInf(): bool;

  public isNaN(): bool;

  public isSubnormal(): bool;

  public setExp(e: int): softfloat;

  /**
   *   Constructs a copy of a number with significand taken from parameter
   */
  public setFrac(s: softfloat): softfloat;

  public setSign(sign: bool): softfloat;

  public static eps(): softfloat;

  /**
   *   Builds new value from raw binary representation
   */
  public static fromRaw(a: uint32_t): softfloat;

  public static inf(): softfloat;

  public static max(): softfloat;

  public static min(): softfloat;

  public static nan(): softfloat;

  public static one(): softfloat;

  public static pi(): softfloat;

  public static zero(): softfloat;
}
