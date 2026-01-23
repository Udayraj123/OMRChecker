import type { bool, int, int32_t, int64_t, uint32_t, uint64_t } from "./_types";

export declare class softdouble {
  public v: uint64_t;

  public constructor();

  public constructor(c: softdouble);

  public constructor(arg159: uint32_t);

  public constructor(arg160: uint64_t);

  public constructor(arg161: int32_t);

  public constructor(arg162: int64_t);

  public constructor(a: any);

  public getExp(): int;

  /**
   *   Returns a number 1 <= x < 2 with the same significand
   */
  public getFrac(): softdouble;

  public getSign(): bool;

  public isInf(): bool;

  public isNaN(): bool;

  public isSubnormal(): bool;

  public setExp(e: int): softdouble;

  /**
   *   Constructs a copy of a number with significand taken from parameter
   */
  public setFrac(s: softdouble): softdouble;

  public setSign(sign: bool): softdouble;

  public static eps(): softdouble;

  /**
   *   Builds new value from raw binary representation
   */
  public static fromRaw(a: uint64_t): softdouble;

  public static inf(): softdouble;

  public static max(): softdouble;

  public static min(): softdouble;

  public static nan(): softdouble;

  public static one(): softdouble;

  public static pi(): softdouble;

  public static zero(): softdouble;
}
