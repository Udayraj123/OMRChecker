import type { int } from "./_types";

export declare class Logger {
  public static error(fmt: any, arg121: any): int;

  public static fatal(fmt: any, arg122: any): int;

  public static info(fmt: any, arg123: any): int;

  /**
   *   Print log message
   *
   * @param level Log level
   *
   * @param fmt Message format
   */
  public static log(level: int, fmt: any, arg124: any): int;

  /**
   *   Sets the logging destination
   *
   * @param name Filename or NULL for console
   */
  public static setDestination(name: any): void;

  /**
   *   Sets the logging level. All messages with lower priority will be ignored.
   *
   * @param level Logging level
   */
  public static setLevel(level: int): void;

  public static warn(fmt: any, arg125: any): int;
}
