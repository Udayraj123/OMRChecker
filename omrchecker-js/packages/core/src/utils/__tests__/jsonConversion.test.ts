import { describe, it, expect } from "vitest";
import {
  camelToSnake,
  snakeToCamel,
  screamingToCamel,
  validateNoKeyClash,
  convertDictKeysToSnake,
  convertDictKeysToCamel,
} from "../jsonConversion";

describe("jsonConversion", () => {
  describe("camelToSnake", () => {
    it("should convert camelCase to snake_case", () => {
      expect(camelToSnake("showImageLevel")).toBe("show_image_level");
      expect(camelToSnake("MLConfig")).toBe("ml_config");
      expect(camelToSnake("globalPageThreshold")).toBe("global_page_threshold");
    });

    it("should handle single words", () => {
      expect(camelToSnake("hello")).toBe("hello");
      expect(camelToSnake("HELLO")).toBe("hello");
    });
  });

  describe("snakeToCamel", () => {
    it("should convert snake_case to camelCase", () => {
      expect(snakeToCamel("show_image_level")).toBe("showImageLevel");
      expect(snakeToCamel("ml_config")).toBe("mlConfig");
      expect(snakeToCamel("global_page_threshold")).toBe("globalPageThreshold");
    });

    it("should handle single words", () => {
      expect(snakeToCamel("hello")).toBe("hello");
    });
  });

  describe("screamingToCamel", () => {
    it("should convert SCREAMING_SNAKE_CASE to camelCase", () => {
      expect(screamingToCamel("GLOBAL_PAGE_THRESHOLD")).toBe(
        "globalPageThreshold"
      );
      expect(screamingToCamel("MIN_JUMP")).toBe("minJump");
    });
  });

  describe("validateNoKeyClash", () => {
    it("should not throw for keys without clash", () => {
      expect(() =>
        validateNoKeyClash({ userName: "Alice", email: "test@example.com" })
      ).not.toThrow();
    });

    it("should throw for keys that clash", () => {
      expect(() =>
        validateNoKeyClash({ userName: "Alice", user_name: "Bob" })
      ).toThrow(
        "Key clash detected: 'userName' and 'user_name' both convert to 'user_name'"
      );
    });

    it("should detect nested clashes", () => {
      expect(() =>
        validateNoKeyClash({
          config: {
            showDebug: true,
            show_debug: false,
          },
        })
      ).toThrow("at 'config': Key clash detected");
    });

    it("should detect clashes in arrays", () => {
      expect(() =>
        validateNoKeyClash({
          items: [{ itemName: "A", item_name: "B" }],
        })
      ).toThrow("at 'items[0]': Key clash detected");
    });

    it("should handle non-object values", () => {
      expect(() => validateNoKeyClash({ value: 123 })).not.toThrow();
      expect(() => validateNoKeyClash({ value: "string" })).not.toThrow();
      expect(() => validateNoKeyClash({ value: null })).not.toThrow();
    });
  });

  describe("convertDictKeysToSnake", () => {
    it("should convert top-level keys", () => {
      const input = { userName: "Alice", userEmail: "test@example.com" };
      const expected = { user_name: "Alice", user_email: "test@example.com" };
      expect(convertDictKeysToSnake(input)).toEqual(expected);
    });

    it("should convert nested keys", () => {
      const input = {
        userConfig: {
          showDebug: true,
          maxRetries: 3,
        },
      };
      const expected = {
        user_config: {
          show_debug: true,
          max_retries: 3,
        },
      };
      expect(convertDictKeysToSnake(input)).toEqual(expected);
    });

    it("should convert keys in arrays", () => {
      const input = {
        users: [
          { userName: "Alice", userId: 1 },
          { userName: "Bob", userId: 2 },
        ],
      };
      const expected = {
        users: [
          { user_name: "Alice", user_id: 1 },
          { user_name: "Bob", user_id: 2 },
        ],
      };
      expect(convertDictKeysToSnake(input)).toEqual(expected);
    });
  });

  describe("convertDictKeysToCamel", () => {
    it("should convert top-level keys", () => {
      const input = { user_name: "Alice", user_email: "test@example.com" };
      const expected = { userName: "Alice", userEmail: "test@example.com" };
      expect(convertDictKeysToCamel(input)).toEqual(expected);
    });

    it("should convert nested keys", () => {
      const input = {
        user_config: {
          show_debug: true,
          max_retries: 3,
        },
      };
      const expected = {
        userConfig: {
          showDebug: true,
          maxRetries: 3,
        },
      };
      expect(convertDictKeysToCamel(input)).toEqual(expected);
    });

    it("should convert keys in arrays", () => {
      const input = {
        users: [
          { user_name: "Alice", user_id: 1 },
          { user_name: "Bob", user_id: 2 },
        ],
      };
      const expected = {
        users: [
          { userName: "Alice", userId: 1 },
          { userName: "Bob", userId: 2 },
        ],
      };
      expect(convertDictKeysToCamel(input)).toEqual(expected);
    });
  });
});
