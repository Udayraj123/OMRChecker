# Testing Tasks Breakdown - Subagent Assignments

**Date**: 2026-02-28  
**Total Issues**: 10 tasks created  
**Status**: 7 ready, 3 blocked (waiting for Vitest setup)

---

## Task Overview

### Priority 1 (High Priority - Do First)
1. **omr-taw**: Vitest-Setup-Agent - Set up Vitest infrastructure
2. **omr-z8z**: Test-Author-Math - Write unit tests for math.ts (blocked by omr-taw)
3. **omr-j58**: Schema-Validator - Write schema validation tests (blocked by omr-taw)
4. **omr-8rc**: Memory-Auditor - Write OpenCV.js memory leak tests (blocked by omr-taw)

### Priority 2 (Medium Priority - Follow Up)
5. **omr-73f**: Test-Author-Geometry - Write unit tests for geometry.ts (blocked by omr-taw)
6. **omr-568**: Test-Author-Stats - Write unit tests for stats.ts (blocked by omr-taw)
7. **omr-3o6**: Test-Author-Checksum - Write unit tests for checksum.ts (blocked by omr-taw)
8. **omr-65w**: Exception-Tester - Write snapshot tests for exceptions.ts (blocked by omr-taw)
9. **omr-r0v**: Integration-Tester - Write processor integration tests (blocked by omr-taw)
10. **omr-owp**: Parity-Validator - Create Python-TypeScript parity tests (blocked by math, stats, checksum)

---

## Dependency Graph

```
omr-taw (Vitest Setup) [READY - START HERE]
  ├─> omr-z8z (Math tests) [P1]
  ├─> omr-73f (Geometry tests) [P2]
  ├─> omr-568 (Stats tests) [P2]
  ├─> omr-3o6 (Checksum tests) [P2]
  ├─> omr-j58 (Schema tests) [P1]
  ├─> omr-r0v (Integration tests) [P2]
  ├─> omr-8rc (Memory tests) [P1]
  └─> omr-65w (Exception tests) [P2]

omr-z8z, omr-568, omr-3o6 (Unit tests complete)
  └─> omr-owp (Parity tests) [P2]
```

---

## Subagent Roles

### 1. Vitest-Setup-Agent (omr-taw)
**Priority**: 1 (CRITICAL - BLOCKS ALL OTHER TASKS)  
**Status**: Ready to start immediately

**Responsibilities**:
- Create `omrchecker-js/packages/core/vitest.config.ts`
- Create `omrchecker-js/packages/core/tests/setup.ts`
- Install dependencies: `vitest`, `@vitest/ui`, `@techstark/opencv-js`
- Configure test environment (jsdom)
- Set up OpenCV.js async loading
- Create memory leak detection utilities (cv.matCount wrapper)
- Verify `npm test` runs successfully

**Success Criteria**:
- ✅ Vitest config created with 90%+ coverage thresholds
- ✅ Test setup handles OpenCV.js initialization
- ✅ `npm test` command works (even with 0 tests)
- ✅ Coverage reporting configured

**Time Estimate**: 30-60 minutes

---

### 2. Test-Author-Math (omr-z8z)
**Priority**: 1 (First proof-of-concept test)  
**Status**: Blocked by omr-taw  
**Blocks**: omr-owp (parity tests)

**Responsibilities**:
- Create `tests/unit/math.test.ts`
- Test MathUtils class (19 methods total)
- Focus on key functions:
  - `distance(p1, p2)` - Euclidean distance
  - `orderFourPoints(points)` - Clockwise ordering from top-left
  - `angleBetweenLines(line1, line2)` - Angle calculation
  - `lineIntersection(line1, line2)` - Intersection point
- Include edge cases: negative coords, zero lengths, collinear points
- Target 90%+ coverage

**Success Criteria**:
- ✅ All 19 methods have at least 1 test
- ✅ Edge cases covered
- ✅ All tests passing
- ✅ Coverage ≥90%

**Time Estimate**: 45-60 minutes

---

### 3. Test-Author-Geometry (omr-73f)
**Priority**: 2  
**Status**: Blocked by omr-taw

**Responsibilities**:
- Create `tests/unit/geometry.test.ts`
- Test rectangle operations, point transformations
- Edge cases: zero dimensions, negative coordinates

**Success Criteria**:
- ✅ Coverage ≥90%
- ✅ All tests passing

**Time Estimate**: 20-30 minutes

---

### 4. Test-Author-Stats (omr-568)
**Priority**: 2  
**Status**: Blocked by omr-taw  
**Blocks**: omr-owp (parity tests)

**Responsibilities**:
- Create `tests/unit/stats.test.ts`
- Test statistical functions: mean, median, mode, variance
- Edge cases: empty arrays, single values, outliers

**Success Criteria**:
- ✅ Coverage ≥90%
- ✅ All tests passing

**Time Estimate**: 20-30 minutes

---

### 5. Test-Author-Checksum (omr-3o6)
**Priority**: 2  
**Status**: Blocked by omr-taw  
**Blocks**: omr-owp (parity tests)

**Responsibilities**:
- Create `tests/unit/checksum.test.ts`
- Test checksum algorithms (CRC, hash functions)
- Use known test vectors

**Success Criteria**:
- ✅ Coverage ≥90%
- ✅ All tests passing
- ✅ Verified against standard implementations

**Time Estimate**: 20-30 minutes

---

### 6. Schema-Validator (omr-j58)
**Priority**: 1  
**Status**: Blocked by omr-taw

**Responsibilities**:
- Create `tests/schema/` directory
- Create `template.test.ts`, `config.test.ts`, `evaluation.test.ts`
- Test Zod schema validation:
  - Valid inputs pass
  - Invalid inputs throw
  - Defaults applied correctly
  - Nested validation works
- Target 100% field coverage

**Success Criteria**:
- ✅ All schema fields tested
- ✅ Valid/invalid cases covered
- ✅ Defaults verified
- ✅ All tests passing

**Time Estimate**: 45-60 minutes

---

### 7. Integration-Tester (omr-r0v)
**Priority**: 2  
**Status**: Blocked by omr-taw

**Responsibilities**:
- Create `tests/integration/` directory
- Create `base.test.ts`, `Pipeline.test.ts`, `coordinator.test.ts`
- Test async processor execution
- Test error handling
- Test pipeline orchestration
- Mock OpenCV.js initially

**Success Criteria**:
- ✅ Coverage ≥80%
- ✅ Async handling verified
- ✅ Error propagation tested
- ✅ All tests passing

**Time Estimate**: 60-90 minutes

---

### 8. Memory-Auditor (omr-8rc)
**Priority**: 1  
**Status**: Blocked by omr-taw

**Responsibilities**:
- Create `tests/memory/drawing.memory.test.ts`
- Create custom memory tracking utilities
- Test all DrawingUtils methods for cv.Mat leaks
- Verify temporary Mats are deleted

**Success Criteria**:
- ✅ 100% cv.Mat usage covered
- ✅ 0 memory leaks detected
- ✅ All tests passing

**Time Estimate**: 45-60 minutes

---

### 9. Exception-Tester (omr-65w)
**Priority**: 2  
**Status**: Blocked by omr-taw

**Responsibilities**:
- Create `tests/unit/exceptions.test.ts`
- Test error hierarchy with snapshots
- Test OMRCheckerError, ImageProcessingError, BubbleDetectionError
- Verify instanceof checks
- Test context formatting
- Test toString output

**Success Criteria**:
- ✅ All exception types tested
- ✅ Snapshots created
- ✅ instanceof hierarchy verified
- ✅ All tests passing

**Time Estimate**: 30-45 minutes

---

### 10. Parity-Validator (omr-owp)
**Priority**: 2  
**Status**: Blocked by omr-z8z, omr-568, omr-3o6  
**Depends on**: Math, Stats, Checksum tests complete

**Responsibilities**:
- Create `scripts/generate_test_fixtures.py`
- Generate Python test data for key functions
- Create `tests/parity/python-comparison.test.ts`
- Verify TypeScript matches Python output
- Document any intentional differences

**Success Criteria**:
- ✅ Python fixtures generated
- ✅ Parity verified for math, stats, checksum
- ✅ Differences documented
- ✅ All tests passing

**Time Estimate**: 60-90 minutes

---

## Execution Strategy

### Week 1: Foundation (Days 1-2)
**Day 1 Morning**: Vitest-Setup-Agent completes omr-taw  
**Day 1 Afternoon**: Test-Author-Math completes omr-z8z (proof of concept)  
**Day 2**: Test-Author-Geometry, Stats, Checksum complete in parallel

**Milestone**: Unit tests for utilities complete (90%+ coverage)

### Week 1: Validation (Days 3-5)
**Day 3**: Schema-Validator completes omr-j58  
**Day 4**: Memory-Auditor completes omr-8rc  
**Day 5**: Exception-Tester completes omr-65w

**Milestone**: Schema validation + memory safety verified

### Week 2: Integration (Days 1-3)
**Day 1-2**: Integration-Tester completes omr-r0v  
**Day 3**: Parity-Validator completes omr-owp

**Milestone**: Full integration + parity verification complete

---

## Commands for Each Agent

### Start Work
```bash
# Claim issue
bd update <issue-id> --status in_progress --json

# Example
bd update omr-taw --status in_progress --json
```

### During Work
```bash
# Run tests
cd omrchecker-js/packages/core
npm test

# Run specific test
npm test math.test.ts

# Check coverage
npm run coverage

# Validate changes
git status
git diff
```

### Complete Work
```bash
# Stage changes
git add <files>

# Commit with issue reference
git commit -m "test: add unit tests for math.ts (omr-z8z)

Test-Author-Math: Comprehensive unit tests for MathUtils
- 19 methods covered with edge cases
- 92% coverage achieved
- All tests passing

Issue: omr-z8z"

# Close issue
bd close <issue-id> --reason "Tests complete, coverage target met" --json

# Example
bd close omr-z8z --reason "92% coverage, all tests passing" --json
```

---

## Success Metrics

### Per-Task Metrics
- ✅ Tests written and passing
- ✅ Coverage targets met (90% for unit, 80% for integration, 100% for schema)
- ✅ Git commit with issue reference
- ✅ Issue closed in beads

### Overall Metrics
- ✅ All 10 tasks completed
- ✅ 2,334 lines of TypeScript now have comprehensive tests
- ✅ CI/CD ready
- ✅ Phase 2 migration can proceed with confidence

---

## Risk Mitigation

### If OpenCV.js Loading Fails
- Skip memory tests gracefully
- Log warning about cv.matCount unavailable
- Continue with other tests

### If Coverage < Target
- Document reasons (e.g., complex error handling, browser-only code)
- Get user approval for exception
- Add TODO for future improvement

### If Python Fixtures Complex
- Start with simple functions (distance, mean, etc.)
- Expand gradually
- Document generation process

---

## Next Immediate Action

**START HERE**: Claim omr-taw and begin Vitest setup

```bash
bd update omr-taw --status in_progress --json
cd omrchecker-js/packages/core
# Follow docs/TESTING_STRATEGY.md for setup instructions
```

---

**Status**: Ready for parallel execution  
**Blocking**: None (omr-taw is ready)  
**Documentation**: See `docs/TESTING_STRATEGY.md` for detailed implementation guide

---

**End of testing tasks breakdown**
