/**
 * Integration tests for the processor pipeline: base, Pipeline, and coordinator.
 *
 * These tests verify that:
 *  - ProcessingContext factory produces correct defaults (base.ts)
 *  - Processor implementations conform to the abstract contract (base.ts)
 *  - ProcessingPipeline composes processors correctly (Pipeline.ts)
 *  - PreprocessingCoordinator delegates to per-image preprocessors (coordinator.ts)
 *
 * OpenCV.js (cv.Mat) is NOT available in the jsdom environment.
 * Every test uses `null as any` for Mat fields and avoids operations that
 * would dereference a real Mat, mirroring how the existing setup.ts handles it.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  createProcessingContext,
  Processor,
  type ProcessingContext,
} from '../../src/processors/base';
import { ProcessingPipeline } from '../../src/processors/Pipeline';
import { PreprocessingCoordinator } from '../../src/processors/image/coordinator';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Minimal Mat stand-in that satisfies the type without requiring cv.Mat. */
const fakeMat = null as any;

/** Minimal template that ProcessingPipeline + PreprocessingCoordinator expect. */
function makeTemplate(overrides: Record<string, any> = {}): any {
  return {
    tuningConfig: {
      outputs: { coloredOutputsEnabled: false },
    },
    templateLayout: {
      processingImageShape: [600, 800],
      outputImageShape: [],
      preProcessors: [] as Processor[],
      getCopyForShifting() {
        return this;
      },
      resetAllShifts() {},
    },
    ...overrides,
  };
}

/** Factory that builds a concrete Processor for testing. */
function makeProcessor(
  name: string,
  mutate?: (ctx: ProcessingContext) => Partial<ProcessingContext>
): Processor {
  return {
    getName: () => name,
    process(context: ProcessingContext): ProcessingContext {
      if (mutate) {
        return { ...context, ...mutate(context) };
      }
      return context;
    },
  };
}

/** Async variant of makeProcessor. */
function makeAsyncProcessor(
  name: string,
  mutate?: (ctx: ProcessingContext) => Partial<ProcessingContext>
): Processor {
  return {
    getName: () => name,
    async process(context: ProcessingContext): Promise<ProcessingContext> {
      if (mutate) {
        return { ...context, ...mutate(context) };
      }
      return context;
    },
  };
}

// ---------------------------------------------------------------------------
// 1. Base module — createProcessingContext
// ---------------------------------------------------------------------------

describe('base.ts — createProcessingContext', () => {
  it('populates all required fields with provided values', () => {
    const template = makeTemplate();
    const ctx = createProcessingContext('/tmp/sheet.png', fakeMat, fakeMat, template);

    expect(ctx.filePath).toBe('/tmp/sheet.png');
    expect(ctx.grayImage).toBe(fakeMat);
    expect(ctx.coloredImage).toBe(fakeMat);
    expect(ctx.template).toBe(template);
  });

  it('initialises processing results to empty / falsy defaults', () => {
    const ctx = createProcessingContext('test.png', fakeMat, fakeMat, makeTemplate());

    expect(ctx.omrResponse).toEqual({});
    expect(ctx.isMultiMarked).toBe(false);
    expect(ctx.fieldIdToInterpretation).toEqual({});
  });

  it('initialises evaluation results to zero / null defaults', () => {
    const ctx = createProcessingContext('test.png', fakeMat, fakeMat, makeTemplate());

    expect(ctx.score).toBe(0.0);
    expect(ctx.evaluationMeta).toBeNull();
    expect(ctx.evaluationConfigForResponse).toBeNull();
    expect(ctx.defaultAnswersSummary).toBe('');
  });

  it('initialises metadata to an empty object', () => {
    const ctx = createProcessingContext('test.png', fakeMat, fakeMat, makeTemplate());
    expect(ctx.metadata).toEqual({});
  });

  it('produces independent context objects for separate calls', () => {
    const ctx1 = createProcessingContext('a.png', fakeMat, fakeMat, makeTemplate());
    const ctx2 = createProcessingContext('b.png', fakeMat, fakeMat, makeTemplate());

    ctx1.omrResponse['Q1'] = 'A';

    expect(ctx2.omrResponse).toEqual({});
    expect(ctx1.filePath).not.toBe(ctx2.filePath);
  });
});

// ---------------------------------------------------------------------------
// 2. Base module — Processor abstract contract
// ---------------------------------------------------------------------------

describe('base.ts — Processor interface', () => {
  it('a synchronous processor returns the same ProcessingContext shape', () => {
    const proc = makeProcessor('Sync');
    const ctx = createProcessingContext('test.png', fakeMat, fakeMat, makeTemplate());
    const result = proc.process(ctx) as ProcessingContext;

    expect(result).toHaveProperty('filePath');
    expect(result).toHaveProperty('omrResponse');
    expect(result).toHaveProperty('score');
  });

  it('an async processor resolves to the same ProcessingContext shape', async () => {
    const proc = makeAsyncProcessor('Async');
    const ctx = createProcessingContext('test.png', fakeMat, fakeMat, makeTemplate());
    const result = await proc.process(ctx);

    expect(result).toHaveProperty('filePath');
    expect(result).toHaveProperty('omrResponse');
  });

  it('getName() returns the processor identifier without side-effects', () => {
    const proc = makeProcessor('MyProcessor');
    expect(proc.getName()).toBe('MyProcessor');
    expect(proc.getName()).toBe('MyProcessor'); // stable across calls
  });

  it('a processor can mutate and return context fields', () => {
    const proc = makeProcessor('Scorer', () => ({ score: 42.5, isMultiMarked: true }));
    const ctx = createProcessingContext('test.png', fakeMat, fakeMat, makeTemplate());
    const result = proc.process(ctx) as ProcessingContext;

    expect(result.score).toBe(42.5);
    expect(result.isMultiMarked).toBe(true);
  });

  it('a processor can add metadata entries', () => {
    const proc = makeProcessor('MetaTagger', (ctx) => ({
      metadata: { ...ctx.metadata, tagged: true, processorRan: 'MetaTagger' },
    }));
    const ctx = createProcessingContext('test.png', fakeMat, fakeMat, makeTemplate());
    const result = proc.process(ctx) as ProcessingContext;

    expect(result.metadata['tagged']).toBe(true);
    expect(result.metadata['processorRan']).toBe('MetaTagger');
  });
});

// ---------------------------------------------------------------------------
// 3. ProcessingPipeline — construction and inspection
// ---------------------------------------------------------------------------

describe('Pipeline.ts — ProcessingPipeline construction', () => {
  it('getTemplate() returns the template passed to the constructor', () => {
    const template = makeTemplate();
    const pipeline = new ProcessingPipeline(template);
    expect(pipeline.getTemplate()).toBe(template);
  });

  it('getTuningConfig() returns the template tuningConfig', () => {
    const template = makeTemplate();
    const pipeline = new ProcessingPipeline(template);
    expect(pipeline.getTuningConfig()).toBe(template.tuningConfig);
  });

  it('getTuningConfig() falls back to tuning_config snake_case key', () => {
    const template = { tuning_config: { outputs: {} }, templateLayout: makeTemplate().templateLayout };
    const pipeline = new ProcessingPipeline(template);
    expect(pipeline.getTuningConfig()).toBe(template.tuning_config);
  });

  it('starts with an empty processor list', () => {
    const pipeline = new ProcessingPipeline(makeTemplate());
    expect(pipeline.getProcessorNames()).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// 4. ProcessingPipeline — addProcessor / removeProcessor / getProcessorNames
// ---------------------------------------------------------------------------

describe('Pipeline.ts — addProcessor / removeProcessor', () => {
  let pipeline: ProcessingPipeline;

  beforeEach(() => {
    pipeline = new ProcessingPipeline(makeTemplate());
  });

  it('addProcessor appends a processor to the pipeline', () => {
    pipeline.addProcessor(makeProcessor('Alpha'));
    expect(pipeline.getProcessorNames()).toEqual(['Alpha']);
  });

  it('addProcessor preserves insertion order', () => {
    pipeline.addProcessor(makeProcessor('Alpha'));
    pipeline.addProcessor(makeProcessor('Beta'));
    pipeline.addProcessor(makeProcessor('Gamma'));
    expect(pipeline.getProcessorNames()).toEqual(['Alpha', 'Beta', 'Gamma']);
  });

  it('removeProcessor removes the named processor', () => {
    pipeline.addProcessor(makeProcessor('Alpha'));
    pipeline.addProcessor(makeProcessor('Beta'));
    pipeline.removeProcessor('Alpha');
    expect(pipeline.getProcessorNames()).toEqual(['Beta']);
  });

  it('removeProcessor is a no-op when the name does not exist', () => {
    pipeline.addProcessor(makeProcessor('Alpha'));
    pipeline.removeProcessor('NonExistent');
    expect(pipeline.getProcessorNames()).toEqual(['Alpha']);
  });

  it('removeProcessor removes all processors with the same name', () => {
    // Two processors sharing the same name (edge case).
    pipeline.addProcessor(makeProcessor('Dup'));
    pipeline.addProcessor(makeProcessor('Dup'));
    pipeline.removeProcessor('Dup');
    expect(pipeline.getProcessorNames()).toEqual([]);
  });

  it('adding then removing leaves an empty pipeline', () => {
    pipeline.addProcessor(makeProcessor('Only'));
    pipeline.removeProcessor('Only');
    expect(pipeline.getProcessorNames()).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// 5. ProcessingPipeline — processFile (sync processors)
// ---------------------------------------------------------------------------

describe('Pipeline.ts — processFile with sync processors', () => {
  it('returns default context when no processors are registered', async () => {
    const pipeline = new ProcessingPipeline(makeTemplate());
    const ctx = await pipeline.processFile('empty.png', fakeMat, fakeMat);

    expect(ctx.filePath).toBe('empty.png');
    expect(ctx.omrResponse).toEqual({});
    expect(ctx.score).toBe(0);
  });

  it('a single processor can write to omrResponse', async () => {
    const pipeline = new ProcessingPipeline(makeTemplate());
    pipeline.addProcessor(
      makeProcessor('ReadOMR', (ctx) => ({
        omrResponse: { Q1: 'B', Q2: 'C' },
      }))
    );

    const ctx = await pipeline.processFile('sheet.png', fakeMat, fakeMat);
    expect(ctx.omrResponse).toEqual({ Q1: 'B', Q2: 'C' });
  });

  it('processors run in registration order, each seeing prior mutations', async () => {
    const order: string[] = [];
    const pipeline = new ProcessingPipeline(makeTemplate());

    pipeline.addProcessor(
      makeProcessor('First', (ctx) => {
        order.push('First');
        return { metadata: { ...ctx.metadata, step: 1 } };
      })
    );
    pipeline.addProcessor(
      makeProcessor('Second', (ctx) => {
        order.push('Second');
        // Reads metadata written by First
        expect(ctx.metadata['step']).toBe(1);
        return { metadata: { ...ctx.metadata, step: 2 } };
      })
    );
    pipeline.addProcessor(
      makeProcessor('Third', (ctx) => {
        order.push('Third');
        expect(ctx.metadata['step']).toBe(2);
        return {};
      })
    );

    const ctx = await pipeline.processFile('chained.png', fakeMat, fakeMat);
    expect(order).toEqual(['First', 'Second', 'Third']);
    expect(ctx.metadata['step']).toBe(2);
  });

  it('later processor can override omrResponse written by an earlier one', async () => {
    const pipeline = new ProcessingPipeline(makeTemplate());

    pipeline.addProcessor(
      makeProcessor('Draft', () => ({ omrResponse: { Q1: 'A' } }))
    );
    pipeline.addProcessor(
      makeProcessor('Corrector', () => ({ omrResponse: { Q1: 'B' } }))
    );

    const ctx = await pipeline.processFile('override.png', fakeMat, fakeMat);
    expect(ctx.omrResponse['Q1']).toBe('B');
  });

  it('score can be accumulated across processors', async () => {
    const pipeline = new ProcessingPipeline(makeTemplate());

    pipeline.addProcessor(
      makeProcessor('Scorer1', (ctx) => ({ score: ctx.score + 10 }))
    );
    pipeline.addProcessor(
      makeProcessor('Scorer2', (ctx) => ({ score: ctx.score + 5 }))
    );

    const ctx = await pipeline.processFile('score.png', fakeMat, fakeMat);
    expect(ctx.score).toBe(15);
  });

  it('filePath in context matches the argument passed to processFile', async () => {
    const pipeline = new ProcessingPipeline(makeTemplate());
    pipeline.addProcessor(makeProcessor('Noop'));

    const ctx = await pipeline.processFile('/data/inputs/scan001.png', fakeMat, fakeMat);
    expect(ctx.filePath).toBe('/data/inputs/scan001.png');
  });
});

// ---------------------------------------------------------------------------
// 6. ProcessingPipeline — processFile (async processors)
// ---------------------------------------------------------------------------

describe('Pipeline.ts — processFile with async processors', () => {
  it('awaits async processors before passing context to the next', async () => {
    const pipeline = new ProcessingPipeline(makeTemplate());

    pipeline.addProcessor(
      makeAsyncProcessor('AsyncFirst', (ctx) => ({
        metadata: { ...ctx.metadata, asyncStep: 'done' },
      }))
    );
    pipeline.addProcessor(
      makeProcessor('SyncSecond', (ctx) => {
        expect(ctx.metadata['asyncStep']).toBe('done');
        return {};
      })
    );

    const ctx = await pipeline.processFile('async.png', fakeMat, fakeMat);
    expect(ctx.metadata['asyncStep']).toBe('done');
  });

  it('handles a pipeline of mixed sync and async processors', async () => {
    const pipeline = new ProcessingPipeline(makeTemplate());
    const log: string[] = [];

    pipeline.addProcessor(makeProcessor('SyncA', () => { log.push('SyncA'); return {}; }));
    pipeline.addProcessor(makeAsyncProcessor('AsyncB', () => { log.push('AsyncB'); return {}; }));
    pipeline.addProcessor(makeProcessor('SyncC', () => { log.push('SyncC'); return {}; }));
    pipeline.addProcessor(makeAsyncProcessor('AsyncD', () => { log.push('AsyncD'); return {}; }));

    await pipeline.processFile('mixed.png', fakeMat, fakeMat);
    expect(log).toEqual(['SyncA', 'AsyncB', 'SyncC', 'AsyncD']);
  });
});

// ---------------------------------------------------------------------------
// 7. ProcessingPipeline — multiple processFile calls (independence)
// ---------------------------------------------------------------------------

describe('Pipeline.ts — multiple processFile calls are independent', () => {
  it('two calls with different paths yield contexts with those paths', async () => {
    const pipeline = new ProcessingPipeline(makeTemplate());

    const [ctx1, ctx2] = await Promise.all([
      pipeline.processFile('scan_A.png', fakeMat, fakeMat),
      pipeline.processFile('scan_B.png', fakeMat, fakeMat),
    ]);

    expect(ctx1.filePath).toBe('scan_A.png');
    expect(ctx2.filePath).toBe('scan_B.png');
  });

  it('mutations in one invocation do not bleed into another', async () => {
    const pipeline = new ProcessingPipeline(makeTemplate());
    pipeline.addProcessor(
      makeProcessor('Write', () => ({ omrResponse: { Q1: 'X' } }))
    );

    const ctx1 = await pipeline.processFile('first.png', fakeMat, fakeMat);
    ctx1.omrResponse['Q99'] = 'mutated-after-resolve';

    const ctx2 = await pipeline.processFile('second.png', fakeMat, fakeMat);
    expect(ctx2.omrResponse['Q99']).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// 8. PreprocessingCoordinator — interface compliance
// ---------------------------------------------------------------------------

describe('coordinator.ts — PreprocessingCoordinator interface', () => {
  it('getName() returns "Preprocessing"', () => {
    const coord = new PreprocessingCoordinator(makeTemplate());
    expect(coord.getName()).toBe('Preprocessing');
  });

  it('implements the Processor interface (has process + getName)', () => {
    const coord = new PreprocessingCoordinator(makeTemplate());
    expect(typeof coord.process).toBe('function');
    expect(typeof coord.getName).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// 9. PreprocessingCoordinator — process() delegates to preProcessors
// ---------------------------------------------------------------------------

describe('coordinator.ts — process() with preProcessors', () => {
  it('returns context unchanged when preProcessors list is empty', async () => {
    const template = makeTemplate();
    const coord = new PreprocessingCoordinator(template);
    const ctx = createProcessingContext('test.png', fakeMat, fakeMat, template);

    const result = await coord.process(ctx);

    expect(result.filePath).toBe('test.png');
    expect(result.omrResponse).toEqual({});
  });

  it('calls each preProcessor.process() in sequence', async () => {
    const log: string[] = [];

    const preProc1 = makeAsyncProcessor('Pre1', (ctx) => {
      log.push('Pre1');
      return { metadata: { ...ctx.metadata, pre1: true } };
    });
    const preProc2 = makeAsyncProcessor('Pre2', (ctx) => {
      log.push('Pre2');
      expect(ctx.metadata['pre1']).toBe(true);
      return { metadata: { ...ctx.metadata, pre2: true } };
    });

    const template = makeTemplate();
    template.templateLayout.preProcessors = [preProc1, preProc2];

    const coord = new PreprocessingCoordinator(template);
    const ctx = createProcessingContext('test.png', fakeMat, fakeMat, template);

    const result = await coord.process(ctx);

    expect(log).toEqual(['Pre1', 'Pre2']);
    expect(result.metadata['pre1']).toBe(true);
    expect(result.metadata['pre2']).toBe(true);
  });

  it('mutations by preProcessors are visible in the returned context', async () => {
    const preProc = makeAsyncProcessor('MarkMulti', () => ({
      isMultiMarked: true,
      omrResponse: { Q1: 'A,B' },
    }));

    const template = makeTemplate();
    template.templateLayout.preProcessors = [preProc];

    const coord = new PreprocessingCoordinator(template);
    const ctx = createProcessingContext('test.png', fakeMat, fakeMat, template);
    const result = await coord.process(ctx);

    expect(result.isMultiMarked).toBe(true);
    expect(result.omrResponse['Q1']).toBe('A,B');
  });

  it('preProcessor errors propagate out of process()', async () => {
    const failingPreProc: Processor = {
      getName: () => 'Failing',
      async process(): Promise<ProcessingContext> {
        throw new Error('preProcessor failure');
      },
    };

    const template = makeTemplate();
    template.templateLayout.preProcessors = [failingPreProc];

    const coord = new PreprocessingCoordinator(template);
    const ctx = createProcessingContext('test.png', fakeMat, fakeMat, template);

    await expect(coord.process(ctx)).rejects.toThrow('preProcessor failure');
  });
});

// ---------------------------------------------------------------------------
// 10. End-to-end integration: PreprocessingCoordinator inside Pipeline
// ---------------------------------------------------------------------------

describe('integration — PreprocessingCoordinator used as a Pipeline stage', () => {
  it('coordinator can be added to a pipeline and runs its preProcessors', async () => {
    const log: string[] = [];

    const innerPreProc = makeAsyncProcessor('InnerPreProc', () => {
      log.push('inner');
      return { metadata: { innerRan: true } };
    });

    const template = makeTemplate();
    template.templateLayout.preProcessors = [innerPreProc];

    const pipeline = new ProcessingPipeline(template);
    pipeline.addProcessor(new PreprocessingCoordinator(template));

    pipeline.addProcessor(
      makeProcessor('PostCoord', (ctx) => {
        log.push('post');
        expect(ctx.metadata['innerRan']).toBe(true);
        return {};
      })
    );

    const ctx = await pipeline.processFile('full-flow.png', fakeMat, fakeMat);

    expect(log).toEqual(['inner', 'post']);
    expect(ctx.metadata['innerRan']).toBe(true);
  });

  it('pipeline with coordinator preserves filePath through all stages', async () => {
    const template = makeTemplate();

    const pipeline = new ProcessingPipeline(template);
    pipeline.addProcessor(new PreprocessingCoordinator(template));
    pipeline.addProcessor(makeProcessor('Noop'));

    const ctx = await pipeline.processFile('/scans/page1.png', fakeMat, fakeMat);
    expect(ctx.filePath).toBe('/scans/page1.png');
  });

  it('pipeline processFile creates context with template from constructor', async () => {
    const template = makeTemplate();

    const pipeline = new ProcessingPipeline(template);
    let capturedTemplate: any = null;

    pipeline.addProcessor(
      makeProcessor('CaptureTemplate', (ctx) => {
        capturedTemplate = ctx.template;
        return {};
      })
    );

    await pipeline.processFile('check.png', fakeMat, fakeMat);
    expect(capturedTemplate).toBe(template);
  });

  it('multiple coordinator stages share the same pipeline context', async () => {
    // Two coordinators, each with one preProcessor, chained in the pipeline.
    const template1 = makeTemplate();
    const template2 = makeTemplate();

    template1.templateLayout.preProcessors = [
      makeAsyncProcessor('Stage1Pre', () => ({ metadata: { stage1: true } })),
    ];
    template2.templateLayout.preProcessors = [
      makeAsyncProcessor('Stage2Pre', (ctx) => {
        // stage1 set by first coordinator's preProcessor
        expect(ctx.metadata['stage1']).toBe(true);
        return { metadata: { ...ctx.metadata, stage2: true } };
      }),
    ];

    const pipeline = new ProcessingPipeline(template1);
    pipeline.addProcessor(new PreprocessingCoordinator(template1));
    pipeline.addProcessor(new PreprocessingCoordinator(template2));

    const ctx = await pipeline.processFile('two-stage.png', fakeMat, fakeMat);
    expect(ctx.metadata['stage1']).toBe(true);
    expect(ctx.metadata['stage2']).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 11. Processor spy — verify call counts and argument passing
// ---------------------------------------------------------------------------

describe('integration — processor spy verification', () => {
  it('each processor is called exactly once per processFile invocation', async () => {
    const pipeline = new ProcessingPipeline(makeTemplate());

    const processFn = vi.fn((ctx: ProcessingContext) => ctx);
    const spyProc: Processor = {
      getName: () => 'Spy',
      process: processFn,
    };

    pipeline.addProcessor(spyProc);
    await pipeline.processFile('spy.png', fakeMat, fakeMat);

    expect(processFn).toHaveBeenCalledTimes(1);
  });

  it('processor receives a context whose filePath matches the call argument', async () => {
    const pipeline = new ProcessingPipeline(makeTemplate());
    let receivedFilePath: string | undefined;

    pipeline.addProcessor(
      makeProcessor('Capture', (ctx) => {
        receivedFilePath = ctx.filePath;
        return {};
      })
    );

    await pipeline.processFile('/path/to/exam.png', fakeMat, fakeMat);
    expect(receivedFilePath).toBe('/path/to/exam.png');
  });

  it('removeProcessor prevents the processor from being called', async () => {
    const pipeline = new ProcessingPipeline(makeTemplate());
    const processFn = vi.fn((ctx: ProcessingContext) => ctx);
    const spyProc: Processor = { getName: () => 'Removable', process: processFn };

    pipeline.addProcessor(spyProc);
    pipeline.removeProcessor('Removable');

    await pipeline.processFile('after-remove.png', fakeMat, fakeMat);
    expect(processFn).not.toHaveBeenCalled();
  });
});
