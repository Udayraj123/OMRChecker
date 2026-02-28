import { ProcessingContext, Processor } from '../base';
// Note: ImageUtils would need to be implemented for browser with OpenCV.js

/**
 * Coordinates all image preprocessing steps in sequence.
 * 
 * This is NOT an individual preprocessor. It orchestrates all preprocessors
 * defined in template.templateLayout.preProcessors.
 * 
 * Responsibilities:
 * 1. Creates a copy of the template layout for mutation
 * 2. Resizes images to processing dimensions
 * 3. Runs all preprocessors in sequence (they implement Processor interface)
 * 4. Optionally resizes to output dimensions
 * 
 * Note: Browser version excludes interactive diff display (InteractionUtils)
 */
export class PreprocessingCoordinator implements Processor {
  private template: any; // Template type would come from template.ts
  private tuningConfig: any; // Config type would come from config.ts

  constructor(template: any) {
    this.template = template;
    this.tuningConfig = template.tuningConfig;
  }

  getName(): string {
    return 'Preprocessing';
  }

  async process(context: ProcessingContext): Promise<ProcessingContext> {
    // Get a copy of the template layout for mutation
    const nextTemplateLayout = context.template.templateLayout.getCopyForShifting();

    // Reset the shifts in the copied template layout
    nextTemplateLayout.resetAllShifts();

    let grayImage = context.grayImage;
    let coloredImage = context.coloredImage;

    // Resize to conform to common preprocessor input requirements
    // Note: ImageUtils.resizeToShape would need OpenCV.js implementation
    // grayImage = ImageUtils.resizeToShape(nextTemplateLayout.processingImageShape, grayImage);
    // if (this.tuningConfig.outputs.coloredOutputsEnabled) {
    //   coloredImage = ImageUtils.resizeToShape(nextTemplateLayout.processingImageShape, coloredImage);
    // }

    // Update context for preprocessors
    context.grayImage = grayImage;
    context.coloredImage = coloredImage;
    context.template.templateLayout = nextTemplateLayout;

    // Run preprocessors in sequence using their process() method
    for (const preProcessor of nextTemplateLayout.preProcessors) {
      // Process using unified interface - preprocessors implement process(context)
      context = await preProcessor.process(context);
    }

    // Resize to output requirements if specified
    const outputImageShape = context.template.templateLayout.outputImageShape;
    if (outputImageShape && outputImageShape.length > 0) {
      // context.grayImage = ImageUtils.resizeToShape(outputImageShape, context.grayImage);
      // if (this.tuningConfig.outputs.coloredOutputsEnabled) {
      //   context.coloredImage = ImageUtils.resizeToShape(outputImageShape, context.coloredImage);
      // }
    }

    console.debug(`Completed ${this.getName()} processor`);

    return context;
  }
}
