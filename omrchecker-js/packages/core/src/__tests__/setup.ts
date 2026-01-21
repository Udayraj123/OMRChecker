// https://github.com/TechStark/opencv-js/blob/main/test/cv.ts

export async function setupOpenCv() {
  const cvModule = import("@techstark/opencv-js");

  console.log('setupOpenCv');
  // Support both Promise and onRuntimeInitialized callback APIs
  let cv;
  if (cvModule instanceof Promise) {
    // Promise API
    console.log('cvModule', cvModule);
    cv = await cvModule;
  } else {
    // Callback API
    console.log('cvModule callback', cvModule);
    await new Promise<void>((resolve) => {
      (cvModule as any).onRuntimeInitialized = () => {
        console.log('onRuntimeInitialized');
        resolve();
      };
    });
    cv = cvModule;
  }
  console.log('cv', cv);
  global.cv = cv as any;
}