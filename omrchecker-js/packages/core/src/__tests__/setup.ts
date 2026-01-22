// https://github.com/TechStark/opencv-js/blob/main/test/cv.ts

import { initOpenCV } from '../utils/opencv';

console.log('setupOpenCv');
global.cv = await initOpenCV();
console.log('cv initialized');