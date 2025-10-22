# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**

- (TODO) YOUR NAME HERE
- Tested on: (TODO) **Google Chrome 222.2** on
  Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### Live Demo

[![](img/thumb.png)](http://TODO.github.io/Project4-WebGPU-Forward-Plus-and-Clustered-Deferred)

### Demo Video/GIF

[![](img/video.mp4)](TODO)

### (TODO: Your README)

_DO NOT_ leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

This assignment has a considerable amount of performance analysis compared
to implementation work. Complete the implementation early to leave time!

### Indirect Buffer

Instead of passing parameters directly from the CPU to the GPU, we can store those parameters in a GPU using an indirect buffer. In this case, the GPU can modify how many things to draw without the CPU knowing ahead of time.

In this project, my compute shader decides which gaussians are visible (or culled) and how many splats actually need to be drawn. The shader will write the new number of visible splats into a counter and that count will be copied into the indirect draw buffer. So instead of drawing all of the point, we only draw the visible ones dynamically on the GPU.

This makes the renderer fully GPU-driven.

### Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
