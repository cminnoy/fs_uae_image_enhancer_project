FS-UAE Amiga Emulator - Image enhancer
======================================

This is an experimental project to add an image upscaler to the FS-UAE Amiga emulator.
The framebuffer in the emulator is 752x576 pixels.
Those pixels are used as follows:
2x2 pixels for one lores pixel
2x1 pixels for one lores interlaced pixel
1x2 pixels for one hires pixel
1x1 pixel  for one hires interlaced pixel

The artifical network upscales images inside the framebuffer.

Known limitations:
- Only trained to upscale the resolution; not enhance the colour palette.
- Very small model to be able to run real-time.
- Only tested on AMD RT 6900 XT under ROCm with ONNX runtime; Ubuntu 24
- The framebuffer is created on CPU side; transfered to GPU memory and upscale; this is very fast (< 1ms); but there is no link between ONNX and the OpenGL render engine yet.
  This implies that the upscaled image needs to be copied back to CPU memory before it can be delivered to OpenGL; this is a major bottleneck.
  
