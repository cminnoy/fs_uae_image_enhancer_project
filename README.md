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

There are four models:
<b>heavy</b> vs <b>light</b>
<b>lores</b> vs <b>non-lores</b>

All models are trained and tested on AMD AI Pro 9700 GPU + AMD Threadripper 1950X.

See here for the matching emulator:  
https://github.com/cminnoy/fs-uae/tree/with_ai_upscaler
