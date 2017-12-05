#Tests for the GPU

1) To debug the test, first check if you have correctly copied all the needed files from CPU simulator.

2) The corner case would be that the end_time of the GPU does not match that of the CPU. If you failed the test, try to debug from that. 

3) It is pretty tricky/hacky to set up test for spiking mnist. You need to enable the rand() for generating the spiking mnist in CPU. And remember to tweak the LSMEndOfSpeech in network.C for CPU simulator so that the corner case can be gone.
