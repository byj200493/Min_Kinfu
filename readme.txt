Min_Kinfu pipeline 
steps for constructing textured mesh
step1 - computing bounding box
step2 - clipping raw mesh
step3 - refinning raw mesh
step4 - decimating mesh refined
step5 - texturing mesh

clipping depth map for raycasting
from depth pixel to 3d world point
if bounding box dose not contain this 3d world point, depth is set as 0.
the depth map clipped is input into tsdf module of Kinfu.

drawing boundingbox on depth image
checking visibility of boundingbox lines

