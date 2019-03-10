#include "stdafx.h"
#include <pcl/io/ply_io.h>
#include <pcl/TextureMesh.h>

typedef struct Edge{
	int v1, v2;
}Edge;
typedef std::vector<Edge> Edges;

void extractHoleBoundaries(pcl::PolygonMesh &mesh, std::vector<Edges> &boundaries);
void closeHoles(pcl::PolygonMesh &mesh, pcl::PolygonMesh &newMesh);
void removeIsolatedPieces(pcl::PolygonMesh &mesh);
void removeIsoPiecesInTexMesh(pcl::TextureMesh &texMesh);
void removeNonManifoldFaces(pcl::PolygonMesh &mesh);
void generateHolePatch(std::vector<std::vector<float>> &holeSamples, std::vector<std::vector<float>> &patchVertices,
					   std::vector<std::vector<float>> &patchNormals, std::vector<std::vector<int>> &patchIndices);
void fillingHoles(pcl::PolygonMesh &mesh);