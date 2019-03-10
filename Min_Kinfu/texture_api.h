#pragma once
#include <iostream>
#ifdef TEXTURE_MAP_EXPORTS
#define TEXTURE_MAP_API __declspec(dllexport) 
#else
#define TEXTURE_MAP_API __declspec(dllimport) 
#endif

TEXTURE_MAP_API void create_texMesh(std::string &in_scene, const char* plyFileName,	std::string &tex_mesh_name, int nThreshold, bool bFill);
TEXTURE_MAP_API void createHoleSamplePoints(std::vector<std::vector<std::vector<float>>> &holeSamples);
TEXTURE_MAP_API void getHolePatches(std::vector<std::vector<std::vector<float>>> &patchVertices,
									std::vector<std::vector<std::vector<float>>> &patchNormals,
									std::vector<std::vector<std::vector<int>>> &patchIndices);
TEXTURE_MAP_API void save(std::string fileName);