#include <pcl/point_types.h>
//#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include "texture_api.h"
#include "holeFix.h"

#define RAD1 1.308996938996
#define RAD2 2.3561944901923
#define PAI  3.1415926535898

/*
created at 03/05/2019
*/
bool isHoleBoundary(pcl::PointCloud<pcl::PointXYZ> &cloud, std::vector<Edge> &edges, float fDiameter)
{
	float maxDiameter = 0.0f;
	for(int i=0; i < edges.size(); ++i)
	{
		Eigen::Vector3f v1 = cloud[edges[i].v1].getVector3fMap();
		for(int j=i+1; j < edges.size(); ++j)
		{
			Eigen::Vector3f v2 = cloud[edges[j].v1].getVector3fMap();
			Eigen::Vector3f vec = v2 - v1;
			float len = vec.norm();
			if(maxDiameter < len)
				maxDiameter = len;
			if(fDiameter < maxDiameter)
				return false;
		}
	}
	return true;
}
/*
created at 03/05/2019
*/
void extractHoleBoundaries(pcl::PolygonMesh &mesh, std::vector<Edges> &boundaries)
{
	pcl::PointCloud<pcl::PointXYZ> cloud;
	pcl::fromPCLPointCloud2(mesh.cloud,cloud);
	std::vector<Edge> totalBoundary;
	for(int i=0; i < mesh.polygons.size(); ++i)
	{
		Edge edges[3];
		for(int j=0; j < 3; ++j)
		{
			edges[j].v1 = mesh.polygons.at(i).vertices[j];
			edges[j].v2 = mesh.polygons.at(i).vertices[(j+1)%3];
		}
		for(int j=0; j < 3; ++j)
		{
			int count = 0;
			for(int k=0; k < mesh.polygons.size(); ++k)
			{
				count = 0;
				if(k==i) continue;
				for(int n=0; n < 3; ++n)
				{
					if(edges[j].v1==mesh.polygons.at(k).vertices[n])
						++count;
					if(edges[j].v2==mesh.polygons.at(k).vertices[n])
						++count;
				}
				if(count == 2)
				{
					break;
				}
			}
			if(count < 2)
				totalBoundary.push_back(edges[j]);
		}
	}
	int count0 = 0;
	while(totalBoundary.size() > 3)
	{
		std::vector<Edge> boundary;
		Edge e = totalBoundary.at(0);
		boundary.push_back(e);
		totalBoundary.erase(totalBoundary.begin());
		int count = 1;
		while(count > 0)
		{
			count = 0;
			for(int i=0; i < totalBoundary.size(); ++i)
			{
				Edge e2 = totalBoundary.at(i);
				if(e.v2==e2.v1 || e.v2==e2.v2)
				{
					e = e2;
					boundary.push_back(e2);
					totalBoundary.erase(totalBoundary.begin()+i);
					count = 1;
					break;
				}		
			}
		}
		if(boundary.size() > 2)
		{
			++count0;
			Edge e0 = boundary.at(0);
			Edge e1 = boundary.at(boundary.size()-1);
			if(e0.v1==e1.v2)
			{
				if(isHoleBoundary(cloud, boundary, 0.6f))
					boundaries.push_back(boundary);
			}
		}
	}
}
/*
created at 02/12/2019
*/
void removeIsolatedPieces(pcl::PolygonMesh &mesh)
{
	pcl::PointCloud<pcl::PointXYZ> cloud;
	pcl::fromPCLPointCloud2(mesh.cloud,cloud);
	std::vector<int> fLabels;
	fLabels.resize(mesh.polygons.size());
	for(int i=0; i < mesh.polygons.size(); ++i)
	{
		fLabels[i] = 1;
	}
	std::vector<std::vector<int>> pieces;
	for(int i=0; i < mesh.polygons.size(); ++i)
	{
		if(fLabels[i]==0)
			continue;
		std::vector<int> tmp, piece;
		tmp.push_back(i);
		piece.push_back(i);
		while(tmp.size() > 0)
		{
			int t=tmp.at(0);
			tmp.erase(tmp.begin());
			pcl::Vertices vertices1 = mesh.polygons.at(t);
			for(int j=0; j < 3; ++j)
			{
				for(int k=0; k < mesh.polygons.size(); ++k)
				{
					if(fLabels[k]==0)
						continue;
					pcl::Vertices vertices2 = mesh.polygons.at(k);
					for(int n=0; n < 3; ++n)
					{
						if(vertices1.vertices[j]==vertices2.vertices[n])
						{
							tmp.push_back(k);
							piece.push_back(k);
							fLabels[k] = 0;
							break;
						}
					}
				}
			}
		}
		pieces.push_back(piece);
	}
	std::vector<pcl::Vertices> polygons;
	PCL_INFO("count of pieces:%d",pieces.size());
	for(int i=0; i < pieces.size(); ++i)
	{
		if(pieces.at(i).size() < 800)
			continue;
		for(int j=0; j < pieces.at(i).size(); ++j)
		{
			int t = pieces.at(i).at(j);
			pcl::Vertices vertices = mesh.polygons.at(t);
			polygons.push_back(vertices);
		}
	}
	mesh.polygons.swap(polygons);
}
/*
created at 02/13/2019
*/
void removeIsoPiecesInTexMesh(pcl::TextureMesh &texMesh)
{
	pcl::PointCloud<pcl::PointXYZ> cloud;
	pcl::fromPCLPointCloud2(texMesh.cloud,cloud);
	std::vector<int> fLabels;
	fLabels.resize(texMesh.tex_polygons.size());
	for(int i=0; i < texMesh.tex_polygons.size(); ++i)
	{
		fLabels[i] = 1;
	}
	std::vector<std::vector<int>> pieces;
	for(int i=0; i < texMesh.tex_polygons.size(); ++i)
	{
		if(fLabels[i]==0)
			continue;
		std::vector<int> tmp, piece;
		tmp.push_back(i);
		piece.push_back(i);
		while(tmp.size() > 0)
		{
			int t=tmp.at(0);
			tmp.erase(tmp.begin());
			pcl::Vertices vertices1 = texMesh.tex_polygons.at(t)[0];
			for(int k=0; k < texMesh.tex_polygons.size(); ++k)
			{
				if(fLabels[k]==0)
					continue;
				pcl::Vertices vertices2 = texMesh.tex_polygons.at(k)[0];
				for(int n=0; n < 3; ++n)
				{
					int count = 0;
					for(int j=0; j < 3; ++j)
					{
						if(vertices1.vertices[j]==vertices2.vertices[n])
						{
							count = 1;
							break;
						}
					}
					if(count == 1)
					{
						tmp.push_back(k);
						piece.push_back(k);
						fLabels[k] = 0;
						break;
					}
				}
			}
		}
		pieces.push_back(piece);
	}
	std::vector<std::vector<pcl::Vertices>> polygons;
	for(int i=0; i < pieces.size(); ++i)
	{
		if(pieces.at(i).size() < 800)
			continue;
		for(int j=0; j < pieces.at(i).size(); ++j)
		{
			int t = pieces.at(i).at(j);
			std::vector<pcl::Vertices> vertices = texMesh.tex_polygons.at(t);
			//pcl::Vertices vertices = texMesh.tex_polygons.at(t)[0];
			polygons.push_back(vertices);
		}
	}
	PCL_INFO("count of original faces:%d\n",texMesh.tex_polygons.size());
	PCL_INFO("count of pieces:%d\n",pieces.size());
	PCL_INFO("first piece size:%d\n",pieces.at(0).size());
	PCL_INFO("count of filtered polygons:%d\n",polygons.size());
	texMesh.tex_polygons.swap(polygons);
}
/*
created at 02/20/2019
*/
void removeNonManifoldFaces(pcl::PolygonMesh &mesh)
{
	std::vector<int> indices;
	for(int i=0; i < mesh.polygons.size(); ++i)
	{
		Edge edges[3];
		int counts[3];
		for(int j=0; j < 3; ++j)
		{
			edges[j].v1 = mesh.polygons[i].vertices[j];
			edges[j].v2 = mesh.polygons[i].vertices[(j+1)%3];
			counts[j] = 0;
		}
		for(int j=0; j < mesh.polygons.size(); ++j)
		{
			if(i==j) continue;
			pcl::Vertices vertices;
			vertices.vertices.push_back(mesh.polygons[j].vertices[0]);
			vertices.vertices.push_back(mesh.polygons[j].vertices[1]);
			vertices.vertices.push_back(mesh.polygons[j].vertices[2]);
			for(int k=0; k < 3; ++k)
			{
				int count = 0;
				for(int n=0; n < 3; ++n)
				{
					if(edges[k].v1==vertices.vertices[n])
						++count;
					if(edges[k].v2==vertices.vertices[n])
						++count;
				}
				if(count==2)
					++counts[k];
			}
		}
		bool flg = false;
		int s = 0;
		int k = 0;
		for(int j=0; j < 3; ++j)
		{
			s += counts[j];
			if(counts[j] >= 2)
			{
				indices.push_back(i);
				//k = j;
				//flg = true;
			}
		}
		//if(flg==true && s==counts[k])
			//indices.push_back(i);
	}
	for(int i=indices.size()-1; i > -1; --i)
	{
		mesh.polygons.erase(mesh.polygons.begin()+indices[i]);
	}
}
/*
creatd at 03/06/2019
modified at 03/07/2019
*/
void generateHoleSamples(pcl::PolygonMesh &mesh, Edges &boundary, pcl::PointCloud<pcl::PointXYZ>::Ptr sampleCloud)
{
	PCL_INFO("start generate hole samples\n");
	pcl::PointCloud<pcl::PointXYZ> cloud;
	pcl::fromPCLPointCloud2(mesh.cloud,cloud);
	float len = 0.0f;
	std::vector<Eigen::Vector3f> vts;
	for(int i=0; i < boundary.size(); ++i)
	{
		Eigen::Vector3f v1 = cloud[boundary[i].v1].getVector3fMap();
		Eigen::Vector3f v2 = cloud[boundary[i].v2].getVector3fMap();
		Eigen::Vector3f vec = v2-v1;
		len += vec.norm();
		vts.push_back(v1);
	}
	int n = boundary.size();
	len = len/n;
	for(int i=1; i < n/2-1; ++i)
	{
		Eigen::Vector3f vec = vts[n-i] - vts[i];
		int m = vec.norm()/len;
		vec.normalize();
		pcl::PointXYZ p;
		for(int j=0; j < m; ++j)
		{
			Eigen::Vector3f v = j*len*vec + vts[i];
			p.x = v[0];
			p.y = v[1];
			p.z = v[2];
			sampleCloud->push_back(p);
		}
		p.x = vts[n-i][0];
		p.y = vts[n-i][1];
		p.z = vts[n-i][2];
		sampleCloud->push_back(p);
	}
}
/*
created at 02/28/2019
modified at 03/05/2019
*/
void createHolePatch(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PolygonMesh &triangles)
{
	PCL_INFO("start create holepatch\n");
	// Normal estimation*
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud);
	n.setInputCloud (cloud);
	n.setSearchMethod (tree);
	n.setKSearch (20);//20
	n.compute (*normals);
	PCL_INFO("pass normal estimation\n");
	//* normals should not contain the point normals + surface curvatures

	// Concatenate the XYZ and normal fields*
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
	PCL_INFO("pass concatenate\n");
	//* cloud_with_normals = cloud + normals
	// Create search tree*
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud (cloud_with_normals);

	// Initialize objects
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	// Set the maximum distance between connected points (maximum edge length)
	gp3.setSearchRadius (0.25);//0.025
	 // Set typical values for the parameters
	gp3.setMu (2.5);
	gp3.setMaximumNearestNeighbors (100);
	gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
	gp3.setMinimumAngle(M_PI/18); // 10 degrees
	gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
	gp3.setNormalConsistency(false);

	// Get result
	gp3.setInputCloud (cloud_with_normals);
	gp3.setSearchMethod (tree2);
	gp3.reconstruct (triangles);
	PCL_INFO("face count:%d\n",triangles.polygons.size());
}
/*
created at 03/06/2019
*/
void createHolePatches(pcl::PolygonMesh &mesh, 
					   std::vector<Edges> &boundaries, 
					   std::vector<pcl::PolygonMesh> &newPatches)
{
	pcl::PointCloud<pcl::PointXYZ> cloud;
	pcl::fromPCLPointCloud2(mesh.cloud,cloud);
	PCL_INFO("count of holes:%d\n",boundaries.size());
	int newFaceCount = 0;
	for(int i=0; i < boundaries.size(); ++i)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr sampleCloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PolygonMesh patchMesh;
		PCL_INFO("size of boundary:%d\n", boundaries[i].size());
		if(boundaries[i].size() < 5) continue;
		generateHoleSamples(mesh, boundaries[i], sampleCloud);
		if(sampleCloud->size() < 10) continue;
		PCL_INFO("count of samples:%d\n", sampleCloud->size());
		createHolePatch(sampleCloud, patchMesh);
		newPatches.push_back(patchMesh);
		/*pcl::PointCloud<pcl::PointXYZ> tmpCloud;
		pcl::fromPCLPointCloud2(patchMesh.cloud, tmpCloud);
		int prevCloudSize = cloud.size();
		for(int j=0; j < patchMesh.polygons.size(); ++j)
		{
			for(int k=0; k < 3; ++k)
			{
				patchMesh.polygons[j].vertices[k] += prevCloudSize;
			}
			mesh.polygons.push_back(patchMesh.polygons[j]);
		}
		for(int j=0; j < tmpCloud.size(); ++j)
		{
			cloud.push_back(tmpCloud[j]);
		}
		newFaceCount += patchMesh.polygons.size();*/
	}
	/*pcl::PointCloud<pcl::PointXYZ> newWholeCloud;
	for(int i=0; i < newPatches.size(); ++i)
	{
		int n = newWholeCloud.size();
		for(int j=0; j < newPatches[i].polygons.size(); ++j)
		{
			pcl::Vertices t;
			t.vertices.resize(3);
			t.vertices[0] = newPatches[i].polygons[j].vertices[0] + n;
			t.vertices[1] = newPatches[i].polygons[j].vertices[1] + n;
			t.vertices[2] = newPatches[i].polygons[j].vertices[2] + n;
			newMesh.polygons.push_back(t);
		}
		pcl::PointCloud<pcl::PointXYZ> newCloud;
		pcl::fromPCLPointCloud2(newPatches[i].cloud, newCloud);
		for(int j=0; j < newCloud.size(); ++j)
		{
			newWholeCloud.push_back(newCloud[j]);
		}
	}
	pcl::toPCLPointCloud2(newWholeCloud, newMesh.cloud);
	PCL_INFO("new face count:%d\n",newFaceCount);*/
}
/*
created at 03/08/2019
*/
void mergePatches(pcl::PolygonMesh &mesh, pcl::PolygonMesh &newMesh, std::vector<pcl::PolygonMesh> &patches)
{
	pcl::PointCloud<pcl::PointXYZ> cloud, tmpCloud;
	pcl::fromPCLPointCloud2(mesh.cloud,cloud);
	pcl::fromPCLPointCloud2(mesh.cloud,tmpCloud);
	int cloudSize = cloud.size();
	for(int i=0; i < patches.size(); ++i)
	{
		for(int j=0; j < patches[i].polygons.size(); ++j)
		{
			pcl::Vertices t;
			t.vertices.resize(3);
			t.vertices[0] = patches[i].polygons[j].vertices[0] + cloudSize;
			t.vertices[1] = patches[i].polygons[j].vertices[1] + cloudSize;
			t.vertices[2] = patches[i].polygons[j].vertices[2] + cloudSize;
			mesh.polygons.push_back(t);
		}
		pcl::PointCloud<pcl::PointXYZ> patchCloud;
		pcl::fromPCLPointCloud2(patches[i].cloud,patchCloud);
		cloudSize += patchCloud.size();
		for(int j=0; j < patchCloud.size(); ++j)
		{
			cloud.push_back(patchCloud[j]);
		}
	}
	pcl::toPCLPointCloud2(cloud,mesh.cloud);
	//merge patches
	pcl::PointCloud<pcl::PointXYZ> newWholeCloud;
	cloudSize = 0;
	for(int i=0; i < patches.size(); ++i)
	{
		for(int j=0; j < patches[i].polygons.size(); ++j)
		{
			pcl::Vertices t;
			t.vertices.resize(3);
			t.vertices[0] = patches[i].polygons[j].vertices[0] + cloudSize;
			t.vertices[1] = patches[i].polygons[j].vertices[1] + cloudSize;
			t.vertices[2] = patches[i].polygons[j].vertices[2] + cloudSize;
			newMesh.polygons.push_back(t);
		}
		pcl::PointCloud<pcl::PointXYZ> newCloud;
		pcl::fromPCLPointCloud2(patches[i].cloud, newCloud);
		for(int j=0; j < newCloud.size(); ++j)
		{
			newWholeCloud.push_back(newCloud[j]);
		}
		cloudSize += newCloud.size();
	}
	pcl::toPCLPointCloud2(newWholeCloud, newMesh.cloud);
	int newFaceCount = newMesh.polygons.size();
	int startPos = mesh.polygons.size() - newFaceCount;
	for(int i=startPos; i < mesh.polygons.size(); ++i)
	{
		Eigen::Vector3f v[3];
		v[0] = cloud[mesh.polygons[i].vertices[0]].getVector3fMap();
		v[1] = cloud[mesh.polygons[i].vertices[1]].getVector3fMap();
		v[2] = cloud[mesh.polygons[i].vertices[2]].getVector3fMap();
		for(int j=0; j < 3; ++j)
		{
			for(int k=0; k < tmpCloud.size(); ++k)
			{
				Eigen::Vector3f vt = tmpCloud[k].getVector3fMap();
				if(vt==v[j])
				{
					mesh.polygons[i].vertices[j] = k;
					break;
				}
			}
		}
	}
}
/*
created at 03/05/2019
*/
void refineHole(pcl::PolygonMesh &mesh, Edges &boundary)
{
	pcl::PointCloud<pcl::PointXYZ> cloud;
	pcl::fromPCLPointCloud2(mesh.cloud, cloud);
	//std::vector<pcl::Vertices> & triangles = mesh.polygons;
	std::vector<pcl::Vertices> newFaces;
	for (int i = 0; i < (boundary.size()-1); ++i)
	{
		if(boundary[i].v1 < 0 || boundary[i].v1 > (cloud.size()-1))
			PCL_INFO("boundary[i].v1 is abnormal.");
		if(boundary[i].v2 < 0 || boundary[i].v2 > (cloud.size()-1))
			PCL_INFO("boundary[i].v2 is abnormal.");
		if(boundary[i+1].v1 < 0 || boundary[i+1].v1 > (cloud.size()-1))
			PCL_INFO("boundary[i+1].v1 is abnormal.");
		if(boundary[i+1].v2 < 0 || boundary[i+1].v2 > (cloud.size()-1))
			PCL_INFO("boundary[i+1].v2 is abnormal.");
		float angle = 2*PAI;
		for (int j = 0; j < mesh.polygons.size(); ++j)
		{
			for (int k = 0; k < 3; ++k)
			{
				if (boundary[i].v2 == mesh.polygons[j].vertices[k])
				{
					int vtIds[3];
					for (int n = 0; n < 3; ++n)
					{
						vtIds[n] = mesh.polygons[j].vertices[(k+n)%3];
					}
					Eigen::Vector3f vec1, vec2;
					vec1 = cloud[vtIds[1]].getVector3fMap() - cloud[vtIds[0]].getVector3fMap();//vertices[vtIds[1]] - vertices[vtIds[0]];
					vec2 = cloud[vtIds[2]].getVector3fMap() - cloud[vtIds[0]].getVector3fMap();//vertices[vtIds[2]] - vertices[vtIds[0]];
					vec1.normalize();
					vec2.normalize();
					angle -= std::acosf(vec1.dot(vec2));
					break;
				}
			}
		}
		if (angle < RAD1)
		{
			pcl::Vertices t;
			t.vertices.push_back(boundary[i].v1);
			t.vertices.push_back(boundary[i].v2);
			t.vertices.push_back(boundary[i+1].v2);
			//mesh.polygons.push_back(t);
			newFaces.push_back(t);
			Edge newEdge;
			newEdge.v1 = boundary[i].v1;
			newEdge.v2 = boundary[i+1].v2;
			boundary[i] = newEdge;
			boundary.erase(boundary.begin()+i+1);
			continue;
		}
		if (angle > RAD1 && angle < RAD2)
		{
			PCL_INFO("start new vert\n");
			Eigen::Vector3f vec0 = cloud.points[boundary[i].v1].getVector3fMap() - cloud.points[boundary[i].v2].getVector3fMap();
			Eigen::Vector3f vec1 = cloud.points[boundary[i+1].v2].getVector3fMap() - cloud.points[boundary[i+1].v1].getVector3fMap();
			vec0 = (vec0 + vec1) / 2;
			Eigen::Vector3f newVert = cloud.points[boundary[i].v2].getVector3fMap() + vec0;
			PCL_INFO("finish new vert\n");
			pcl::PointXYZ p;
			p.x = newVert[0];
			p.y = newVert[1];
			p.z = newVert[2];
			PCL_INFO("start adding new vert1\n");
			cloud.push_back(p);
			PCL_INFO("finish adding new vert1\n");
			int newVtId = cloud.size()-1; 
			pcl::Vertices t1;
			t1.vertices.resize(3);
			t1.vertices[0] = boundary[i].v1;
			t1.vertices[1] = boundary[i].v2;
			t1.vertices[2] = newVtId;
			PCL_INFO("start adding new face1\n");
			//mesh.polygons.push_back(t1);
			newFaces.push_back(t1);
			PCL_INFO("finish adding new face1\n");
			pcl::Vertices t2;
			t2.vertices.resize(3);
			t2.vertices[0] = boundary[i+1].v1;
			t2.vertices[1] = boundary[i+1].v2;
			t2.vertices[2] = newVtId;
			PCL_INFO("start adding new face2\n");
			//mesh.polygons.push_back(t2);
			newFaces.push_back(t2);
			PCL_INFO("finish adding new face2\n");
			boundary[i].v2 = newVtId;
			boundary[i+1].v1 = newVtId;
			PCL_INFO("finish adding new one\n");
		}
	}
	PCL_INFO("start adding new cloud\n");
	pcl::toPCLPointCloud2(cloud, mesh.cloud);
	PCL_INFO("finish adding new cloud\n");
	for(int i=0; i < newFaces.size(); ++i)
	{
		mesh.polygons.push_back(newFaces[i]);
	}
	PCL_INFO("finish adding new faces\n");
}
/*
created at 03/05/2019
*/
void refineHoles(pcl::PolygonMesh &mesh, std::vector<Edges> &boundaries)
{
	PCL_INFO("count of boundaries:%d\n",boundaries.size());
	int nLoop = 5;
	while(nLoop > 0)
	{
		for(int i=0; i < boundaries.size(); ++i)
		{
			PCL_INFO("size of boundary:%d\n",boundaries[i].size());
			if(boundaries[i].size() < 5)
				continue;
			refineHole(mesh,boundaries[i]);
			PCL_INFO("boundary no:%d\n",i);
		}
		--nLoop;
	}
	PCL_INFO("refine is finished.");
}
/*
created at 03/05/2019
*/
void removeErrorFaces(pcl::PolygonMesh &mesh)
{
	int nPos = 0;
	pcl::PointCloud<pcl::PointXYZ> cloud;
	pcl::fromPCLPointCloud2(mesh.cloud,cloud);
	for(int i=0; i < mesh.polygons.size(); ++i)
	{
		for(int j=0; j < 3; ++j)
		{
			if(mesh.polygons[i].vertices[j] < 0 || mesh.polygons[i].vertices[j] > (cloud.size()-1))
				mesh.polygons[i].vertices[j] = 0;
		}
	}
	while(nPos < mesh.polygons.size())
	{
		bool flg = false;
		for(int i=0; i < 3; ++i)
		{
			for(int j=0; j < 3; ++j)
			{
				if(i==j) continue;
				if(mesh.polygons[nPos].vertices[i]==mesh.polygons[nPos].vertices[j])
				{
					mesh.polygons.erase(mesh.polygons.begin()+nPos);
					flg = true;
					break;
				}
			}
			if(flg==true)
				break;
		}
		if(flg==false)
			++nPos;
	}
}
/*
created at 03/05/2019
*/
void fillingHoles(pcl::PolygonMesh &mesh, pcl::PolygonMesh &newPatchMesh)
{
	PCL_INFO("start filling holes");
	std::vector<Edges> boundaries;
	removeIsolatedPieces(mesh);
	extractHoleBoundaries(mesh, boundaries);
	PCL_INFO("start refine\n");
	refineHoles(mesh,boundaries);
	PCL_INFO("finish refine\n");
	PCL_INFO("start create hole patches\n");
	//createHolePatches(mesh, boundaries, newPatchMesh);
}
/*
created at 02/09/2019
modified at 02/11/2019
*/
void closeHoles(pcl::PolygonMesh &mesh, pcl::PolygonMesh &newMesh)
{
	std::vector<Edges> boundaries;
	removeIsolatedPieces(mesh);
	extractHoleBoundaries(mesh, boundaries);
	PCL_INFO("start refine\n");
	refineHoles(mesh,boundaries);
	PCL_INFO("finish refine\n");
	std::vector<pcl::PolygonMesh> patches;
	createHolePatches(mesh, boundaries, patches);
	mergePatches(mesh, newMesh, patches);
	removeErrorFaces(mesh);
	return;
	//PCL_INFO("finish create hole patch\n");
	pcl::PointCloud<pcl::PointXYZ> cloud;
	pcl::fromPCLPointCloud2(mesh.cloud,cloud);
	for(int i=0; i < boundaries.size(); ++i)
	{
		int n = cloud.size();
		Eigen::Vector3f vCenter(0.0f,0.0f,0.0f);
		for(int j=0; j < boundaries.at(i).size(); ++j)
		{
			int v1 = boundaries.at(i).at(j).v1;
			int v2 = boundaries.at(i).at(j).v2;
			vCenter += cloud[v1].getVector3fMap();//cloud.at(v1).getVector3fMap();
			pcl::Vertices vertices;
			vertices.vertices.push_back(v1);
			vertices.vertices.push_back(n);
			vertices.vertices.push_back(v2);
			mesh.polygons.push_back(vertices);
		}
		vCenter /= boundaries.at(i).size();
		pcl::PointXYZ xyzCenter;
		xyzCenter.x = vCenter[0];
		xyzCenter.y = vCenter[1];
		xyzCenter.z = vCenter[2];
		cloud.push_back(xyzCenter);
	}
	pcl::toPCLPointCloud2(cloud,mesh.cloud);
}