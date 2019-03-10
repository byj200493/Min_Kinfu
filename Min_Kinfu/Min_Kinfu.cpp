// Min_Kinfu.cpp : Defines the entry point for the console application.
#define _CRT_SECURE_NO_DEPRECATE
#include "stdafx.h"
#include <iostream>
#include <vector>

#include <pcl/console/parse.h>

#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/raycaster.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/gpu/containers/initialization.h>

#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/exceptions.h>
#include <pcl/io/png_io.h>
#include <pcl/io/obj_io.h>//2017/05/08

#include <pcl/common/angles.h>

#include <pcl/TextureMesh.h>
#include <pcl/features/normal_3d.h>

#include <pcl/visualization/point_cloud_color_handlers.h>

#include "camera_pose.h"
#include <direct.h>
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include "Simplify.h"
#include "texture_api.h"
#include <vtkFillHolesFilter.h>
#include "holeFix.h"

#ifdef HAVE_OPENCV  
  #include <opencv2/highgui/highgui.hpp>
  #include <opencv2/imgproc/imgproc.hpp>
//#include "video_recorder.h"
#endif
typedef pcl::ScopeTime ScopeTimeT;

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
namespace pc = pcl::console;

#include "OpenNI.h"

#pragma comment(lib, "pcl_common_release.lib")
#pragma comment(lib, "pcl_gpu_containers_release.lib")
#pragma comment(lib, "pcl_gpu_kinfu_release.lib")
#pragma comment(lib, "pcl_io_ply_release.lib")
#pragma comment(lib, "pcl_io_release.lib")
#pragma comment(lib, "pcl_kdtree_release.lib")
#pragma comment(lib, "pcl_kinfu_app_release.lib")
#pragma comment(lib, "pcl_visualization_release.lib")
#pragma comment(lib, "pcl_surface_release.lib")
#pragma comment(lib, "pcl_search_release.lib")
#pragma comment(lib, "pcl_features_release.lib")

#pragma comment(lib, "vtkexoIIc.lib")
#pragma comment(lib, "vtkftgl.lib")
#pragma comment(lib, "vtkGraphics.lib")
#pragma comment(lib, "vtkImaging.lib")
#pragma comment(lib, "vtkjpeg.lib")
#pragma comment(lib, "vtklibxml2.lib")
#pragma comment(lib, "vtkpng.lib")
#pragma comment(lib, "vtkRendering.lib")
#pragma comment(lib, "vtktiff.lib")
#pragma comment(lib, "vtkViews.lib")
#pragma comment(lib, "vtkWidgets.lib")
#pragma comment(lib, "vtkIO.lib")
#pragma comment(lib, "vtkalglib.lib")
#pragma comment(lib, "vtkCommon.lib")
#pragma comment(lib, "vtkDICOMParser.lib")
#pragma comment(lib, "vtkexpat.lib")
#pragma comment(lib, "vtkfreetype.lib")
#pragma comment(lib, "vtkzlib.lib")
#pragma comment(lib, "vtkInfovis.lib")
#pragma comment(lib, "vtkmetaio.lib")
#pragma comment(lib, "vtkNetCDF.lib")
#pragma comment(lib, "vtkNetCDF_cxx.lib")
#pragma comment(lib, "vtkproj4.lib")
#pragma comment(lib, "vtksqlite.lib")
#pragma comment(lib, "vtksys.lib")

#pragma comment(lib, "opencv_core2413.lib")
#pragma comment(lib, "opencv_highgui2413.lib")
#pragma comment(lib, "opencv_imgproc2413.lib")

#pragma comment(lib, "texture_dll.lib")

Eigen::Affine3f g_camPose;
bool g_bSnap = false;
float g_fx, g_fy, g_cx, g_cy;
//
std::vector<KinfuTracker::PixelRGB> g_source_image_data;
std::vector<unsigned short> g_source_depth_data;
PtrStepSz<const unsigned short> g_depth;
PtrStepSz<const KinfuTracker::PixelRGB> g_rgb24;
bool g_enable_texture_extraction;
int g_snapshot_rate;
int g_screenshot_counter=0;
int g_frame_counter;
//openni2
openni::Device g_device;
openni::VideoStream g_stream_depth, g_stream_color;

namespace pcl
{
  namespace gpu
  {
    void paint3DView (const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f);
    void mergePointNormal (const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output);
  }

  namespace visualization
  {
    //////////////////////////////////////////////////////////////////////////////////////
    /** \brief RGB handler class for colors. Uses the data present in the "rgb" or "rgba"
      * fields from an additional cloud as the color at each point.
      * \author Anatoly Baksheev
      * \ingroup visualization
      */
    template <typename PointT>
    class PointCloudColorHandlerRGBCloud : public PointCloudColorHandler<PointT>
    {
      using PointCloudColorHandler<PointT>::capable_;
      using PointCloudColorHandler<PointT>::cloud_;

      typedef typename PointCloudColorHandler<PointT>::PointCloud::ConstPtr PointCloudConstPtr;
      typedef typename pcl::PointCloud<RGB>::ConstPtr RgbCloudConstPtr;

      public:
        typedef boost::shared_ptr<PointCloudColorHandlerRGBCloud<PointT> > Ptr;
        typedef boost::shared_ptr<const PointCloudColorHandlerRGBCloud<PointT> > ConstPtr;
        
        /** \brief Constructor. */
        PointCloudColorHandlerRGBCloud (const PointCloudConstPtr& cloud, const RgbCloudConstPtr& colors)
          : rgb_ (colors)
        {
          cloud_  = cloud;
          capable_ = true;
        }
              
        /** \brief Obtain the actual color for the input dataset as vtk scalars.
          * \param[out] scalars the output scalars containing the color for the dataset
          * \return true if the operation was successful (the handler is capable and 
          * the input cloud was given as a valid pointer), false otherwise
          */
        virtual bool
        getColor (vtkSmartPointer<vtkDataArray> &scalars) const
        {
          if (!capable_ || !cloud_)
            return (false);
         
          if (!scalars)
            scalars = vtkSmartPointer<vtkUnsignedCharArray>::New ();
          scalars->SetNumberOfComponents (3);
            
          vtkIdType nr_points = vtkIdType (cloud_->points.size ());
          reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->SetNumberOfTuples (nr_points);
          unsigned char* colors = reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->GetPointer (0);
            
          // Color every point
          if (nr_points != int (rgb_->points.size ()))
            std::fill (colors, colors + nr_points * 3, static_cast<unsigned char> (0xFF));
          else
            for (vtkIdType cp = 0; cp < nr_points; ++cp)
            {
              int idx = cp * 3;
              colors[idx + 0] = rgb_->points[cp].r;
              colors[idx + 1] = rgb_->points[cp].g;
              colors[idx + 2] = rgb_->points[cp].b;
            }
          return (true);
        }

      private:
        virtual std::string 
        getFieldName () const { return ("additional rgb"); }
        virtual std::string 
        getName () const { return ("PointCloudColorHandlerRGBCloud"); }
        
        RgbCloudConstPtr rgb_;
    };
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SampledScopeTime : public StopWatch
{          
  enum { EACH = 33 };
  SampledScopeTime(int& time_ms) : time_ms_(time_ms) {}
  ~SampledScopeTime()
  {
    static int i_ = 0;
    static boost::posix_time::ptime starttime_ = boost::posix_time::microsec_clock::local_time();
    time_ms_ += getTime ();
    if (i_ % EACH == 0 && i_)
    {
      boost::posix_time::ptime endtime_ = boost::posix_time::microsec_clock::local_time();
      cout << "Average frame time = " << time_ms_ / EACH << "ms ( " << 1000.f * EACH / time_ms_ << "fps )"
           << "( real: " << 1000.f * EACH / (endtime_-starttime_).total_milliseconds() << "fps )"  << endl;
      time_ms_ = 0;
      starttime_ = endtime_;
    }
    ++i_;
  }
private:    
  int& time_ms_;
};
 /*
  created at 01/18/2019
  */
  bool isInBox(Eigen::Vector3f &pt, Eigen::Vector3f *vCenters, Eigen::Vector3f *vNormals)
  {
	  for(int i=0; i<6; ++i)
	  {
		  Eigen::Vector3f vec3 = vCenters[i] - pt;
		  if(vec3.dot(vNormals[i]) < 0.0f)
			  return false;
	  }
	  return true;
  }
void
setViewerPose (visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}

void
setViewerPose (visualization::PCLVisualizer::Ptr viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
  viewer->setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}
/*
created at 01/28/2019
*/
void setOverViewCamPoses(visualization::PCLVisualizer::Ptr viewer, const Eigen::Affine3f& cur_pose)
{
	Eigen::Vector3f pos_vector = cur_pose * Eigen::Vector3f (0, 0, 0);
	Eigen::Vector3f look_at_vector = 1.5f*cur_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
	Eigen::Vector3f up_vector = cur_pose.rotation () * Eigen::Vector3f (0, -1, 0);
	Eigen::Vector3f vForward = cur_pose.rotation () * Eigen::Vector3f (0, 0, 1);
	pos_vector = look_at_vector + 3.0*up_vector;
	up_vector = up_vector.cross(vForward);
	viewer->setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}
/*
created at 02/07/2019
*/
void fillingHolesInMesh(PolygonMesh &in_mesh, PolygonMesh &out_mesh)
{
	/*MatrixXd originalV;
	originalV.resize(10,3);
	for(int row = 0; row < 10; ++row)
	{
		for(int col = 0; col < 3; ++col)
		{
			//originalV.
		}
	}*/
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Eigen::Affine3f 
getViewerPose (visualization::PCLVisualizer& viewer)
{
  Eigen::Affine3f pose = viewer.getViewerPose();
  Eigen::Matrix3f rotation = pose.linear();

  Matrix3f axis_reorder;  
  axis_reorder << 0,  0,  1,
                 -1,  0,  0,
                  0, -1,  0;

  rotation = rotation * axis_reorder;
  pose.linear() = rotation;
  return pose;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
writePolygonMeshFile (int format, const pcl::PolygonMesh& mesh, string& meshFileName);
void filterMesh(PolygonMesh &input_mesh);
void filterMesh(PolygonMesh &input_mesh, PolygonMesh &out_mesh);
void refiningMesh(PolygonMesh &in_mesh, PolygonMesh &out_mesh);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const DeviceArray<PointXYZ>& triangles)
{ 
  if (triangles.empty())
      return boost::shared_ptr<pcl::PolygonMesh>();

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width  = (int)triangles.size();
  cloud.height = 1;
  triangles.download(cloud.points);

  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr( new pcl::PolygonMesh() ); 
  pcl::toPCLPointCloud2(cloud, mesh_ptr->cloud);
      
  mesh_ptr->polygons.resize (triangles.size() / 3);
  for (size_t i = 0; i < mesh_ptr->polygons.size (); ++i)
  {
    pcl::Vertices v;
    v.vertices.push_back(i*3+0);
    v.vertices.push_back(i*3+2);
    v.vertices.push_back(i*3+1);              
    mesh_ptr->polygons[i] = v;
  }    
  return mesh_ptr;
}

struct CurrentFrameCloudView
{
  CurrentFrameCloudView() : cloud_device_ (480, 640), cloud_viewer_ ("Frame Cloud Viewer")
  {
    cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);

    cloud_viewer_.setBackgroundColor (0, 0, 0.15);
    cloud_viewer_.setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE, 1);
    cloud_viewer_.addCoordinateSystem (1.0, "global");
    cloud_viewer_.initCameraParameters ();
    cloud_viewer_.setPosition (0, 500);
    cloud_viewer_.setSize (640, 480);
    cloud_viewer_.setCameraClipDistances (0.01, 10.01);
  }

  void
  show (const KinfuTracker& kinfu)
  {
    kinfu.getLastFrameCloud (cloud_device_);

    int c;
    cloud_device_.download (cloud_ptr_->points, c);
    cloud_ptr_->width = cloud_device_.cols ();
    cloud_ptr_->height = cloud_device_.rows ();
    cloud_ptr_->is_dense = false;

    cloud_viewer_.removeAllPointClouds ();
    cloud_viewer_.addPointCloud<PointXYZ>(cloud_ptr_);
    cloud_viewer_.spinOnce ();
  }

  void
  setViewerPose (const Eigen::Affine3f& viewer_pose) {
    ::setViewerPose (cloud_viewer_, viewer_pose);
  }

  PointCloud<PointXYZ>::Ptr cloud_ptr_;
  DeviceArray2D<PointXYZ> cloud_device_;
  visualization::PCLVisualizer cloud_viewer_;
};
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ImageView
{
  ImageView(int viz) : viz_(viz)//);, paint_image_ (false), accumulate_views_ (false)
  {
    if (viz_)
    {
        viewerScene_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);
        viewerScene_->setWindowTitle ("View3D from ray tracing");
        viewerScene_->setPosition (0, 500);
		viewerScene_->setSize(0,0);
		raycaster_ptr_ = RayCaster::Ptr(new RayCaster);
    }
	loop = 0;
  }
  /*
  02/04/2019
  */
  /*void drawBoundingBox(cv::Mat &lineMat)
  {
	  Eigen::Affine3f pose = kinfu_.getCameraPose();
	  Eigen::Affine3f inverse = pose.inverse();
	  Eigen::Vector2f pixel_coords[24];
	  Eigen::Vector3f inv_vertices[8];
	  int width = lineMat.cols;
	  int height = lineMat.rows;
	  _fx = 589.948608;
	  _fy = 591.047363;
	  _px = 337.480164;
	  _py = 253.006744;
	  int c=0;
	  for(int i=0; i < 8; ++i)
	  {
		 pcl::transformPoint(_box_vertices[i],inv_vertices[i],inverse);
	  }
	  float k=0.0f;
	  for(int i=0; i < 12; ++i)
	  {
		  Eigen::Vector3f pt1 = inv_vertices[_box_edges[2*i]];
		  Eigen::Vector3f pt2 = inv_vertices[_box_edges[2*i+1]];
		  pixel_coords[2*i][0] = (pt1[0] * _fx / pt1[2]) + _px;
		  pixel_coords[2*i][1] = (pt1[1] * _fy / pt1[2]) + _py;
		  pixel_coords[2*i+1][0] = (pt2[0] * _fx / pt2[2]) + _px;
		  pixel_coords[2*i+1][1] = (pt2[1] * _fy / pt2[2]) + _py;
	  }
	  for(int i=0; i < 12; ++i)
	  {
		  cv::Point p1, p2;
		  p1.x = pixel_coords[2*i][0];
		  p1.y = pixel_coords[2*i][1];
		  p2.x = pixel_coords[2*i+1][0];
		  p2.y = pixel_coords[2*i+1][1];
		  cv::line(lineMat,p1,p2,cvScalar(1),1,8,0);
	  }
  }*/
  /*
  created at 02/02/2019
  */
  void showScene(KinfuTracker& kinfu, Eigen::Vector3f* vFaceCenters, Eigen::Vector3f* vBoxNormals, Eigen::Affine3f* pose_ptr,
				 cv::Mat &lineMat, int fx, int fy, int cx, int cy)
  {
	if (pose_ptr)
    {
	   raycaster_ptr_->run(kinfu.volume(), *pose_ptr);
       raycaster_ptr_->generateSceneView(view_device_);
	   int cols;
	   view_device_.download (view_host_, cols);
	   pcl::gpu::RayCaster::Depth depthDevice;
	   depthDevice.create(480,640);
	   raycaster_ptr_->generateDepthImage(depthDevice);
	   std::vector<unsigned short> depthMap;
	   int s=2;
	   depthDevice.download(depthMap,s);
	   for(int iy=0; iy < 480; ++iy)
	   {
		   for(int ix=0; ix < 640; ++ix)
		   {
			   unsigned int l = ix + iy*view_device_.cols();
			   Eigen::Vector3f v1, v2;
			   float d = (float)depthMap.at(l)/1000;
			   if(d == 0)
			   {
				 continue;
			   }
			   v1[0] = (ix-cx)*d/fx;
			   v1[1] = (iy-cy)*d/fy;
			   v1[2] = d;
			   pcl::transformPoint(v1, v2, *pose_ptr);
			   if(isInBox(v2,vFaceCenters,vBoxNormals)==false)
			   {
				  view_host_[l].b = 0;
				  view_host_[l].g = 0;
				  view_host_[l].r = 0;
			   }
			   if(lineMat.at<unsigned char>(iy,ix)==1)
			   {
				   unsigned int l = ix + iy*view_device_.cols();
				   view_host_[l].b = 0;
				   view_host_[l].g = 255;
				   view_host_[l].r = 0;
			   }
		   }
	   }
	   viewerScene_->showRGBImage (reinterpret_cast<unsigned char*> (&view_host_[0]), view_device_.cols (), view_device_.rows ());
    }
	if(loop > 26)
	{
	  viewerScene_->spinOnce(false);
	  loop = 0;
	}
	++loop;
  }

  void
  showScene (KinfuTracker& kinfu, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24,
			 cv::Mat &lineMat, Eigen::Affine3f* pose_ptr = 0)
  {
	if (pose_ptr)
    {
       raycaster_ptr_->run(kinfu.volume(), *pose_ptr);
       raycaster_ptr_->generateSceneView(view_device_);
    }
    else
     kinfu.getImage (view_device_);
	 int cols;
     view_device_.download (view_host_, cols);
     if (viz_)
	 {
		 unsigned char r, g, b;
		 r = 0; g = 255; b = 0;
		 for(int y=0; y < view_device_.rows(); ++y)
		 {
			 for(int x=0; x < view_device_.cols(); ++x)
			 {
				 if(lineMat.at<unsigned char>(y,x)==1)
				 {
					unsigned int l = x + y*view_device_.cols();
					view_host_[l].b = b;
					view_host_[l].g = g;
					view_host_[l].r = r;
				 }
			 }
		 }
		 viewerScene_->showRGBImage (reinterpret_cast<unsigned char*> (&view_host_[0]), view_device_.cols (), view_device_.rows ());
	 }
	 if(loop > 26)
	 {
		viewerScene_->spinOnce(false);
		loop = 0;
	 }
	 ++loop;
#ifdef HAVE_OPENCV
    if (accumulate_views_)
    {
      views_.push_back (cv::Mat ());
      cv::cvtColor (cv::Mat (480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back (), CV_RGB2GRAY);
      //cv::copy(cv::Mat(480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back());
    }
#endif
  }
  int loop;
  int viz_;
  visualization::ImageViewer::Ptr viewerScene_;
  KinfuTracker::View view_device_;
  vector<KinfuTracker::PixelRGB> view_host_;
  RayCaster::Ptr raycaster_ptr_;
#ifdef HAVE_OPENCV
  vector<cv::Mat> views_;
#endif
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SceneCloudView
{
  enum { GPU_Connected6 = 0, CPU_Connected6 = 1, CPU_Connected26 = 2 };

  SceneCloudView(int viz) : viz_(viz), extraction_mode_ (GPU_Connected6), compute_normals_ (false), valid_combined_ (false), cube_added_(false)
  {
    cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
    normals_ptr_ = PointCloud<Normal>::Ptr (new PointCloud<Normal>);
    combined_ptr_ = PointCloud<PointNormal>::Ptr (new PointCloud<PointNormal>);
    point_colors_ptr_ = PointCloud<RGB>::Ptr (new PointCloud<RGB>);
	_cloud_clipped_ptr = PointCloud<PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
    if (viz_)
    {
        cloud_viewer_ = pcl::visualization::PCLVisualizer::Ptr( new pcl::visualization::PCLVisualizer("Scene Cloud Viewer") );
		cloud_viewer_->setBackgroundColor (0, 0, 0);
        cloud_viewer_->addCoordinateSystem (1.0, "global");
        cloud_viewer_->initCameraParameters ();
        cloud_viewer_->setPosition (0, 0);
        cloud_viewer_->setSize (640, 480);
		cloud_viewer_->setCameraClipDistances (0.01, 10.01);
		cloud_viewer_->addText ("H: print help", 2, 15, 20, 34, 135, 246);

		/*pose_viewer_ = pcl::visualization::PCLVisualizer::Ptr( new pcl::visualization::PCLVisualizer("camera pose Viewer") );
		pose_viewer_->setBackgroundColor (0, 0, 0);
        pose_viewer_->addCoordinateSystem (1.0, "global1");
        pose_viewer_->initCameraParameters ();
        pose_viewer_->setPosition (640, 0);
        pose_viewer_->setSize (640, 480);
		pose_viewer_->setCameraClipDistances (0.01, 10.01);
		pose_viewer_->addText ("H: print help", 2, 15, 20, 34, 135, 246);*/
    }
  }
  /*
  created at 01/29/2019
  */
  void drawBoundingBox(Eigen::Vector3f *box_vertices, visualization::PCLVisualizer::Ptr viewer, int *indices)
  {
	  for(int i=0; i < 12; ++i)
	  {
		pcl::PointXYZ p1, p2;
		p1.x = box_vertices[indices[2*i]][0];
		p1.y = box_vertices[indices[2*i]][1];
		p1.z = box_vertices[indices[2*i]][2];
		p2.x = box_vertices[indices[2*i+1]][0];
		p2.y = box_vertices[indices[2*i+1]][1];
		p2.z = box_vertices[indices[2*i+1]][2];
		string line = boost::lexical_cast<std::string> (i);
		viewer->addLine<pcl::PointXYZ>(p1, p2, 0, 255, 0, line);
	  }
  }
  /*
  created at 01/29/2018
  */
  void drawCameraPoses(KinfuTracker& kinfu, std::vector<Eigen::Affine3f> &camPoses, 
					   Eigen::Vector3f *box_vertices, int *indices, 
					   pcl::PointCloud<pcl::PointXYZRGB>::Ptr camPoints)
  {
	 Eigen::Affine3f viewPose = kinfu.getCameraPose();
	 setOverViewCamPoses(pose_viewer_, viewPose);
	 pose_viewer_->removeAllPointClouds ();
	 pose_viewer_->removeAllShapes();
	 if(pose_viewer_->addPointCloud<PointXYZ>(_cloud_clipped_ptr,"cloud1")==true)
			  cout << "adding point cloud is finished.\n" << endl;
		  else
			  cout << "adding point cloud is failed.\n" << endl;
	 if(pose_viewer_->addPointCloud<PointXYZRGB>(camPoints,"cloud2")==true)
			  cout << "adding cam point cloud is finished.\n" << endl;
		  else
			  cout << "adding cam point cloud is failed.\n" << endl;
	 for(int i=0; i < camPoses.size(); ++i)
	 {
		Eigen::Vector3f v = camPoses[i] * Eigen::Vector3f (0, 0, 0);
		Eigen::Vector3f vDir = 0.1f*camPoses[i].rotation () * Eigen::Vector3f (0, 0, 1);
		pcl::ModelCoefficients coeffs;
		coeffs.values.resize(7);
		coeffs.values[0] = v[0];
		coeffs.values[1] = v[1];
		coeffs.values[2] = v[2];
		coeffs.values[3] = vDir[0];
		coeffs.values[4] = vDir[1];
		coeffs.values[5] = vDir[2];
		coeffs.values[6] = 30;
		string cone = "cone" + boost::lexical_cast<std::string> (i);
		pose_viewer_->addCone(coeffs, cone);
	 }
	 drawBoundingBox(box_vertices, pose_viewer_, indices);
	 pose_viewer_->spinOnce(false);
  }
  /*
  modified at 01/19/2019
  modified at 02/02/2019 added text "pause"
  */
  void
  show (KinfuTracker& kinfu, bool integrate_colors, Eigen::Vector3f *centers, 
       Eigen::Vector3f *normals, Eigen::Vector3f *box_vertices, int *indices,
	   bool bOverView, std::vector<Eigen::Affine3f> &camPoses, bool bPaused)
  {
	viewer_pose_ = kinfu.getCameraPose();
	/*if(bOverView)
	{
		setOverViewCamPoses(cloud_viewer_, viewer_pose_);
	}
	else*/
	setViewerPose (cloud_viewer_, viewer_pose_);
    ScopeTimeT time ("PointCloud Extraction");
    cout << "\nGetting cloud... " << flush;

    valid_combined_ = false;

    if (extraction_mode_ != GPU_Connected6)     // So use CPU
    {
      kinfu.volume().fetchCloudHost (*cloud_ptr_, extraction_mode_ == CPU_Connected26);
    }
    else
    {
      DeviceArray<PointXYZ> extracted = kinfu.volume().fetchCloud (cloud_buffer_device_);             

      if (compute_normals_)
      {
        kinfu.volume().fetchNormals (extracted, normals_device_);
        pcl::gpu::mergePointNormal (extracted, normals_device_, combined_device_);
        combined_device_.download (combined_ptr_->points);
        combined_ptr_->width = (int)combined_ptr_->points.size ();
        combined_ptr_->height = 1;

        valid_combined_ = true;
      }
      else
      {
        extracted.download (cloud_ptr_->points);
        cloud_ptr_->width = (int)cloud_ptr_->points.size ();
        cloud_ptr_->height = 1;
      }

      if (integrate_colors)
      {
        kinfu.colorVolume().fetchColors(extracted, point_colors_device_);
        point_colors_device_.download(point_colors_ptr_->points);
        point_colors_ptr_->width = (int)point_colors_ptr_->points.size ();
        point_colors_ptr_->height = 1;
	  }
      else
        point_colors_ptr_->points.clear();
    }
    size_t points_size = valid_combined_ ? combined_ptr_->points.size () : cloud_ptr_->points.size ();
    cout << "Done.  Cloud size: " << points_size / 1000 << "K" << endl;

    if (viz_)
    {
        cloud_viewer_->removeAllPointClouds ();
		cloud_viewer_->removeAllShapes();
		if (valid_combined_)
        {
		  cout << "\n" << "valid_combined vis\n" << endl;
          visualization::PointCloudColorHandlerRGBCloud<PointNormal> rgb(combined_ptr_, point_colors_ptr_);
          cloud_viewer_->addPointCloud<PointNormal> (combined_ptr_, rgb, "Cloud");
          cloud_viewer_->addPointCloudNormals<PointNormal>(combined_ptr_, 50);
        }
        else
        {
		  cout << "\n" << "cloud vis\n" << endl;
          visualization::PointCloudColorHandlerRGBCloud<PointXYZ> rgb(cloud_ptr_, point_colors_ptr_);
          /*_cloud_clipped_ptr->clear();
		  for(int i=0; i < cloud_ptr_->size(); ++i)
		  {
			 pcl::PointXYZ p = cloud_ptr_->points[i];
			 Eigen::Vector3f v = p.getVector3fMap();
			 if(isInBox(v, centers, normals)==true)
			 	 _cloud_clipped_ptr->points.push_back(p);
		  }
		  if(cloud_viewer_->addPointCloud<PointXYZ>(_cloud_clipped_ptr)==true)
			  cout << "adding point cloud is finished.\n" << endl;
		  else
			  cout << "adding point cloud is failed.\n" << endl;
		  drawBoundingBox(box_vertices, cloud_viewer_, indices);*/
		  cloud_viewer_->addPointCloud<PointXYZ>(cloud_ptr_);
		}
		//drawCameraPoses(kinfu, camPoses);
		if(bPaused==true)
			cloud_viewer_->addText("pause!",0,450,22,1.0,1.0,0.0,"snap");
		cloud_viewer_->spinOnce(false);
    }
  }

  void
  toggleCube(const Eigen::Vector3f& size)
  {
      if (!viz_)
          return;

      if (cube_added_)
          cloud_viewer_->removeShape("cube");
      else
        cloud_viewer_->addCube(size*0.5, Eigen::Quaternionf::Identity(), size(0), size(1), size(2));

      cube_added_ = !cube_added_;
  }

  void
  toggleExtractionMode ()
  {
    extraction_mode_ = (extraction_mode_ + 1) % 3;

    switch (extraction_mode_)
    {
    case 0: cout << "Cloud exctraction mode: GPU, Connected-6" << endl; break;
    case 1: cout << "Cloud exctraction mode: CPU, Connected-6    (requires a lot of memory)" << endl; break;
    case 2: cout << "Cloud exctraction mode: CPU, Connected-26   (requires a lot of memory)" << endl; break;
    }
    ;
  }

  void
  toggleNormals ()
  {
    compute_normals_ = !compute_normals_;
    cout << "Compute normals: " << (compute_normals_ ? "On" : "Off") << endl;
  }

  void
  clearClouds (bool print_message = false)
  {
    if (!viz_)
        return;

    cloud_viewer_->removeAllPointClouds ();
    cloud_ptr_->points.clear ();
    normals_ptr_->points.clear ();    
    if (print_message)
      cout << "Clouds/Meshes were cleared" << endl;
  }

  void
  showMesh(KinfuTracker& kinfu, bool /*integrate_colors*/)
  {
    if (!viz_)
       return;

    ScopeTimeT time ("Mesh Extraction");
    cout << "\nGetting mesh... " << flush;

    if (!marching_cubes_)
      marching_cubes_ = MarchingCubes::Ptr( new MarchingCubes() );

    DeviceArray<PointXYZ> triangles_device = marching_cubes_->run(kinfu.volume(), triangles_buffer_device_);    
    mesh_ptr_ = convertToMesh(triangles_device);
    
    cloud_viewer_->removeAllPointClouds ();
    if (mesh_ptr_)
      cloud_viewer_->addPolygonMesh(*mesh_ptr_);
    cout << "Done.  Triangles number: " << triangles_device.size() / MarchingCubes::POINTS_PER_TRIANGLE / 1000 << "K" << endl;
  }
  /*
  created at 01/30/2019
  */
  void createMeshViewer(Eigen::Affine3f &camPose)
  {
	  _mesh_viewer = pcl::visualization::PCLVisualizer::Ptr( new pcl::visualization::PCLVisualizer("Scene mesh Viewer") );
	  _mesh_viewer->setBackgroundColor (0, 0, 0);
      _mesh_viewer->addCoordinateSystem (1.0, "global");
      _mesh_viewer->initCameraParameters ();
      _mesh_viewer->setPosition (0, 0);
      _mesh_viewer->setSize (640, 480);
	  _mesh_viewer->setCameraClipDistances (0.01, 10.01);
	  setOverViewCamPoses(_mesh_viewer, camPose);
  }
  /*
  created at 01/30/2019
  */  
  void showTexMesh(std::vector<Eigen::Affine3f> &camPoses, PointCloud<PointXYZRGB>::Ptr pathPointsPtr, PolygonMesh &mesh)
  {
	  _mesh_viewer->removeAllPointClouds();
	  _mesh_viewer->removeAllShapes();
	  _mesh_viewer->addPolygonMesh(mesh);
	  //_mesh_viewer->addTextureMesh(texMesh,"texture");
	  _mesh_viewer->addPointCloud(pathPointsPtr,"poseCloud");
	  for(int i=0; i < camPoses.size(); ++i)
	  {
		Eigen::Vector3f v = camPoses[i] * Eigen::Vector3f (0, 0, 0);
		Eigen::Vector3f vDir = 0.1f*camPoses[i].rotation () * Eigen::Vector3f (0, 0, 1);
		pcl::ModelCoefficients coeffs;
		coeffs.values.resize(7);
		coeffs.values[0] = v[0];
		coeffs.values[1] = v[1];
		coeffs.values[2] = v[2];
		coeffs.values[3] = vDir[0];
		coeffs.values[4] = vDir[1];
		coeffs.values[5] = vDir[2];
		coeffs.values[6] = 30;
		string cone = "cone_mesh_viewr" + boost::lexical_cast<std::string> (i);
		_mesh_viewer->addCone(coeffs, cone);
	  }
	  _mesh_viewer->spinOnce(false);
  }

  int viz_;
  int extraction_mode_;
  bool compute_normals_;
  bool valid_combined_;
  bool cube_added_;

  Eigen::Affine3f viewer_pose_;
  std::vector<Eigen::Affine3f> _camPoses;
  visualization::PCLVisualizer::Ptr cloud_viewer_;
  visualization::PCLVisualizer::Ptr pose_viewer_;//added at 01/28/2019
  pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud_clipped_ptr;//added at 01/28/2019
  visualization::PCLVisualizer::Ptr _mesh_viewer;//added at 01/30/2019
  PointCloud<PointXYZ>::Ptr cloud_ptr_;
  PointCloud<Normal>::Ptr normals_ptr_;

  DeviceArray<PointXYZ> cloud_buffer_device_;
  DeviceArray<Normal> normals_device_;

  PointCloud<PointNormal>::Ptr combined_ptr_;
  DeviceArray<PointNormal> combined_device_;  

  DeviceArray<RGB> point_colors_device_; 
  PointCloud<RGB>::Ptr point_colors_ptr_;

  MarchingCubes::Ptr marching_cubes_;
  DeviceArray<PointXYZ> triangles_buffer_device_;

  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr_;
};
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct KinFuApp
{
   enum { PCD_BIN = 1, PCD_ASCII = 2, PLY = 3, MESH_PLY = 7, MESH_VTK = 8 };
   KinFuApp(float vsz, int icp, int viz, boost::shared_ptr<CameraPoseProcessor> pose_processor=boost::shared_ptr<CameraPoseProcessor> ()) : exit_ (false), scan_ (false), scan_mesh_(false), scan_volume_ (false), independent_camera_ (false),
      registration_ (false), integrate_colors_ (false), focal_length_(-1.f), scene_cloud_view_(viz), image_view_(viz), time_ms_(0), icp_(icp), viz_(viz), pose_processor_ (pose_processor)
   {    
    //Init Kinfu Tracker
	Eigen::Vector3f volume_size = Vector3f::Constant (vsz/*meters*/);    
	kinfu_.volume().setSize (volume_size);
	Eigen::Matrix3f R = Eigen::Matrix3f::Identity ();   // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());
    Eigen::Vector3f t = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

    Eigen::Affine3f pose = Eigen::Translation3f (t) * Eigen::AngleAxisf (R);

    kinfu_.setInitalCameraPose (pose);
    kinfu_.volume().setTsdfTruncDist (0.030f/*meters*/);    
    kinfu_.setIcpCorespFilteringParams (0.1f/*meters*/, sin ( pcl::deg2rad(20.f) ));
    //kinfu_.setDepthTruncationForICP(5.f/*meters*/);
    kinfu_.setCameraMovementThreshold(0.001f);

    if (!icp)
      kinfu_.disableIcp();

    //Init KinfuApp            
    if (viz_)
    {
		scene_cloud_view_.cloud_viewer_->registerKeyboardCallback (keyboard_callback, (void*)this);
        //image_view_.viewerScene_->registerKeyboardCallback (keyboard_callback, (void*)this);
		//scene_cloud_view_.toggleCube(volume_size);
    }
	limit_count_ = 1200;
	max_depth_ = 3.0f;
	_bBoxMove = true;
	_view_count = 0;
	_view_limit_count = 5;
	_bOverView = false;
	_camPoints_ptr = PointCloud<PointXYZRGB>::Ptr (new PointCloud<PointXYZRGB>);
	_bLoop = true;
  }

  ~KinFuApp()
  {
    //if (evaluation_ptr_)
      //evaluation_ptr_->saveAllPoses(kinfu_);
  }

   void
  initCurrentFrameView ()
  {
    current_frame_cloud_view_ = boost::shared_ptr<CurrentFrameCloudView>(new CurrentFrameCloudView ());
    current_frame_cloud_view_->cloud_viewer_.registerKeyboardCallback (keyboard_callback, (void*)this);
    current_frame_cloud_view_->setViewerPose (kinfu_.getCameraPose ());
  }
  void
  initRegistration ()
  {        
    //registration_ = capture_.providesCallback<pcl::ONIGrabber::sig_cb_openni_image_depth_image> ();
    cout << "Registration mode: " << (registration_ ? "On" : "Off (not supported by source)") << endl;
    if (registration_)
      kinfu_.setDepthIntrinsics(KINFU_DEFAULT_RGB_FOCAL_X, KINFU_DEFAULT_RGB_FOCAL_Y);
  }
  
  void
  setDepthIntrinsics(std::vector<float> depth_intrinsics)
  {
    float fx = depth_intrinsics[0];
    float fy = depth_intrinsics[1];
    
    if (depth_intrinsics.size() == 4)
    {
        float cx = depth_intrinsics[2];
        float cy = depth_intrinsics[3];
        kinfu_.setDepthIntrinsics(fx, fy, cx, cy);
        cout << "Depth intrinsics changed to fx="<< fx << " fy=" << fy << " cx=" << cx << " cy=" << cy << endl;
    }
    else {
        kinfu_.setDepthIntrinsics(fx, fy);
        cout << "Depth intrinsics changed to fx="<< fx << " fy=" << fy << endl;
    }
	_fx = fx;
	_fy = fy;
	_px = depth_intrinsics[2];
	_py = depth_intrinsics[3];
  }

  void 
  toggleColorIntegration()
  {
    if(registration_)
    {
      const int max_color_integration_weight = 2;
      kinfu_.initColorIntegration(max_color_integration_weight);
      integrate_colors_ = true;      
    }    
    cout << "Color integration: " << (integrate_colors_ ? "On" : "Off ( requires registration mode )") << endl;
  }
  
  void 
  enableTruncationScaling()
  {
    kinfu_.volume().setTsdfTruncDist (kinfu_.volume().getSize()(0) / 100.0f);
  }

  void
  toggleIndependentCamera()
  {
    independent_camera_ = !independent_camera_;
    cout << "Camera mode: " << (independent_camera_ ?  "Independent" : "Bound to Kinect pose") << endl;
  }
  /*
  modified at 02/02/2019
  */
  void execute(const PtrStepSz<const unsigned short>& depth, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool has_data, bool bSnap)
  { 
	bool has_image = false;
    if (has_data)
    {
      depth_device_.upload (depth.data, depth.step, depth.rows, depth.cols);
      SampledScopeTime fps(time_ms_);
      has_image = kinfu_ (depth_device_);                  
	  // process camera pose
      /*if (pose_processor_)
      {
        pose_processor_->processPose (kinfu_.getCameraPose ());
      }*/
    }
	if (viz_ && has_image)
    {
		if(_bBoxMove == true)
		{
			moveBox();
		}
		//cv::Mat lineMat(cvSize(640,480), CV_8UC1, cvScalar(0));
		//drawBoundingBox(lineMat);
		//image_view_.showScene(kinfu_,_box_face_centers,_box_face_normals,&kinfu_.getCameraPose(),lineMat,_fx,_fy,_px,_py);
		//image_view_.showScene (kinfu_, rgb24, lineMat, &kinfu_.getCameraPose());
		scene_cloud_view_.show(kinfu_, false, _box_face_centers, _box_face_normals, _box_vertices, _box_edges, _bOverView, _camPoses, bSnap);
		//scene_cloud_view_.drawCameraPoses(kinfu_, _camPoses, _box_vertices, _box_edges, _camPoints_ptr);
		//if (current_frame_cloud_view_)
			//current_frame_cloud_view_->show (kinfu_);    
	}    
	/*if (frame_counter_ > 1) {
        if ( frame_counter_%snapshot_rate_== 0)
        {
			saveImage (kinfu_.getCameraPose(), rgb24);
		}
	}
	frame_counter_++;*/
  }
  /*
  created at 01/15/2019
  */
  void getColorMat(cv::Mat &rgbMat)
  {
	  //changeImageExposure(_exposure_delta);
	  stream_color_.start();
	  openni::VideoFrameRef colorMD;
	  int loop = 0;
	  while(loop < 3)
	  {
		stream_color_.readFrame(&colorMD);
		++loop;
	  }
	  if (!colorMD.isValid())
		 return;
	  if (colorMD.getFrameIndex() == 0)
		 return;
	  int width = colorMD.getWidth();
	  int height = colorMD.getHeight();
	  if(rgbMat.rows > 0)
		 rgbMat.empty();
	  //rgbMat.create(cvSize(width,height),CV_8UC3);
	  int fullWidth = colorMD.getVideoMode().getResolutionX();
	  int fullHeight = colorMD.getVideoMode().getResolutionY();
	  unsigned char* pColor = (unsigned char*)colorMD.getData();
	  openni::PixelFormat format = colorMD.getVideoMode().getPixelFormat();
	  unsigned int nPoints = 0;
	  for (int y = 0; y < height; y++)
	  {
		for (int x = 0; x < width; x++)
		{
			switch (format)
			{
			case openni::PIXEL_FORMAT_RGB888:
				rgbMat.at<cv::Vec3b>(y, x)[2] = pColor[0];
				rgbMat.at<cv::Vec3b>(y, x)[1] = pColor[1];
				rgbMat.at<cv::Vec3b>(y, x)[0] = pColor[2];
				pColor += 3;
				break;
			case openni::PIXEL_FORMAT_GRAY8:
				rgbMat.at<cv::Vec3b>(y, x)[0] = pColor[0];
				rgbMat.at<cv::Vec3b>(y, x)[1] = pColor[0];
				rgbMat.at<cv::Vec3b>(y, x)[2] = pColor[0];
				pColor += 1;
				break;
			case openni::PIXEL_FORMAT_GRAY16:
				rgbMat.at<cv::Vec3b>(y, x)[0] = rgbMat.at<cv::Vec3b>(y, x)[1] = rgbMat.at<cv::Vec3b>(y, x)[2] = *((unsigned short*)pColor) >> 2;
				pColor += 2;
				break;
			}
			++nPoints;
		}
	 }
	 stream_color_.stop();
  }
  /*
  created at 01/12/2019
  */
  void create_view_snapped()
  {
	  Eigen::Affine3f camPose = kinfu_.getCameraPose();
	  static Eigen::Vector3f prev_pos = camPose.translation();
	  Eigen::Vector3f vDiff = camPose.translation() - prev_pos;
	  pcl::PointXYZRGB colorPt;
	  colorPt.r = 0;
	  colorPt.g = 255;
	  colorPt.b = 0;
	  colorPt.x = camPose.translation().x();
	  colorPt.y = camPose.translation().y();
	  colorPt.z = camPose.translation().z();
	  if(vDiff.norm() < 0.2f)
	  {
		  _camPoints_ptr->push_back(colorPt);
		  return;
	  }
	  colorPt.r = 255;
	  colorPt.g = 0;
	  colorPt.b = 0;
	  _camPoints_ptr->push_back(colorPt);
	  PCL_WARN("start snapping view.\n");
	  _camPoses.push_back(camPose);
	  prev_pos = camPose.translation();
	  static int screenshot_counter = 1;
	  std::string view_root_name = "scene\\views\\";// + boost::lexical_cast<std::string> (screenshot_counter_);
	  std::string view_name;
	  int view_id = screenshot_counter - 1;
	  ++screenshot_counter;
	  if(screenshot_counter < 10)
	  {
		 view_name = view_root_name + "000" + boost::lexical_cast<std::string> (view_id) + ".mve";
	  }
	  if(screenshot_counter >= 10 && screenshot_counter < 100)
	  {
		 view_name = view_root_name + "00" + boost::lexical_cast<std::string> (view_id) + ".mve";
	  }
	  if(screenshot_counter >= 100)
	  {
		 view_name = view_root_name + "0" + boost::lexical_cast<std::string> (view_id) + ".mve";
	  }
	  mkdir(view_name.c_str());
	  std::string filename_image = view_name + "\\original.jpg";
	  cv::Mat rgbMat(cvSize(640,480),CV_8UC3,cvScalar(0,0,0));
	  getColorMat(rgbMat);
	  cv::imwrite(filename_image,rgbMat);
	  //cv::imshow("grabbing color",rgbMat);
	  std::string filename_ini = view_name + "\\meta.ini";
	  float fx, fy, cx, cy;
	  kinfu_.getDepthIntrinsics(fx,fy,cx,cy);
	  float focal_length = 0.92f;
	  float pixel_aspect = 1.0f;
	  float principal_x = cx/rgbMat.cols;
	  float principal_y = cy/rgbMat.rows;
	  Eigen::Affine3f inverse = camPose.inverse();
	  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotMat = inverse.linear();
	  Eigen::Vector3f trans = inverse.translation();
	  std::ofstream out(filename_ini.c_str(), std::ios::binary);
	   /* Write meta data to file. */
	  out << "# MVE view meta data is stored in INI-file syntax.\n";
      out << "# This file is generated, formatting will get lost.\n";
	  out << "\n[" << "camera" << "]\n" 
	  << "focal_length" << " = " << 0.92 << endl
	  << "focal_length_x" << " = " << fx << endl
	  << "focal_length_y" << " = " << fy << endl
	  << "pixel_aspect" << " = " << 1 << endl
	  << "principal_point" << " = " << principal_x << " " << principal_y << endl 
	  << "rotation =" << rotMat(0,0) <<" "<< rotMat(0,1) <<" "<< rotMat(0,2) <<" "
					  << rotMat(1,0) <<" "<< rotMat(1,1) <<" "<< rotMat(1,2) <<" "
					  << rotMat(2,0) <<" "<< rotMat(2,1) <<" "<< rotMat(2,2) << endl
	  << "translation =" << trans(0) <<" "<< trans(1) <<" "<< trans(2) << endl 
	  << "\n[" << "view" << "]\n" << "id" << " = " << view_id << endl << "name" <<" = "<< view_id << endl;
	  out.close();
	  ++_view_count;
	  PCL_WARN("snapping view is finished.\n");
	  static int count = 0;
	  if(count >=_view_limit_count)
		  exit_ = true;
	  //if(count > 1)
		  //_bBoxMove = false;
	  ++count;
  }
  /*
  created at 02/01/2019
  */
  void snappingScene(Eigen::Affine3f &camPose, cv::Mat &rgbMat)
  {
	  static int screenshot_counter = 0;
	  std::string scene_root_name = "scenes\\scene";
	  std::string scene_name;
	  int scene_id = screenshot_counter;
	  ++screenshot_counter;
	  if(screenshot_counter < 10)
	  {
		 scene_name = scene_root_name + "000" + boost::lexical_cast<std::string> (scene_id);
	  }
	  if(screenshot_counter >= 10 && screenshot_counter < 100)
	  {
		 scene_name = scene_root_name + "00" + boost::lexical_cast<std::string> (scene_id);
	  }
	  if(screenshot_counter >= 100)
	  {
		 scene_name = scene_root_name + "0" + boost::lexical_cast<std::string> (scene_id);
	  }
	  mkdir(scene_name.c_str());
	  _sceneNames.push_back(scene_name);
	  std::string texMeshfileName = scene_name + "\\textured";
	  _texMeshFiles.push_back(texMeshfileName);
	  std::string view_root_name = scene_name + "\\views";
	  mkdir(view_root_name.c_str());
	  std::string view_name = view_root_name + "\\0000" + ".mve";
	  mkdir(view_name.c_str());
	  std::string filename_image = view_name + "\\original.jpg";
	  cv::imwrite(filename_image,rgbMat);

	  std::string filename_ini = view_name + "\\meta.ini";
	  float fx, fy, cx, cy;
	  kinfu_.getDepthIntrinsics(fx,fy,cx,cy);
	  float focal_length = 0.92f;
	  float pixel_aspect = 1.0f;
	  float principal_x = cx/rgbMat.cols;
	  float principal_y = cy/rgbMat.rows;
	  Eigen::Affine3f inverse = camPose.inverse();
	  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotMat = inverse.linear();
	  Eigen::Vector3f trans = inverse.translation();
	  std::ofstream out(filename_ini.c_str(), std::ios::binary);
	   /* Write meta data to file. */
	  out << "# MVE view meta data is stored in INI-file syntax.\n";
      out << "# This file is generated, formatting will get lost.\n";
	  out << "\n[" << "camera" << "]\n" 
	  << "focal_length" << " = " << 0.92 << endl
	  << "focal_length_x" << " = " << fx << endl
	  << "focal_length_y" << " = " << fy << endl
	  << "pixel_aspect" << " = " << 1 << endl
	  << "principal_point" << " = " << principal_x << " " << principal_y << endl 
	  << "rotation =" << rotMat(0,0) <<" "<< rotMat(0,1) <<" "<< rotMat(0,2) <<" "
					  << rotMat(1,0) <<" "<< rotMat(1,1) <<" "<< rotMat(1,2) <<" "
					  << rotMat(2,0) <<" "<< rotMat(2,1) <<" "<< rotMat(2,2) << endl
	  << "translation =" << trans(0) <<" "<< trans(1) <<" "<< trans(2) << endl 
	  << "\n[" << "view" << "]\n" << "id" << " = " << 0 << endl << "name" <<" = "<< 0 << endl;
	  out.close();
  }
  /*
  created at 01/31/2019
  */
  void snappingView(cv::Mat &rgbMat)
  {
	  Eigen::Affine3f camPose = kinfu_.getCameraPose();
	  static int screenshot_counter = 1;
	  std::string view_root_name = "scene\\views\\";// + boost::lexical_cast<std::string> (screenshot_counter_);
	  std::string view_name;
	  int view_id = screenshot_counter - 1;
	  ++screenshot_counter;
	  if(screenshot_counter < 10)
	  {
		 view_name = view_root_name + "000" + boost::lexical_cast<std::string> (view_id) + ".mve";
	  }
	  if(screenshot_counter >= 10 && screenshot_counter < 100)
	  {
		 view_name = view_root_name + "00" + boost::lexical_cast<std::string> (view_id) + ".mve";
	  }
	  if(screenshot_counter >= 100)
	  {
		 view_name = view_root_name + "0" + boost::lexical_cast<std::string> (view_id) + ".mve";
	  }
	  mkdir(view_name.c_str());
	  std::string filename_image = view_name + "\\original.jpg";
	  cv::imwrite(filename_image,rgbMat);
	  //cv::imshow("grabbing color",rgbMat);
	  std::string filename_ini = view_name + "\\meta.ini";
	  float fx, fy, cx, cy;
	  kinfu_.getDepthIntrinsics(fx,fy,cx,cy);
	  float focal_length = 0.92f;
	  float pixel_aspect = 1.0f;
	  float principal_x = cx/rgbMat.cols;
	  float principal_y = cy/rgbMat.rows;
	  Eigen::Affine3f inverse = camPose.inverse();
	  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotMat = inverse.linear();
	  Eigen::Vector3f trans = inverse.translation();
	  std::ofstream out(filename_ini.c_str(), std::ios::binary);
	   /* Write meta data to file. */
	  out << "# MVE view meta data is stored in INI-file syntax.\n";
      out << "# This file is generated, formatting will get lost.\n";
	  out << "\n[" << "camera" << "]\n" 
	  << "focal_length" << " = " << 0.92 << endl
	  << "focal_length_x" << " = " << fx << endl
	  << "focal_length_y" << " = " << fy << endl
	  << "pixel_aspect" << " = " << 1 << endl
	  << "principal_point" << " = " << principal_x << " " << principal_y << endl 
	  << "rotation =" << rotMat(0,0) <<" "<< rotMat(0,1) <<" "<< rotMat(0,2) <<" "
					  << rotMat(1,0) <<" "<< rotMat(1,1) <<" "<< rotMat(1,2) <<" "
					  << rotMat(2,0) <<" "<< rotMat(2,1) <<" "<< rotMat(2,2) << endl
	  << "translation =" << trans(0) <<" "<< trans(1) <<" "<< trans(2) << endl 
	  << "\n[" << "view" << "]\n" << "id" << " = " << view_id << endl << "name" <<" = "<< view_id << endl;
	  out.close();
	  static int count = 0;
	  if(count >=_view_limit_count)
		  exit_ = true;
	  //if(count > 1)
		  //_bBoxMove = false;
	  ++count;
	  snappingScene(camPose, rgbMat);
	  cv::imshow("grabbing color",rgbMat);
  }
  /*
  created at 01/31/2019
  */
  bool recordingPoses()
  {
	  Eigen::Affine3f camPose = kinfu_.getCameraPose();
	  static Eigen::Vector3f prev_pos = camPose.translation();
	  Eigen::Vector3f vDiff = camPose.translation() - prev_pos;
	  pcl::PointXYZRGB colorPt;
	  colorPt.r = 0;
	  colorPt.g = 255;
	  colorPt.b = 0;
	  colorPt.x = camPose.translation().x();
	  colorPt.y = camPose.translation().y();
	  colorPt.z = camPose.translation().z();
	  if(vDiff.norm() < 0.2f)
	  {
		  _camPoints_ptr->push_back(colorPt);
		  return false;
	  }
	  colorPt.r = 255;
	  colorPt.g = 0;
	  colorPt.b = 0;
	  _camPoints_ptr->push_back(colorPt);
	  _camPoses.push_back(camPose);
	  prev_pos = camPose.translation();
	  return true;
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /*
  modified at 02/02/2019
  */
  void
  startMainLoop (bool triggered_capture)
  { 
	  //bool image_view_not_stopped= viz_ ? !image_view_.viewerScene_->wasStopped () : true;
      frame_counter_ = 0;
	  screenshot_counter_ = 0;
	  bool bSnap = false;
	  cv::Mat rgbMat(cvSize(640,480),CV_8UC3,cvScalar(0,0,0));
	  int nLoopForSnap = 0;
	  while (!exit_)// && image_view_not_stopped)
      { 
		  //if(frame_counter_ > limit_count_)
			  //break;
          //readFrame3();
		  if(bSnap)
		  {
			  ++nLoopForSnap;
		  }
		  if(nLoopForSnap > 8)
			  getColorMat(rgbMat);
			  //cv::imshow("grabbing color",rgbMat);
		  getVGADepthData2();
		  bool has_data = true;
		  try { this->execute (depth_, rgb24_, has_data, bSnap); }
		  catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; break; }
          catch (const std::exception& /*e*/) { cout << "Exception" << endl; break; }
		  if(nLoopForSnap > 8)
		  {
			  snappingView(rgbMat);
			  bSnap = false;
			  nLoopForSnap = 0;
			  stream_color_.stop();
		  }
		  if(bSnap==false)
		  {
			  bSnap = recordingPoses();
		  }
		  //create_view_snapped();
	  }
	  stream_depth_.stop();
	  stream_depth_.destroy();
	  stream_color_.stop();
	  stream_color_.destroy();
	  saveMesh();
	  //pcl::TextureMesh texMesh;
	  //_texMeshFileName += ".obj";
	  //pcl::io::loadOBJFile(_texMeshFileName, texMesh);
	  //PCL_INFO("texturefile:%s",texMesh.tex_materials[0].tex_file.c_str());
	  scene_cloud_view_.createMeshViewer(_camPoses[0]);
	  while(_bLoop)
	  {
		scene_cloud_view_.showTexMesh(_camPoses, _camPoints_ptr, _deci_mesh);
		//scene_cloud_view_.showTexMesh(_camPoses, _camPoints_ptr, texMesh);
	  }
  }
  /*
  created at 01/19/2019
  */
  void clippingMesh(boost::shared_ptr<pcl::PolygonMesh> mesh_ptr, boost::shared_ptr<pcl::PolygonMesh> out_mesh_ptr)
  {
	  PCL_INFO("starting clipping");
	  std::vector<int> indices;
	  pcl::PointCloud<pcl::PointXYZ> cloud_origin, cloud_clipped;
	  pcl::fromPCLPointCloud2(mesh_ptr->cloud, cloud_origin);
	  indices.resize(cloud_origin.size());
	  PCL_INFO("cloud origin size:%d",cloud_origin.size());
	  for(int i=0; i < indices.size(); ++i)
		  indices[i] = -1;
	  int count_clipped = 0;
	  for(int i=0; i < cloud_origin.size(); ++i)
	  {
		 pcl::PointXYZ p = cloud_origin.at(i);
		 Eigen::Vector3f v = p.getVector3fMap();
		 if(isInBox(v, _box_face_centers, _box_face_normals)==false)
		   continue;
		 cloud_clipped.push_back(p);
		 indices[i] = count_clipped;
		 ++count_clipped;
	  }
	  PCL_INFO("cloud clipped size:%d",count_clipped);
	  int face_count_clipped = 0;
	  for(int i=0; i < mesh_ptr->polygons.size(); ++i)
	  {
		 pcl::Vertices vertices = mesh_ptr->polygons.at(i);
		 int count = 0;
		 int vert_ids[3];
		 for(int j=0; j < 3; ++j)
		 {
			 if(indices[vertices.vertices[j]]==-1)
				break;
			 vert_ids[j] = indices[vertices.vertices[j]];
			++count;
		 }
		 if(count==3)
		 {
			 pcl::Vertices vertices_clipped;
			 vertices_clipped.vertices.resize(3);
			 for(int j=0; j < 3; ++j)
			 {
				 vertices_clipped.vertices[j] = vert_ids[j]; 
			 }
			 out_mesh_ptr->polygons.push_back(vertices_clipped);
		 }
	  }
	  pcl::toPCLPointCloud2(cloud_clipped,out_mesh_ptr->cloud);
	  PCL_INFO("face count clipped:%d", out_mesh_ptr->polygons.size());
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /*
  created at 01/22/2019
  */
  void decimateMesh(PolygonMesh& in_mesh, PolygonMesh& out_mesh)
  {
	  PolygonMesh mesh_refined;
	  refiningMesh(in_mesh, mesh_refined);
	  pcl::PointCloud<pcl::PointXYZ> cloud_clipped;
	  pcl::fromPCLPointCloud2(mesh_refined.cloud, cloud_clipped);
	  std::vector<Eigen::Vector3f> vertices_clipped;
	  for(int i=0; i < cloud_clipped.size(); ++i)
	  {
		  Eigen::Vector3f v = cloud_clipped.at(i).getVector3fMap();
		  vertices_clipped.push_back(v);
	  }
	  std::vector<int> indices_clipped;
	  for(int i=0; i < mesh_refined.polygons.size(); ++i)
	  {
		  pcl::Vertices vertices = mesh_refined.polygons[i];
		  int id = vertices.vertices[0];
		  indices_clipped.push_back(id);
		  id = vertices.vertices[1];
		  indices_clipped.push_back(id);
		  id = vertices.vertices[2];
		  indices_clipped.push_back(id);
	  }
	  Simplify::importMesh(vertices_clipped, indices_clipped);
	  int target_count = mesh_refined.polygons.size()/10;
	  double agressiveness = 7;
	  Simplify::simplify_mesh(target_count, agressiveness, true);
	  std::vector<Eigen::Vector3f> vertices_decimated;
	  std::vector<int> indices_decimated;
	  Simplify::exportMesh(vertices_decimated, indices_decimated);
	  pcl::PointCloud<pcl::PointXYZ> cloud_decimated;
	  for(int i=0; i < vertices_decimated.size(); ++i)
	  {
		  pcl::PointXYZ p;
		  p.x = vertices_decimated[i][0];
		  p.y = vertices_decimated[i][1];
		  p.z = vertices_decimated[i][2];
		  cloud_decimated.push_back(p);
	  }
	  pcl::toPCLPointCloud2(cloud_decimated, out_mesh.cloud);
	  for(int i=0; i < indices_decimated.size()/3; ++i)
	  {
		  pcl::Vertices vertices;
		  vertices.vertices.resize(3);
		  vertices.vertices[0] = indices_decimated[3*i];
		  vertices.vertices[1] = indices_decimated[3*i+1];
		  vertices.vertices[2] = indices_decimated[3*i+2];
		  out_mesh.polygons.push_back(vertices);
	  }
  }
  /*
  created in 05/04/2017
  modified in 01/31/2019
  */
  void
  writeMesh(int format) 
  {
	  MarchingCubes::Ptr marching_cubes = MarchingCubes::Ptr( new MarchingCubes() );
	  DeviceArray<PointXYZ> triangles_buffer_device;
	  DeviceArray<PointXYZ> triangles_device = marching_cubes->run(kinfu_.volume(), triangles_buffer_device);    
      boost::shared_ptr<pcl::PolygonMesh> mesh_ptr = convertToMesh(triangles_device);
      boost::shared_ptr<pcl::PolygonMesh> clipped_mesh_ptr( new pcl::PolygonMesh() ); 
	  clippingMesh(mesh_ptr, clipped_mesh_ptr);
	  //string rawMesh_file = "scene\\raw_mesh.ply";
	  //writePolygonMeshFile(format, *clipped_mesh_ptr, rawMesh_file);
	  decimateMesh(*clipped_mesh_ptr, _deci_mesh);
	  writePolygonMeshFile(format, _deci_mesh, _plyFileName);
	  //create_texMesh(_sceneName, _plyFileName.c_str(),	_texMeshFileName, 0, false);
	  string fixed_meshfile = "scene\\fixed_mesh.ply";
	  pcl::PolygonMesh newMesh;
	  closeHoles(_deci_mesh, newMesh);
	  PCL_INFO("new patch face count:%d\n", newMesh.polygons.size());
	  string patch_meshfile = "scene\\patch_mesh.ply";
	  writePolygonMeshFile(format, newMesh, patch_meshfile);
	  //fillingHoles(_deci_mesh);
	  writePolygonMeshFile(format, _deci_mesh, fixed_meshfile);
	  //create_texMesh(_sceneName, fixed_meshfile.c_str(),	_texMeshFileName, 0, false);
	  std::string fixedTextured = "fixed_textured";
	  create_texMesh(_sceneName, fixed_meshfile.c_str(), fixedTextured, 600, true);
  }
  /*
  created at 01/30/2019
  */
  void createDeciMesh()
  {
	  MarchingCubes::Ptr marching_cubes = MarchingCubes::Ptr( new MarchingCubes() );
	  DeviceArray<PointXYZ> triangles_buffer_device;
	  DeviceArray<PointXYZ> triangles_device = marching_cubes->run(kinfu_.volume(), triangles_buffer_device);    
      boost::shared_ptr<pcl::PolygonMesh> mesh_ptr = convertToMesh(triangles_device);
      boost::shared_ptr<pcl::PolygonMesh> clipped_mesh_ptr( new pcl::PolygonMesh() ); 
	  clippingMesh(mesh_ptr, clipped_mesh_ptr);
	  decimateMesh(*clipped_mesh_ptr, _deci_mesh);
  }
  /*
  created in 05/04/2017
  */
  void saveMesh()
  {
	  writeMesh(KinFuApp::MESH_PLY);
  }
  //03/14/2017
  void saveImage(const Eigen::Affine3f &camPose, pcl::gpu::PtrStepSz<const PixelRGB> rgb24)
  {
	  PCL_WARN ("[o] [o] [o] [o] Saving screenshot [o] [o] [o] [o]\n");
	  std::string file_extension_image = ".png";
	  std::string file_extension_pose = ".txt";
	  std::string filename_image = "snapshots/";
	  std::string filename_pose = "snapshots/";

	   // Get Pose
	   Eigen::Matrix<float, 3, 3, Eigen::RowMajor> erreMats = camPose.linear ();
       Eigen::Vector3f teVecs = camPose.translation ();

       // Create filenames
       filename_pose = filename_pose + boost::lexical_cast<std::string> (screenshot_counter_) + file_extension_pose;
       filename_image = filename_image + boost::lexical_cast<std::string> (screenshot_counter_) + file_extension_image;

       // Write files
       writePose (filename_pose, teVecs, erreMats);
          
       // Save Image
	   pcl::io::saveRgbPNGFile (filename_image, (unsigned char*)rgb24.data, rgb24.cols,rgb24.rows);
	   screenshot_counter_++;
  }

  /*03/14/2017*/
  void 
  writePose(const std::string &filename_pose, const Eigen::Vector3f &teVecs, const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &erreMats)
  {
    std::ofstream poseFile;
    poseFile.open (filename_pose.c_str());
	float fx, fy, cx, cy;
	kinfu_.getDepthIntrinsics(fx,fy,cx,cy);
	float focal_length = (fx + fy)/2;
	int width = 2*cx;
	int height = 2*cy;
    if (poseFile.is_open())
    {
    poseFile << "TVector" << std::endl << teVecs << std::endl << std::endl 
            << "RMatrix" << std::endl << erreMats << std::endl << std::endl 
            << "Camera Intrinsics: focal height width" << std::endl << focal_length << " " << height << " " << width << std::endl << std::endl;
    poseFile.close ();
    }
    else
    {
    PCL_WARN ("Unable to open/create output file for camera pose!\n");
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  printHelp ()
  {
    cout << endl;
    cout << "KinFu app hotkeys" << endl;
    cout << "=================" << endl;
    cout << "    H    : print this help" << endl;
    cout << "   Esc   : exit" << endl;
    cout << "    T    : take cloud" << endl;
    cout << "    A    : take mesh" << endl;
    cout << "    M    : toggle cloud exctraction mode" << endl;
    cout << "    N    : toggle normals exctraction" << endl;
    cout << "    I    : toggle independent camera mode" << endl;
    cout << "    B    : toggle volume bounds" << endl;
    cout << "    *    : toggle scene view painting ( requires registration mode )" << endl;
    cout << "    C    : clear clouds" << endl;    
    cout << "   1,2,3 : save cloud to PCD(binary), PCD(ASCII), PLY(ASCII)" << endl;
    cout << "    7,8  : save mesh to PLY, VTK" << endl;
    cout << "   X, V  : TSDF volume utility" << endl;
    cout << endl;
  }  

  bool exit_;
  bool scan_;
  bool scan_mesh_;
  bool scan_volume_;

  bool independent_camera_;

  bool registration_;
  bool integrate_colors_;
  float focal_length_;
  
  KinfuTracker kinfu_;
  SceneCloudView scene_cloud_view_;
  ImageView image_view_;
  boost::shared_ptr<CurrentFrameCloudView> current_frame_cloud_view_;
  KinfuTracker::DepthMap depth_device_;

  std::vector<KinfuTracker::PixelRGB> source_image_data_;
  std::vector<unsigned short> source_depth_data_;
  PtrStepSz<const unsigned short> depth_;
  PtrStepSz<const KinfuTracker::PixelRGB> rgb24_;
  boost::shared_ptr<CameraPoseProcessor> pose_processor_;
  int time_ms_;
  int icp_, viz_;

//  boost::shared_ptr<CameraPoseProcessor> pose_processor_;
  //01/16/2018
  float _box_width, _box_height, _box_length, _fOffset;
  Eigen::Vector3f _box_center;
  /////03/14/2017
  bool enable_texture_extraction_;
  int snapshot_rate_;
  int screenshot_counter_;
  int frame_counter_;
  unsigned int limit_count_;
  float max_depth_;
  Eigen::Vector3f _init_vertices[8];
  Eigen::Vector3f _init_centers[8];
  Eigen::Vector3f _init_normals[8];
  Eigen::Vector3f _box_vertices[8];
  int             _box_edges[24];
  int			  _box_faces[24];
  Eigen::Vector3f _box_face_normals[6];
  Eigen::Vector3f _box_face_centers[6];
  float _fx, _fy, _px, _py;
  bool _bBoxMove;
  int _view_count;
  int _view_limit_count;
  //openni2
  openni::Device device_;
  openni::VideoStream stream_depth_, stream_color_;
  const openni::SensorInfo* depthSensorInfo_;
  const openni::SensorInfo* colorSensorInfo_;
  //01/22/2019
  std::string _sceneName;
  std::string _plyFileName;
  std::string _texMeshFileName;
  //01/28/2019
  std::vector<Eigen::Affine3f> _camPoses;
  bool _bOverView;
  PointCloud<PointXYZRGB>::Ptr _camPoints_ptr;
  PolygonMesh _deci_mesh;
  bool _bLoop;
  //02/01/2019
  std::vector<std::string> _sceneNames;
  std::vector<std::string> _texMeshFiles;
  //02/21/2019
  int _exposure_delta;
  /*03/24/2017*/
  bool openDepthStream(int nDepthVideoMode)
  {
	  openni::Status rc = stream_depth_.create(device_, openni::SENSOR_DEPTH);
	  if(rc == openni::STATUS_OK)
	  {
		  depthSensorInfo_ = device_.getSensorInfo(openni::SENSOR_DEPTH);
		  openni::VideoMode mode = depthSensorInfo_->getSupportedVideoModes()[nDepthVideoMode];
		  stream_depth_.setVideoMode(mode);
		  stream_depth_.setMirroringEnabled(false);
		  rc = stream_depth_.start();
		  if(rc != openni::STATUS_OK)
		  {
			printf("Could not start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
			stream_depth_.destroy();
		  }
	  }
	  else
	  {
		  printf("Could not find depth stream:\n%s\n", openni::OpenNI::getExtendedError());
		  return false;
	  }
	  return true;
  }
  /*
  02/21/2019
  */
  void toggleImageAutoExposure()
  {
	if (stream_color_.getCameraSettings() == NULL)
	{
		PCL_INFO("Color stream doesn't support camera settings");
		return;
	}
	stream_color_.getCameraSettings()->setAutoExposureEnabled(stream_color_.getCameraSettings()->getAutoExposureEnabled());
	PCL_INFO("Auto Exposure: %s", stream_color_.getCameraSettings()->getAutoExposureEnabled() ? "ON" : "OFF");
  }
  /*
  02/21/2019
  */
  void changeImageExposure(int delta)
  {
	if (stream_color_.getCameraSettings() == NULL)
	{
		PCL_INFO("Color stream doesn't support camera settings");
		return;
	}
	int exposure = stream_color_.getCameraSettings()->getExposure();
	PCL_INFO("expose time:%d",exposure);
	openni::Status rc = stream_color_.getCameraSettings()->setExposure(exposure + delta);
	PCL_INFO("expose delta:%d",delta);
	if (rc != openni::STATUS_OK)
	{
		PCL_INFO("Can't change exposure");
		return;
	}
	PCL_INFO("Changed exposure to: %d", stream_color_.getCameraSettings()->getExposure());
  }
  /*03/24/2017*/
  bool openColorStream(int nColorVideoMode)
  {
	  openni::Status rc = stream_color_.create(device_, openni::SENSOR_COLOR);
	  if(rc == openni::STATUS_OK)
	  {
		  colorSensorInfo_ = device_.getSensorInfo(openni::SENSOR_COLOR);
		  openni::VideoMode mode = colorSensorInfo_->getSupportedVideoModes()[nColorVideoMode];
		  stream_color_.setVideoMode(mode);
		  stream_color_.setMirroringEnabled(false);
		  //changeImageExposure(_exposure_delta);
		  /*rc = stream_color_.start();
		  if(rc != openni::STATUS_OK)
		  {
			printf("Could not start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			stream_color_.destroy();
		  }*/
	  }
	  else
	  {
		  printf("Could not find color stream:\n%s\n", openni::OpenNI::getExtendedError());
		  return false;
	  }
	  return true;
  }
  /*03/24/2017*/
  bool initOpenni2(int nDepthVideoMode, int nColorVideoMode)
  {
	  openni::Status rc = openni::STATUS_OK;
	  const char* deviceURI = openni::ANY_DEVICE;
	  rc = openni::OpenNI::initialize();
	  printf("After initialization:\n%s\n",openni::OpenNI::getExtendedError());
	  rc = device_.open(deviceURI);
	  if(rc != openni::STATUS_OK)
	  {
		  printf("Device open failed:\%s\n",openni::OpenNI::getExtendedError());
		  openni::OpenNI::shutdown();
		  return 0;
	  }
	  if(!openDepthStream(nDepthVideoMode))
		  return 0;
	  if(!openColorStream(nColorVideoMode))
		  return 0;
	  bool bRegist = device_.isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	  if (bRegist == true)
		 device_.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	 /* if(!stream_depth_.isValid() || !stream_color_.isValid())
	  {
		  printf("No valid streams. Exiting\n");
		  openni::OpenNI::shutdown();
	  }*/
	  return 1;
  }
  /*03/24/2017*/
  void getColorData2()
  {
	  openni::VideoFrameRef colorMD;
	  stream_color_.readFrame(&colorMD);
	  if(!colorMD.isValid())
		  return;
	  if(colorMD.getFrameIndex()==0)
		  return;
	  int width = colorMD.getWidth();
	  int height = colorMD.getHeight();
	  rgb24_.cols = width;
      rgb24_.rows = height;
      rgb24_.step = rgb24_.cols * rgb24_.elemSize(); 
	  source_image_data_.resize(rgb24_.cols * rgb24_.rows);
	  int fullWidth = colorMD.getVideoMode().getResolutionX();
	  int fullHeight = colorMD.getVideoMode().getResolutionY();
	  unsigned char* pColor = (unsigned char*)colorMD.getData();
	  openni::PixelFormat format = colorMD.getVideoMode().getPixelFormat();
	  rgb24_.cols = width;
      rgb24_.rows = height;
      rgb24_.step = rgb24_.cols * rgb24_.elemSize(); 

      source_image_data_.resize(rgb24_.cols * rgb24_.rows);
	  unsigned int nPoints = 0;
	  for(int y=0; y < height; y++)
	  {
		  for(int x=0; x < width; x++)
		  {
			  switch(format)
			  {
			  case openni::PIXEL_FORMAT_RGB888:
				  source_image_data_[nPoints].r = pColor[0];
				  source_image_data_[nPoints].g = pColor[1];
				  source_image_data_[nPoints].b = pColor[2];
				  pColor+=3;
				  break;
			  case openni::PIXEL_FORMAT_GRAY8:
				  source_image_data_[nPoints].r = pColor[0];
				  source_image_data_[nPoints].g = pColor[0];
				  source_image_data_[nPoints].b = pColor[0];
				  pColor+=1;
				  break;
			 case openni::PIXEL_FORMAT_GRAY16:
				  source_image_data_[nPoints].r = source_image_data_[nPoints].g = source_image_data_[nPoints].b = *((unsigned short*)pColor) >> 2;
				  pColor+=2;
				  break;
			  }
			  ++nPoints;
		  }
	  }
	  rgb24_.data = &source_image_data_[0];          
  }
  /*03/24/2017*/
   void getQVGADepthData2()
  {
	  openni::VideoFrameRef depthMD;
	  stream_depth_.readFrame(&depthMD);
	  if(!depthMD.isValid())
		  return;
	  const openni::DepthPixel* pDepthRow = (openni::DepthPixel*)depthMD.getData();
	  int width = 2*depthMD.getWidth();
	  int height = 2*depthMD.getHeight();
	  if(depthMD.getFrameIndex()==0)
		  return;
	  depth_.cols = width;
      depth_.rows = height;
      depth_.step = depth_.cols * depth_.elemSize();
	  source_depth_data_.resize(depth_.cols * depth_.rows);
	  unsigned int nPoints = 0;
	  int sx, sy;
	  int rowSize = depthMD.getStrideInBytes()/sizeof(openni::DepthPixel);
	  for(int y = 0; y < depthMD.getHeight(); ++y)
	  {
		  const openni::DepthPixel* pDepth = pDepthRow;
		  for(int x=0; x < depthMD.getWidth(); ++x, ++pDepth)
		  {
			  sx = 2*x;
			  sy = 2*y;
			  nPoints = sx + sy*width;
			  source_depth_data_[nPoints] = *pDepth;
			  sx = 2*x + 1;
			  nPoints = sx + sy*width;
			  source_depth_data_[nPoints] = *pDepth;
			  sy = 2*y + 1;
			  nPoints = sx + sy*width;
			  source_depth_data_[nPoints] = *pDepth;
			  sx = 2*x;
			  nPoints = sx + sy*width;
			  source_depth_data_[nPoints] = *pDepth;
		  }
		  pDepthRow += rowSize;
	  }
	  depth_.data = &source_depth_data_[0];      
  }
  /*
  created at 01/21/2019
  */
   bool clippingDepthPixel(int x, int y, float depth)
   {
	   depth /= 1000;
	   pcl::PointXYZ pt;
	   pt.x = (x - _px)*depth/_fx;
	   pt.y = (y - _py)*depth/_fy;
	   pt.z = (float)depth;
	   Eigen::Affine3f pose = kinfu_.getCameraPose();
	   Eigen::Vector3f vt_cam, vt_world;
	   vt_cam = pt.getVector3fMap();
	   pcl::transformPoint(vt_cam, vt_world, pose);
	   return isInBox(vt_world, _box_face_centers, _box_face_normals);
   }
   /*
  description:
  get depth values from depth sensor
  convert the depth values more than max_depth in to -1
  created 03/24/2017
  modified 05/05/2017
  */
  void getVGADepthData2()
  {
	  openni::VideoFrameRef depthMD;
	  stream_depth_.readFrame(&depthMD);
	  if(!depthMD.isValid())
		  return;
	  const openni::DepthPixel* pDepthRow = (openni::DepthPixel*)depthMD.getData();
	  int width = depthMD.getWidth();
	  int height = depthMD.getHeight();
	  if(depthMD.getFrameIndex()==0)
		  return;
	  depth_.cols = width;
      depth_.rows = height;
      depth_.step = depth_.cols * depth_.elemSize();
	  source_depth_data_.resize(depth_.cols * depth_.rows);
	  unsigned int nPoints = 0;
	  int rowSize = depthMD.getStrideInBytes()/sizeof(openni::DepthPixel);
	  for(int y = 0; y < height; ++y)
	  {
		  const openni::DepthPixel* pDepth = pDepthRow;
		  for(int x=0; x < width; ++x, ++pDepth)
		  {
			  if(*pDepth > 1000*max_depth_)
				  source_depth_data_[nPoints] = 0;
			  else
			  {
				 source_depth_data_[nPoints] = *pDepth;
			  }
			  ++nPoints;
		  }
		  pDepthRow += rowSize;
	  }
	  depth_.data = &source_depth_data_[0];
  }
  /*03/24/2017*/
  void readFrame2()
  {
	  openni::VideoStream* streams[] = {&stream_depth_, &stream_color_};
	  int changedIndex = -1;
	  openni::Status rc = openni::STATUS_OK;
	  int count1 = 0;
	  int count2 = 0;
	  while (count1 < 1 || count2 < 1)
	  {
		  rc = openni::OpenNI::waitForAnyStream(streams, 2, &changedIndex);
		  if(rc == openni::STATUS_OK)
		  {
			  switch (changedIndex)
			  {
			  case 0:
					getQVGADepthData2();
					count1++;
					break;
			  case 1:
					getColorData2();
					count2++;
					break;
			  default:
					printf("Error in wait\n");
			  }
		  }
	  }
  }
  /*03/26/2017*/
  void readFrame3()
  {
	  if ( frame_counter_ == 0 || (frame_counter_ % snapshot_rate_) > 0 )
		 getVGADepthData2();
	  //if (enable_texture_extraction_ && frame_counter_ > 1)
	  if (frame_counter_ > 1)
	  {
        if ( (frame_counter_  % snapshot_rate_) == 0 )   // Should be defined as a parameter. Done.
        {
			stream_color_.start();
			openni::VideoStream* streams[] = {&stream_depth_, &stream_color_};
			int changedIndex = -1;
			openni::Status rc = openni::STATUS_OK;
			int count1 = 0;
			int count2 = 0;
			while (count1 < 1 || count2 < 1)
			{
			    rc = openni::OpenNI::waitForAnyStream(streams, 2, &changedIndex);
				if(rc == openni::STATUS_OK)
				{
					switch (changedIndex)
					{
					case 0:
						getVGADepthData2();
						count1++;
						break;
					case 1:
						getColorData2();
						count2++;
						break;
					 default:
						printf("Error in wait\n");
					}
				}
			}
			stream_color_.stop();
		}
	 }
  }

  void setLimitCount(unsigned int count)
  {
	  limit_count_ = count;
  }
  void setMaxDepth(float maxDepth)
  {
	  max_depth_ = maxDepth;
  }
  /*05/08/2017*/
  void createObjFile(PolygonMesh &mesh)
  {
	  pcl::io::saveOBJFile("mesh.obj",mesh);
  }
  void computePointNormalOfMesh(PolygonMesh &mesh, pcl::TextureMesh &texMesh)
  {
	  PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
	  pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
	  PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	  tree->setInputCloud(cloud);
	  n.setInputCloud(cloud);
	  n.setSearchMethod(tree);
	  n.setKSearch(20);
	  n.compute(*normals);
	  //Concatenate XYZ and normal fields
	  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
	  pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
	  pcl::toPCLPointCloud2(*cloud_with_normals, texMesh.cloud);
  }
  /*
  created at 01/16/2019
  */
  void initBoundingBox(float width, float height, float length, float fOffset)
  {
	  _box_width = width;
	  _box_height = height;
	  _box_length = length;
	  _fOffset = fOffset;
	  //Eigen::Vector3f vOrigin(-width/2, -height/2, fOffset/2);
	  Eigen::Vector3f vOrigin(0.0f, 0.0f, 0.0f);
	  for(int i=0; i < 8; ++i)
	  {
		 _init_vertices[i] = vOrigin;
	  }
	  _init_vertices[1][0] = 1.0f;
	  _init_vertices[2][0] = 1.0f;
	  _init_vertices[2][2] = 1.0f;
	  _init_vertices[3][2] = 1.0f;
	  for(int i=0;i < 4; ++i)
	  {
		  _init_vertices[i+4] = _init_vertices[i];
		  _init_vertices[i+4][1] = -1.0f;
	  }
	  _box_edges[0] = 0; _box_edges[1] = 1;
	  _box_edges[2] = 1; _box_edges[3] = 2;
	  _box_edges[4] = 2; _box_edges[5] = 3;
	  _box_edges[6] = 3; _box_edges[7] = 0;

	  _box_edges[8] = 4; _box_edges[9] = 5;
	  _box_edges[10] = 5; _box_edges[11] = 6;
	  _box_edges[12] = 6; _box_edges[13] = 7;
	  _box_edges[14] = 7; _box_edges[15] = 4;

	  _box_edges[16] = 0; _box_edges[17] = 4;
	  _box_edges[18] = 1; _box_edges[19] = 5;
	  _box_edges[20] = 2; _box_edges[21] = 6;
	  _box_edges[22] = 3; _box_edges[23] = 7;
	  for(int i=0; i < 8; ++i)
	  {
		 _box_vertices[i] = _init_vertices[i];
	  }

	  _init_normals[0][0] = 0.0f;
	  _init_normals[0][1] = 1.0f;
	  _init_normals[0][2] = 0.0f;

	  _init_centers[0][0] = 0.5f;
	  _init_centers[0][1] = 0.0f;
	  _init_centers[0][2] = 0.5f;

	  _init_normals[1][0] = 0.0f;
	  _init_normals[1][1] = -1.0f;
	  _init_normals[1][2] = 0.0f;

	  _init_centers[1][0] = 0.5f;
	  _init_centers[1][1] = -1.0f;
	  _init_centers[1][2] = 0.5f;

	  _init_normals[2][0] = 0.0f;
	  _init_normals[2][1] = 0.0f;
	  _init_normals[2][2] = -1.0f;

	  _init_centers[2][0] = 0.5f;
	  _init_centers[2][1] = -0.5f;
	  _init_centers[2][2] = 0.0f;

	  _init_normals[3][0] = 0.0f;
	  _init_normals[3][1] = 0.0f;
	  _init_normals[3][2] = 1.0f;

	  _init_centers[3][0] = 0.5f;
	  _init_centers[3][1] = -0.5f;
	  _init_centers[3][2] = 1.0f;

	  _init_normals[4][0] = -1.0f;
	  _init_normals[4][1] = 0.0f;
	  _init_normals[4][2] = 0.0f;
	  
	  _init_centers[4][0] = 0.0f;
	  _init_centers[4][1] = -0.5f;
	  _init_centers[4][2] = 0.5f;

	  _init_normals[5][0] = 1.0f;
	  _init_normals[5][1] = 0.0f;
	  _init_normals[5][2] = 0.0f;
	  
	  _init_centers[5][0] = 1.0f;
	  _init_centers[5][1] = -0.5f;
	  _init_centers[5][2] = 0.5f;
	  for(int i=0; i < 8; ++i)
	  {
		  _init_vertices[i][0] *= width; 
		  _init_vertices[i][1] *= height;
		  _init_vertices[i][2] *= length;
	  }
	  for(int i=0; i < 6; ++i)
	  {
		  _init_centers[i][0] *= width;
		  _init_centers[i][1] *= height;
		  _init_centers[i][2] *= length;
	  }
	  Eigen::Vector3f vOffset(-width/2, height/2, fOffset);
	  for(int i=0; i < 8; ++i)
	  {
		  _init_vertices[i] += vOffset;
	  }
	  for(int i=0; i < 6; ++i)
	  {
		  _init_centers[i] += vOffset;
	  }
	  _box_faces[0] = 0; _box_faces[1] = 1; _box_faces[2] = 2; _box_faces[3] = 3;
	  _box_faces[4] = 4; _box_faces[5] = 5; _box_faces[6] = 6; _box_faces[7] = 7;

	  _box_faces[8] = 0; _box_faces[9] = 1; _box_faces[10] = 5; _box_faces[11] = 4;
	  _box_faces[12] = 3; _box_faces[13] = 2; _box_faces[14] = 6; _box_faces[15] = 7;

	  _box_faces[16] = 0; _box_faces[17] = 3; _box_faces[18] = 7; _box_faces[19] = 4;
	  _box_faces[20] = 1; _box_faces[21] = 2; _box_faces[22] = 6; _box_faces[23] = 5;
  }
  /*
  created at 01/16/2019
  */
  void moveBox()
  {
	  Eigen::Affine3f pose = kinfu_.getCameraPose();
	  Eigen::Vector3f translation = pose.translation();
	  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotMat = pose.linear();
	  for(int i=0; i < 8; ++i)
	  {
		  //_box_vertices[i] = rotMat*_init_vertices[i];
		  //_box_vertices[i] += translation;
		  pcl::transformPoint(_init_vertices[i], _box_vertices[i], pose);
	  }
	  for(int i=0; i < 6; ++i)
	  {
		  pcl::transformPoint(_init_centers[i], _box_face_centers[i], pose);
		  _box_face_normals[i] = rotMat*_init_normals[i];
	  }
  }
  /*
  created at 01/16/2019
  */
  void drawBoundingBox(cv::Mat &lineMat)
  {
	  Eigen::Affine3f pose = kinfu_.getCameraPose();
	  Eigen::Affine3f inverse = pose.inverse();
	  /*Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotMat = inverse.linear();
	  Eigen::Vector3f translation = inverse.translation();*/
	  Eigen::Vector2f pixel_coords[24];
	  Eigen::Vector3f inv_vertices[8];
	  int width = lineMat.cols;
	  int height = lineMat.rows;
	  _fx = 589.948608;
	  _fy = 591.047363;
	  _px = 337.480164;
	  _py = 253.006744;
	  int c=0;
	  for(int i=0; i < 8; ++i)
	  {
		 pcl::transformPoint(_box_vertices[i],inv_vertices[i],inverse);
	  }
	  float k=0.0f;
	  for(int i=0; i < 12; ++i)
	  {
		  Eigen::Vector3f pt1 = inv_vertices[_box_edges[2*i]];
		  Eigen::Vector3f pt2 = inv_vertices[_box_edges[2*i+1]];
		  pixel_coords[2*i][0] = (pt1[0] * _fx / pt1[2]) + _px;
		  pixel_coords[2*i][1] = (pt1[1] * _fy / pt1[2]) + _py;
		  pixel_coords[2*i+1][0] = (pt2[0] * _fx / pt2[2]) + _px;
		  pixel_coords[2*i+1][1] = (pt2[1] * _fy / pt2[2]) + _py;
	  }
	  for(int i=0; i < 12; ++i)
	  {
		  cv::Point p1, p2;
		  p1.x = pixel_coords[2*i][0];
		  p1.y = pixel_coords[2*i][1];
		  p2.x = pixel_coords[2*i+1][0];
		  p2.y = pixel_coords[2*i+1][1];
		  cv::line(lineMat,p1,p2,cvScalar(1),1,8,0);
	  }
  }

  void drawBoxOnDepth()
  {
	  int width = 640;
	  int height = 480;
	  Eigen::Affine3f pose = kinfu_.getCameraPose();
	  Eigen::Affine3f inverse = pose.inverse();
	  Eigen::Vector3f pixel_coords[8];
	  _fx = 589.948608;
	  _fy = 591.047363;
	  _px = 337.480164;
	  _py = 253.006744;
	  for(int i=0; i < 8; ++i)
	  {
		  Eigen::Vector3f pt;
		  pcl::transformPoint(_box_vertices[i],pt,inverse);
		  pixel_coords[i][0] = (pt[0] * _fx / pt[2]) + _px;
		  pixel_coords[i][1] = (pt[1] * _fy / pt[2]) + _py;
		  pixel_coords[i][2] = pt[2];
		  if(pixel_coords[i][0] < 0.0f)
			  pixel_coords[i][0] = 0.0f;
		  if(pixel_coords[i][0] > (width-1))
			  pixel_coords[i][0] = width-1;
		  if(pixel_coords[i][1] < 0.0f)
			  pixel_coords[i][1] = 0.0f;
		  if(pixel_coords[i][1] > (height-1))
			  pixel_coords[i][1] = height-1;
	  }
	  for(int i=0; i < 12; ++i)
	  {
		  Eigen::Vector3f p1, p2;
		  p1[0] = pixel_coords[_box_edges[2*i]][0];
		  p1[1] = pixel_coords[_box_edges[2*i]][1];
		  p1[2] = pixel_coords[_box_edges[2*i]][2];
		  p2[0] = pixel_coords[_box_edges[2*i+1]][0];
		  p2[1] = pixel_coords[_box_edges[2*i+1]][1];
		  p2[2] = pixel_coords[_box_edges[2*i+1]][2];
		  int dx = std::abs(p2[0] - p1[0]);
		  int dy = std::abs(p2[1] - p1[1]);
		  int d = std::max(dx,dy);
		  if(d==0)
			  continue;
		  Eigen::Vector3f vDir = (p2 - p1)/d;
		  for(int i=0; i < d; ++i)
		  {
			  Eigen::Vector3f p = p1 + i*vDir;
			  int nLocation = p[0] + p[1]*width;
			  source_depth_data_[nLocation] = 1000*p[2];
		  }
	  }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  static void
  keyboard_callback (const visualization::KeyboardEvent &e, void *cookie)
  {
    KinFuApp* app = reinterpret_cast<KinFuApp*> (cookie);

    int key = e.getKeyCode ();

    if (e.keyUp ())    
      switch (key)
      {
      case 27: 
		  app->exit_ = true;
		  app->_bLoop = false;
		  break;
	  case (int)'o': case (int)'O': app->_bOverView = true; break;
	  case (int)'p': case (int)'P': app->_bOverView = false; break;
	  case (int)'b': case (int)'B': app->_bBoxMove = true; break;
	  case (int)'n': case (int)'N': app->_bBoxMove = false; break;
      case (int)'t': case (int)'T': app->scan_ = true; break;
      case (int)'a': case (int)'A': app->scan_mesh_ = true; break;
      case (int)'h': case (int)'H': app->printHelp (); break;
//    case (int)'i': case (int)'I': app->toggleIndependentCamera (); break;
      case (int)'7': case (int)'8': app->writeMesh (key - (int)'0'); break;
	  case (int)'s': case (int) 'S': app->saveMesh();
      case (int)'x': case (int)'X':
        app->scan_volume_ = !app->scan_volume_;
        cout << endl << "Volume scan: " << (app->scan_volume_ ? "enabled" : "disabled") << endl << endl;
        break;
      default:
        break;
      }    
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
writePolygonMeshFile (int format, const pcl::PolygonMesh& mesh, string &meshFileName)
{
  if (format == KinFuApp::MESH_PLY)
  {
    cout << "Saving mesh to to 'mesh.ply'... " << flush;
	pcl::io::savePLYFile(meshFileName, mesh);		
    //pcl::io::savePLYFile("scene\\mesh.ply", mesh);		
  }
  else /* if (format == KinFuApp::MESH_VTK) */
  {
    cout << "Saving mesh to to 'mesh.vtk'... " << flush;
    pcl::io::saveVTKFile("mesh.vtk", mesh);    
  }  
  cout << "Done" << endl;
}

void filterMesh(PolygonMesh &input_mesh, PolygonMesh &out_mesh)
{
	PointCloud<PointXYZ>::Ptr input_cloud(new PointCloud<PointXYZ>);
	pcl::fromPCLPointCloud2(input_mesh.cloud, *input_cloud);
	PointCloud<PointXYZ>::Ptr out_cloud(new PointCloud<PointXYZ>);
	vector<int> indices;
	indices.resize(input_cloud->size());
	cout << indices.size() << flush;
	for(int i=0; i < indices.size(); i++)
	{
		indices[i] = 1;
	}
	out_mesh.header = input_mesh.header;
	for(int j=0; j < input_cloud->size(); j++)
	{
		if(indices[j]==0)
			continue;
		Eigen::Vector3f vec3_1 = input_cloud->points[j].getVector3fMap();
		out_cloud->push_back(input_cloud->points[j]);
		for(int i=j; i < input_cloud->size(); i++)
		{
			Eigen::Vector3f vec3_2 = input_cloud->points[i].getVector3fMap();
			if(vec3_1==vec3_2)
				indices[i] = 0;
		}
	}
	for(size_t t=0; t < input_mesh.polygons.size(); t++)
	{
		unsigned int v[3];
		v[0] = input_mesh.polygons[t].vertices[0];
		v[1] = input_mesh.polygons[t].vertices[1];
		v[2] = input_mesh.polygons[t].vertices[2];
		Eigen::Vector3f vec3f[3];
		vec3f[0] = input_cloud->points[v[0]].getVector3fMap();
		vec3f[1] = input_cloud->points[v[1]].getVector3fMap();
		vec3f[2] = input_cloud->points[v[2]].getVector3fMap();
		if(vec3f[0] == vec3f[1] || vec3f[0] == vec3f[2])
			continue;
		if(vec3f[1] == vec3f[2])
			continue;
		pcl::Vertices vertices;
		for(int j=0; j < 3; j++)
		{
			Eigen::Vector3f vec_1 = input_cloud->points[v[j]].getVector3fMap();
			int count = 0;
			for(int i=0; i < out_cloud->size(); i++)
			{
				Eigen::Vector3f vec_2 = out_cloud->points[i].getVector3fMap();
				if(vec_1==vec_2)
				{
					input_mesh.polygons[t].vertices[j] = i;
				}
			}
		}
	}
	pcl::toPCLPointCloud2(*out_cloud, out_mesh.cloud);
}
/*
created at 01/21/2019
modified at 01/22/2019
*/
void refiningMesh(PolygonMesh &in_mesh, PolygonMesh &out_mesh)
{
	PointCloud<PointXYZ> cloud1, cloud2, cloud3;
	pcl::fromPCLPointCloud2(in_mesh.cloud, cloud1);
	int count = 0;
	//extracting valid vertices from mesh
	for(int i=0; i < in_mesh.polygons.size(); ++i)
	{
		for(int j=0; j < 3; ++j)
		{
			int id = in_mesh.polygons[i].vertices[j];
			PointXYZ p = cloud1.points[id];
			cloud2.push_back(p);
			in_mesh.polygons[i].vertices[j] = count;
			++count;
		}
	}
	//extracting unique vertices
	std::vector<int> indices;
	indices.resize(cloud2.size());
	for(int i=0; i < cloud2.size(); ++i)
	{
		indices[i] = -1;
	}
	count = 0;
	for(int i=0; i < cloud2.size(); ++i)
	{
		if(indices[i]==-1)
		{
			PointXYZ p1 = cloud2.points[i];
			for(int j=i; j < cloud2.size(); ++j)
			{
				PointXYZ p2 = cloud2.points[j];
				if(p1.x==p2.x && p1.y==p2.y && p1.z==p2.z)
					indices[j] = count;
			}
			cloud3.push_back(p1);
			++count;
		}
	}
	//constructing new mesh refined
	for(int i=0; i < in_mesh.polygons.size(); ++i)
	{
		Vertices vertices;
		vertices.vertices.resize(3);
		for(int j=0; j < 3; ++j)
		{
			int id = in_mesh.polygons[i].vertices[j];
			vertices.vertices[j] = indices[id];
		}
		if(vertices.vertices[0]==vertices.vertices[1] || vertices.vertices[0]==vertices.vertices[2])
			continue;
		if(vertices.vertices[1]==vertices.vertices[2])
			continue;
		out_mesh.polygons.push_back(vertices);
	}
	pcl::toPCLPointCloud2(cloud3,out_mesh.cloud);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
print_cli_help ()
{
  cout << "\nKinFu parameters:" << endl;
  cout << "    --help, -h                              : print this message" << endl;  
  cout << "    --registration, -r                      : try to enable registration (source needs to support this)" << endl;
  cout << "    --current-cloud, -cc                    : show current frame cloud" << endl;
  cout << "    --save-views, -sv                       : accumulate scene view and save in the end ( Requires OpenCV. Will cause 'bad_alloc' after some time )" << endl;  
  cout << "    --integrate-colors, -ic                 : enable color integration mode (allows to get cloud with colors)" << endl;   
  cout << "    --scale-truncation, -st                 : scale the truncation distance and raycaster based on the volume size" << endl;
  cout << "    -volume_size <size_in_meters>           : define integration volume size" << endl;
  cout << "    --depth-intrinsics <fx>,<fy>[,<cx>,<cy> : set the intrinsics of the depth camera" << endl;
  cout << "    -save_pose <pose_file.csv>              : write tracked camera positions to the specified file" << endl;
  cout << "Valid depth data sources:" << endl; 
  cout << "    -dev <device> (default), -oni <oni_file>, -pcd <pcd_file or directory>" << endl;
  cout << "";
  cout << " For RGBD benchmark (Requires OpenCV):" << endl; 
  cout << "    -eval <eval_folder> [-match_file <associations_file_in_the_folder>]" << endl;
    
  return 0;
}
  
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
main (int argc, char* argv[])
{  
  if (pc::find_switch (argc, argv, "--help") || pc::find_switch (argc, argv, "-h"))
    return print_cli_help ();
  
  int device = 0;
  pc::parse_argument (argc, argv, "-gpu", device);
  pcl::gpu::setDevice (device);
  pcl::gpu::printShortCudaDeviceInfo (device);
   
  bool triggered_capture = false;
   
  float volume_size = 3.f;
  pc::parse_argument (argc, argv, "-volume-size", volume_size);

  int icp = 1, visualization = 1;
  std::vector<float> depth_intrinsics;
  pc::parse_argument (argc, argv, "--icp", icp);
  pc::parse_argument (argc, argv, "--viz", visualization);
  boost::shared_ptr<CameraPoseProcessor> pose_processor;
  KinFuApp app (volume_size, icp, visualization, pose_processor);
    
  if (pc::find_switch (argc, argv, "--integrate-colors") || pc::find_switch (argc, argv, "-ic"))
    app.toggleColorIntegration();

  if (pc::find_switch (argc, argv, "--scale-truncation") || pc::find_switch (argc, argv, "-st"))
    app.enableTruncationScaling();
  depth_intrinsics.resize(4);
  depth_intrinsics[0] = 589.948608;
  depth_intrinsics[1] = 591.047363;
  depth_intrinsics[2] = 337.480164;
  depth_intrinsics[3] = 253.006744;
  app.setDepthIntrinsics(depth_intrinsics);
  if (pc::parse_x_arguments (argc, argv, "--depth-intrinsics", depth_intrinsics) > 0)
  {
    if ((depth_intrinsics.size() == 4) || (depth_intrinsics.size() == 2))
    {
       app.setDepthIntrinsics(depth_intrinsics);
    }
    else
    {
        pc::print_error("Depth intrinsics must be given on the form fx,fy[,cx,cy].\n");
        return -1;
    }   
  }
  int exposure_delta = 10;
  pc::parse_argument (argc, argv, "-expose", exposure_delta);
  app._exposure_delta = exposure_delta;
  int depth_mode = 4;
  int color_mode = 9;
  pc::parse_argument (argc, argv, "-depth-mode", depth_mode);
  pc::parse_argument (argc, argv, "-color-mode", color_mode);
   if(!app.initOpenni2(depth_mode, color_mode))
	   return -1;
   app.initBoundingBox(0.8f, 0.8f, 0.8f, 0.5f);
   if (pc::find_switch (argc, argv, "--current-cloud") || pc::find_switch (argc, argv, "-cc"))
    app.initCurrentFrameView ();
  //03/14/2017
  /*app.enable_texture_extraction_ = false;
  if (pc::find_switch (argc, argv, "--extract-textures") || pc::find_switch (argc, argv, "-et"))  
  {
	  app.enable_texture_extraction_ = true;
  }*/
  //03/30/2017
  unsigned int loop_count = 1200;
  if (pc::parse_argument (argc, argv, "-loop-count", loop_count) > 0)
	  app.setLimitCount(loop_count);
  int view_count = 5;
  if (pc::parse_argument (argc, argv, "-view-count", view_count) > 0)
	  app._view_limit_count = view_count;
  //03/14/2017
  int snapshot_rate = 45;
  pc::parse_argument (argc, argv, "--snapshot_rate", snapshot_rate);
  pc::parse_argument (argc, argv, "-sr", snapshot_rate);
  app.snapshot_rate_ = snapshot_rate;
  float max_depth = 3.0f;
  pc::parse_argument (argc, argv, "-max-depth", max_depth);
  app.setMaxDepth(max_depth);
  cv::namedWindow("grabbing color");
  std::string scene_name="e:\\bin\\scene::original";
  //std::string scene_name="c:\\Code\\bin\\scene::original";
  pc::parse_argument (argc, argv, "-scene", scene_name);
  app._sceneName = scene_name;
  std::string plyFileName="e:\\bin\\scene\\surface.ply";
  //std::string plyFileName="c:\\Code\\bin\\scene\\surface.ply";
  pc::parse_argument (argc, argv, "-plyFileName", plyFileName);
  app._plyFileName = plyFileName;
  std::string texFileName="textured";
  //std::string texFileName="c:\\Code\\bin\\textured";
  pc::parse_argument (argc, argv, "-texFileName", texFileName);
  app._texMeshFileName = texFileName;
  // executing
  try { app.startMainLoop (triggered_capture); }
  catch (const pcl::PCLException& /*e*/) { cout << "PCLException" << endl; }
  catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; }
  catch (const std::exception& /*e*/) { cout << "Exception" << endl; }

#ifdef HAVE_OPENCV
  for (size_t t = 0; t < app.image_view_.views_.size (); ++t)
  {
    if (t == 0)
    {
      cout << "Saving depth map of first view." << endl;
      cv::imwrite ("./depthmap_1stview.png", app.image_view_.views_[0]);
      cout << "Saving sequence of (" << app.image_view_.views_.size () << ") views." << endl;
    }
    char buf[4096];
    sprintf (buf, "./%06d.png", (int)t);
    cv::imwrite (buf, app.image_view_.views_[t]);
    printf ("writing: %s\n", buf);
  }
#endif

  return 0;
}

