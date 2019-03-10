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
//#include "tsdf_volume.h"
//#include "tsdf_volume.hpp"

//#include "camera_pose.h"

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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
writePolygonMeshFile (int format, const pcl::PolygonMesh& mesh);
void filterMesh(PolygonMesh &input_mesh);
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ImageView
{
  ImageView(int viz) : viz_(viz)//);, paint_image_ (false), accumulate_views_ (false)
  {
    if (viz_)
    {
        viewerScene_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);
        viewerScene_->setWindowTitle ("View3D from ray tracing");
        viewerScene_->setPosition (0, 0);
    }
	loop = 0;
  }

  void
  showScene (KinfuTracker& kinfu, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool bWarnning, 
			 int frame_count, int snap_rate, Eigen::Affine3f* pose_ptr = 0)
  {
     kinfu.getImage (view_device_);
	 int cols;
     view_device_.download (view_host_, cols);
     if (viz_)
	 {
		 unsigned char r, g, b;
		 r = 0; g = 255; b = 0;
		 float rate = (float)(frame_count%snap_rate)/snap_rate;
		 int nWarn = rate*view_device_.cols();
		 //draw progress bar
		 for(int y=0; y < 5; y++)
		 {
			 for(int x=0; x < nWarn; x++)
			 {
				 unsigned int l = x + y*view_device_.cols();
				 view_host_[l].b = b;
				 view_host_[l].g = g;
				 view_host_[l].r = r;
			 }
		 }
        viewerScene_->showRGBImage (reinterpret_cast<unsigned char*> (&view_host_[0]), view_device_.cols (), view_device_.rows ());
	 }
	 if(loop > 26)
	 {
		viewerScene_->spinOnce(false);
		loop = 0;
	 }
     loop++;
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
#ifdef HAVE_OPENCV
  vector<cv::Mat> views_;
#endif
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct KinFuApp
{
   enum { PCD_BIN = 1, PCD_ASCII = 2, PLY = 3, MESH_PLY = 7, MESH_VTK = 8 };
   KinFuApp(float vsz, int icp, int viz) : exit_ (false), scan_ (false), scan_mesh_(false), scan_volume_ (false), independent_camera_ (false),
      registration_ (false), integrate_colors_ (false), focal_length_(-1.f), image_view_(viz), time_ms_(0), icp_(icp), viz_(viz)
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
        image_view_.viewerScene_->registerKeyboardCallback (keyboard_callback, (void*)this);
    }
	limit_count_ = 1200;
	max_depth_ = 3.0f;
  }

  ~KinFuApp()
  {
    //if (evaluation_ptr_)
      //evaluation_ptr_->saveAllPoses(kinfu_);
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
  
  void execute(const PtrStepSz<const unsigned short>& depth, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool has_data)
  {        
    bool has_image = false;
    if (has_data)
    {
      depth_device_.upload (depth.data, depth.step, depth.rows, depth.cols);
      SampledScopeTime fps(time_ms_);
      has_image = kinfu_ (depth_device_);                  
    }
	if (viz_ && has_image)
    {
		image_view_.showScene (kinfu_, rgb24, true, frame_counter_, snapshot_rate_, 0);
	}    
	if (frame_counter_ > 1) {
        if ( frame_counter_%snapshot_rate_== 0)
        {
			saveImage (kinfu_.getCameraPose(), rgb24);
		}
	}
	frame_counter_++;
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  startMainLoop (bool triggered_capture)
  { 
	  bool image_view_not_stopped= viz_ ? !image_view_.viewerScene_->wasStopped () : true;
      frame_counter_ = 0;
	  screenshot_counter_ = 0;
	  while (!exit_ && image_view_not_stopped)
      { 
		  if(frame_counter_ > limit_count_)
			  break;
          readFrame3();
		  bool has_data = true;
		  try { this->execute (depth_, rgb24_, has_data); }
		  catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; break; }
          catch (const std::exception& /*e*/) { cout << "Exception" << endl; break; }
      }
      stream_depth_.stop();
	  stream_depth_.destroy();
	  stream_color_.stop();
	  stream_color_.destroy();
	  saveMesh();
  }
  
  
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /*
  created in 05/04/2017
  */
  void
  writeMesh(int format) const
  {
	  MarchingCubes::Ptr marching_cubes = MarchingCubes::Ptr( new MarchingCubes() );
	  DeviceArray<PointXYZ> triangles_buffer_device;
	  DeviceArray<PointXYZ> triangles_device = marching_cubes->run(kinfu_.volume(), triangles_buffer_device);    
      boost::shared_ptr<pcl::PolygonMesh> mesh_ptr = convertToMesh(triangles_device);
      writePolygonMeshFile(format, *mesh_ptr);
	  //pcl::io::saveOBJFile("mesh.obj",*mesh_ptr);
	  //filterMesh(*mesh_ptr);
	  //pcl::io::saveOBJFile("mesh1.obj",*mesh_ptr);
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

  ImageView image_view_;

  KinfuTracker::DepthMap depth_device_;

  std::vector<KinfuTracker::PixelRGB> source_image_data_;
  std::vector<unsigned short> source_depth_data_;
  PtrStepSz<const unsigned short> depth_;
  PtrStepSz<const KinfuTracker::PixelRGB> rgb24_;

  int time_ms_;
  int icp_, viz_;

//  boost::shared_ptr<CameraPoseProcessor> pose_processor_;
  /////03/14/2017
  bool enable_texture_extraction_;
  int snapshot_rate_;
  int screenshot_counter_;
  int frame_counter_;
  unsigned int limit_count_;
  float max_depth_;
  //openni2
  openni::Device device_;
  openni::VideoStream stream_depth_, stream_color_;
  const openni::SensorInfo* depthSensorInfo_;
  const openni::SensorInfo* colorSensorInfo_;
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
				  source_depth_data_[nPoints] = -1;
			  else
				  source_depth_data_[nPoints] = *pDepth;
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
		  break;
      case (int)'t': case (int)'T': app->scan_ = true; break;
      case (int)'a': case (int)'A': app->scan_mesh_ = true; break;
      case (int)'h': case (int)'H': app->printHelp (); break;
//      case (int)'i': case (int)'I': app->toggleIndependentCamera (); break;
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
writePolygonMeshFile (int format, const pcl::PolygonMesh& mesh)
{
  if (format == KinFuApp::MESH_PLY)
  {
    cout << "Saving mesh to to 'mesh.ply'... " << flush;
    pcl::io::savePLYFile("mesh.ply", mesh);		
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
        
  KinFuApp app (volume_size, icp, visualization);
    
  if (pc::find_switch (argc, argv, "--integrate-colors") || pc::find_switch (argc, argv, "-ic"))
    app.toggleColorIntegration();

  if (pc::find_switch (argc, argv, "--scale-truncation") || pc::find_switch (argc, argv, "-st"))
    app.enableTruncationScaling();
  
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
  int depth_mode = 4;
  int color_mode = 15;
  pc::parse_argument (argc, argv, "-depth-mode", depth_mode);
  pc::parse_argument (argc, argv, "-color-mode", color_mode);
   if(!app.initOpenni2(depth_mode, color_mode))
	   return -1;
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
  //03/14/2017
  int snapshot_rate = 45;
  pc::parse_argument (argc, argv, "--snapshot_rate", snapshot_rate);
  pc::parse_argument (argc, argv, "-sr", snapshot_rate);
  app.snapshot_rate_ = snapshot_rate;
  float max_depth = 3.0f;
  pc::parse_argument (argc, argv, "-max-depth", max_depth);
  app.setMaxDepth(max_depth);
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

