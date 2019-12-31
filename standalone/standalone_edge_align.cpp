#include <iostream>
using namespace std;

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>
#include <string>
#include <fstream>
// Eigen
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

// ceres
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>
using namespace ceres;
// using namespace ceres::Grid2D;


#include "utils.h"
#include "PoseManipUtils.h"

#include <chrono>

int SavePointCloudToObj(const std::string saveToPath, const cv::Mat &points, int iMatDataType)
{
	//#define CV_8U   0
	//#define CV_8S   1
	//#define CV_16U  2
	//#define CV_16S  3
	//#define CV_32S  4
	//#define CV_32F  5
	//#define CV_64F  6
	//#define CV_USRTYPE1 7
	if (points.empty())
	{
		std::cout << "no points\n";
		return -1;
	}
	if (saveToPath.empty())
	{
		std::cout << "path empty\n";
		return -1;
	}

	std::ofstream fout(saveToPath.c_str());
	for (int i = 0; i < points.rows; i++)
	{
		fout << "v" << ' ';
		for (int j = 0; j < points.cols; j++)
		{
			if (iMatDataType == 0)
			{
				if (j < points.cols - 1)
				{
					fout << points.at<unsigned char>(i, j) << ' ';
				}
				else
				{
					fout << points.at<unsigned char>(i, j);
				}
			}
			else if (iMatDataType == 4)
			{
				if (j < points.cols - 1)
				{
					fout << points.at<int>(i, j) << ' ';
				}
				else
				{
					fout << points.at<int>(i, j);
				}
			}
			else if (iMatDataType == 5)
			{
				if (j < points.cols - 1)
				{
					fout << points.at<float>(i, j) << ' ';
				}
				else
				{
					fout << points.at<float>(i, j);
				}
			}
			else if (iMatDataType == 6)
			{
				if (j < points.cols - 1)
				{
					fout << std::setprecision(45) << points.at<double>(i, j) << ' ';
				}
				else
				{
					fout << std::setprecision(45) << points.at<double>(i, j);
				}
			}
		}
		if (i < points.rows - 1)
		{
			fout << '\n';
		}
	}
	fout.close();
	return 0;
}

// original test,  align two images
int edge_align_test1()
{
    /**********************************************
     * 
     * loading rgb and depth images
     * 
     * ****************************************** */
    
    // Load Image A and its Depth image
    cv::Mat imA = cv::imread( "../rgb-d/rgb/1.png");
    cv::Mat imA_depth = cv::imread( "../rgb-d/depth/1.png", CV_LOAD_IMAGE_ANYDEPTH);
    //
    // Load Image B. No depth is needed for this
    cv::Mat imB = cv::imread( "../rgb-d/rgb/3.png");

    //
    // Print Info on Images
    cout << "==========INFO ON IMages ===========\n";
    cout << "imA.type "<< mat_info( imA ) << endl;  // 640x480 CV_8UC3
    cout << "imA_depth.type "<< mat_info( imA_depth ) << endl; // 640x480 CV_16UC1
    cout << "imB.type "<< mat_info( imB ) << endl;  // 640x480 CV_8UC3

    //cv::imshow( "A", imA );
    //cv::imshow( "B", imB );
    //cv::imshow( "imA_depth", imA_depth );
    cv::imwrite("A.png", imA);
    cv::imwrite("B.png", imB);
    cv::imwrite("imA_depth.png", imA_depth);

    double min, max;
    cv::minMaxLoc(imA_depth, &min, &max);
    cout << "imA_depth min,max: " << min << " " << max  << endl;
    //
    // DONE printing debug info on images
    cout << "=========== Done ============\n";
    /***********************************************
     * 
     *  getting calibration parameters 
     * 
     * ********************************************/
    //
    // Intrinsic Calibration. Obtained from https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    Eigen::Matrix3d K;
    double fx = 525., fy = 525., cx = 319.5, cy = 239.5;
    K << fx, 0., cx , 0., fy, cy, 0., 0., 1.;
    cout << "K\n" << K << endl;

    Eigen::Matrix<double,5,1> D; //d0, d1, d2, d3, d4
    D << 0.2624,	-0.9531,	-0.0054,	0.0026,	1.1633;
    cout << "D\n" << D << endl;

	float  zScaling = 5000;
	cout << "zScaling\n" << zScaling << endl;
    /***********************************************
     * 
     *  Get 3D points using the depth image and 
     *  intrinsic matrix
     * ********************************************/
    //
    // Get 3D points of imA as a 4xN matrix. ^aX. These 3d points are in frame-of-ref of imA.
    Eigen::MatrixXd a_X;
    get_aX( imA, imA_depth, K, zScaling, a_X );
	cout << "Total number of Edge points = " << a_X.cols() << endl;
	cv::Mat edgePointcloud = cv::Mat(a_X.cols(), 3, CV_32FC1);
	for (int i = 0; i < a_X.cols(); i++)
	{
		edgePointcloud.at<float>(i, 0) = a_X(0, i);
		edgePointcloud.at<float>(i, 1) = a_X(1, i);
		edgePointcloud.at<float>(i, 2) = a_X(2, i);
	}
	SavePointCloudToObj("sceneEdgeCloud.obj", edgePointcloud, CV_32FC1);


	Eigen::MatrixXd all_a_X;
	get_AllaX(imA, imA_depth, K, zScaling, all_a_X);
	cout << "Total number of points = " << all_a_X.cols() << endl;
	cv::Mat pointcloud = cv::Mat(all_a_X.cols(), 3, CV_32FC1);
	for (int i = 0; i < all_a_X.cols(); i++)
	{
		pointcloud.at<float>(i, 0) = all_a_X(0, i);
		pointcloud.at<float>(i, 1) = all_a_X(1, i);
		pointcloud.at<float>(i, 2) = all_a_X(2, i);
	}
	SavePointCloudToObj("sceneCloud.obj", pointcloud, CV_32FC1);
    /***********************************************
     * 
     * Get Distance Transform 
     * 
     * ********************************************/
    //
    // Distance Transform of edges of imB
    cv::Mat disTrans;
    get_distance_transform( imB, disTrans );
     // cv::imshow("Distance Transform Image", disTrans); //numbers between 0 and 1.
    cv::imwrite("Distance Transform Image.png", disTrans * 255);

    Eigen::MatrixXd e_disTrans;
    cv::cv2eigen( disTrans, e_disTrans );

    /***********************************************
     * 
     *  Verification
     * 
     * ********************************************/
    #if 1
    //
    // Verify if everything is OK.
    // Use the 3d points are reproject of imA and overlay those reprojected points.
    cout << "a_X\n" << a_X.leftCols(10) << endl;
    for( int ss=0 ; ss<150 ; ss+=10 )
    {
        cout << ":::::::" <<  ss << " to " << ss+10 << endl;
        cout << a_X.block( 0, ss, 4, 10  ) << endl;
    }

    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    Eigen::MatrixXd b_u_rep;
    reproject( a_X, M, K, b_u_rep );
    cout << "b_u\n" << b_u_rep.leftCols(10) << endl;
   // s_overlay( imB, b_u_rep );


    #endif


    /***********************************************
     * 
     *  Initial Guess
     * 
     * ********************************************/
    // Initial Guess
    Eigen::Matrix4d b_T_a_optvar = Eigen::Matrix4d::Identity();

    cout << "Initial Guess : " << PoseManipUtils::prettyprintMatrix4d( b_T_a_optvar ) << endl;
    Eigen::MatrixXd b_u;
    reproject( a_X, b_T_a_optvar, K, b_u );
    s_overlay( imB, b_u, "initial.png" );

    /***********************************************
     * 
     *  Solve
     * 
     * ********************************************/
    ////////////////////////////////////////////////////////////////////////////
    ///////////////////// Setup non-linear Least Squares ///////////////////////
    ////////////////////////////////////////////////////////////////////////////

    // Using `a_X` and `disTrans` setup the non-linear least squares problem
    cout << "e_disTrans.shape = " << e_disTrans.rows() << ", " << e_disTrans.cols() << endl;
    ceres::Grid2D<double,1> grid( e_disTrans.data(), 0, e_disTrans.cols(), 0, e_disTrans.rows() );
    ceres::BiCubicInterpolator< ceres::Grid2D<double,1> > interpolated_imb_disTrans( grid );

    double b_quat_a[10], b_t_a[10]; // quaternion, translation
    PoseManipUtils::eigenmat_to_raw( b_T_a_optvar, b_quat_a, b_t_a ); 

    // Residues for each 3d points
    ceres::Problem problem;
    int count = 0;
    for( int i=0 ; i< a_X.cols() ; i+=30 )
    {
        // ceres::CostFunction * cost_function = EAResidue::Create( K, a_X.col(i), interpolated_imb_disTrans);

        ceres::CostFunction * cost_function = EAResidue::Create( fx,fy,cx,cy,  a_X(0,i),a_X(1,i),a_X(2,i), interpolated_imb_disTrans);
        problem.AddResidualBlock( cost_function, new CauchyLoss(1.), b_quat_a, b_t_a );
        count++;
    }
    std::cout << "Point count = " << count << "\n"; 

    ceres::LocalParameterization * quaternion_parameterization = new ceres::QuaternionParameterization;
    problem.SetParameterization( b_quat_a, quaternion_parameterization );

    auto start1 = std::chrono::high_resolution_clock::now();
    // Run
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;
    Solver::Summary summary;
    ceres::Solve( options, &problem, &summary );

    auto finish1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed1 = finish1 - start1;
	double time = elapsed1.count();
	std::cout << "solve done" << ". " << "time = " << time << "\n";

    std::cout << summary.FullReport() << "\n";

   /***********************************************
     * 
     *  Result
     * 
     * ********************************************/
    PoseManipUtils::raw_to_eigenmat( b_quat_a, b_t_a, b_T_a_optvar );
    cout << "Final Guess : " << PoseManipUtils::prettyprintMatrix4d( b_T_a_optvar ) << endl;

    reproject( a_X, b_T_a_optvar, K, b_u );
    s_overlay( imB, b_u, "final.png" );
}

// my test, align two images
int edge_align_test2()
{
	/**********************************************
	*
	* loading rgb and depth images
	*
	* ****************************************** */

	// Load Image A and its Depth image
	cv::Mat imA = cv::imread("../rgb-d4/1/Image0001.png");
	cv::Mat imA_depth = cv::imread("../rgb-d4/1/depth0001.png", CV_LOAD_IMAGE_ANYDEPTH);
	//
	// Load Image B. No depth is needed for this
	cv::Mat imB = cv::imread("../rgb-d4/0036.bmp");

	//
	// Print Info on Images
	cout << "==========INFO ON IMages ===========\n";
	cout << "imA.type " << mat_info(imA) << endl;  
	cout << "imA_depth.type " << mat_info(imA_depth) << endl; 
	cout << "imB.type " << mat_info(imB) << endl; 

	cv::imwrite("./log/A.png", imA);
	cv::imwrite("./log/B.png", imB);
	cv::imwrite("./log/imA_depth.png", imA_depth);

	double min, max;
	cv::minMaxLoc(imA_depth, &min, &max);
	cout << "imA_depth min,max: " << min << " " << max << endl;
	//
	// DONE printing debug info on images
	
	/***********************************************
	*
	*  getting calibration parameters
	*
	* ********************************************/
    cout << "=========== calibration parameters ============\n";
	//
	// Intrinsic Calibration. Obtained from https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
	Eigen::Matrix3d K;
	//double fx = 320.377, fy = 320.377, cx = 322.565, cy = 179.8;
    double fx = 2167.430, fy = 2168.014, cx = 621.946, cy = 404.4;
	K << fx, 0., cx, 0., fy, cy, 0., 0., 1.;
	cout << "K\n" << K << endl;

	Eigen::Matrix<double, 5, 1> D; //d0, d1, d2, d3, d4
	//D << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633;
	//cout << "D\n" << D << endl;


	float  zScaling = 10000;
	cout << "zScaling = " << zScaling << endl;
	/***********************************************
	*
	*  Get 3D points using the depth image and
	*  intrinsic matrix
	* ********************************************/
	//
	// Get 3D points of imA as a 4xN matrix. ^aX. These 3d points are in frame-of-ref of imA.
    cout << "=========== Get 3D points ============\n";
	Eigen::MatrixXd a_X;
	get_aX(imA, imA_depth, K, zScaling, a_X);
	cout << "Total number of Edge points = " << a_X.cols() << endl;
	cv::Mat edgePointcloud = cv::Mat(a_X.cols(), 3, CV_32FC1);
	for (int i = 0; i < a_X.cols(); i++)
	{
		edgePointcloud.at<float>(i, 0) = a_X(0, i);
		edgePointcloud.at<float>(i, 1) = a_X(1, i);
		edgePointcloud.at<float>(i, 2) = a_X(2, i);
	}
	SavePointCloudToObj("./log/sceneEdgeCloud.obj", edgePointcloud, CV_32FC1);


	Eigen::MatrixXd all_a_X;
	get_AllaX(imA, imA_depth, K, zScaling, all_a_X);
	cout << "Total number of points = " << all_a_X.cols() << endl;
	cv::Mat pointcloud = cv::Mat(all_a_X.cols(), 3, CV_32FC1);
	for (int i = 0; i < all_a_X.cols(); i++)
	{
		pointcloud.at<float>(i, 0) = all_a_X(0, i);
		pointcloud.at<float>(i, 1) = all_a_X(1, i);
		pointcloud.at<float>(i, 2) = all_a_X(2, i);
	}
	SavePointCloudToObj("./log/sceneCloud.obj", pointcloud, CV_32FC1);
	/***********************************************
	*
	* Get Distance Transform
	*
	* ********************************************/
	//
	// Distance Transform of edges of imB
    cout << "=========== Distance Transform ============\n";
	cv::Mat disTrans;
	get_distance_transform2(imB, disTrans);
	// cv::imshow("Distance Transform Image", disTrans); //numbers between 0 and 1.
	cv::imwrite("./log/Distance_Transform_Image.png", disTrans * 255);

	Eigen::MatrixXd e_disTrans;
	cv::cv2eigen(disTrans, e_disTrans);

	/***********************************************
	*
	*  Verification
	*
	* ********************************************/
#if 0
	//
	// Verify if everything is OK.
	// Use the 3d points are reproject of imA and overlay those reprojected points.
    cout << "=========== Verification ============\n";
	cout << "a_X\n" << a_X.leftCols(10) << endl;
	for (int ss = 0; ss<150; ss += 10)
	{
		cout << ":::::::" << ss << " to " << ss + 10 << endl;
		cout << a_X.block(0, ss, 4, 10) << endl;
	}

	Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
	Eigen::MatrixXd b_u_rep;
	reproject(a_X, M, K, b_u_rep);
	cout << "b_u\n" << b_u_rep.leftCols(10) << endl;
	// s_overlay( imB, b_u_rep );


#endif


	/***********************************************
	*
	*  Initial Guess
	*
	* ********************************************/
	// Initial Guess
    cout << "===========  Initial Guess ============\n";
	Eigen::Matrix4d b_T_a_optvar = Eigen::Matrix4d::Identity();

	cout << "Initial Guess : " << PoseManipUtils::prettyprintMatrix4d(b_T_a_optvar) << endl;
	Eigen::MatrixXd b_u;
	reproject(a_X, b_T_a_optvar, K, b_u);
	s_overlay(imB, b_u, "./log/initial.png");

	/***********************************************
	*
	*  Solve
	*
	* ********************************************/
	////////////////////////////////////////////////////////////////////////////
	///////////////////// Setup non-linear Least Squares ///////////////////////
	////////////////////////////////////////////////////////////////////////////
    cout << "===========  Solve ============\n";
	// Using `a_X` and `disTrans` setup the non-linear least squares problem
	cout << "e_disTrans.shape = " << e_disTrans.rows() << ", " << e_disTrans.cols() << endl;
	ceres::Grid2D<double, 1> grid(e_disTrans.data(), 0, e_disTrans.cols(), 0, e_disTrans.rows());
	ceres::BiCubicInterpolator< ceres::Grid2D<double, 1> > interpolated_imb_disTrans(grid);

	double b_quat_a[10], b_t_a[10]; // quaternion, translation
	PoseManipUtils::eigenmat_to_raw(b_T_a_optvar, b_quat_a, b_t_a);

	// Residues for each 3d points
	ceres::Problem problem;
	int count = 0;
	for (int i = 0; i< a_X.cols(); i += 10)
	{
		// ceres::CostFunction * cost_function = EAResidue::Create( K, a_X.col(i), interpolated_imb_disTrans);

		ceres::CostFunction * cost_function = EAResidue::Create(fx, fy, cx, cy, a_X(0, i), a_X(1, i), a_X(2, i), interpolated_imb_disTrans);
		problem.AddResidualBlock(cost_function, new CauchyLoss(1.), b_quat_a, b_t_a);
		count++;
	}
	std::cout << "-----> Use Point count = " << count << "\n";

	ceres::LocalParameterization * quaternion_parameterization = new ceres::QuaternionParameterization;
	problem.SetParameterization(b_quat_a, quaternion_parameterization);

	auto start1 = std::chrono::high_resolution_clock::now();
	// Run
	ceres::Solver::Options options;
	options.minimizer_progress_to_stdout = true;
	options.linear_solver_type = ceres::DENSE_QR;
	Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	auto finish1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed1 = finish1 - start1;
	double time = elapsed1.count();
	std::cout << "solve done" << ". " << "time = " << time << "\n";

	std::cout << summary.FullReport() << "\n";

	/***********************************************
	*
	*  Result
	*
	* ********************************************/
     cout << "===========  Result ============\n";
	PoseManipUtils::raw_to_eigenmat(b_quat_a, b_t_a, b_T_a_optvar);
	cout << "Final Guess : " << PoseManipUtils::prettyprintMatrix4d(b_T_a_optvar) << endl;

	reproject(a_X, b_T_a_optvar, K, b_u);
	s_overlay(imB, b_u, "./log/final.png");
}

// my test, stereo minimization
int edge_align_test3()
{
	/**********************************************
	*
	* loading rgb and depth images
	*
	* ****************************************** */
	std::string rootpath = "../stereo1/";

	// Load Image A and its Depth image
	cv::Mat imA = cv::imread(rootpath  + "l/Image0001.png");
	cv::Mat imA_depth = cv::imread(rootpath + "l/depth0001.png", CV_LOAD_IMAGE_ANYDEPTH);

	cv::Mat imA2 = cv::imread(rootpath + "r/Image0001.png");
	cv::Mat imA2_depth = cv::imread(rootpath + "r/depth0001.png", CV_LOAD_IMAGE_ANYDEPTH);

	//
	// Load Image B. No depth is needed for this
	cv::Mat imB = cv::imread(rootpath + "/Cam_0_0.bmp");
	cv::Mat imB2 = cv::imread(rootpath + "/Cam_1_0.bmp");
	//
	// Print Info on Images
	cout << "==========INFO ON IMages ===========\n";
	cout << "imA.type " << mat_info(imA) << endl;
	cout << "imA_depth.type " << mat_info(imA_depth) << endl;

	cout << "imA2.type " << mat_info(imA2) << endl;
	cout << "imA2_depth.type " << mat_info(imA2_depth) << endl;
	cout << "imB.type " << mat_info(imB) << endl;
	cout << "imB2.type " << mat_info(imB2) << endl;

	cv::imwrite("./log/A.png", imA);
	cv::imwrite("./log/A2.png", imA2);
	cv::imwrite("./log/B.png", imB);
	cv::imwrite("./log/B2.png", imB2);
	cv::imwrite("./log/imA_depth.png", imA_depth);
	cv::imwrite("./log/imA2_depth.png", imA2_depth);

	double min, max;
	cv::minMaxLoc(imA_depth, &min, &max);
	cout << "imA_depth min,max: " << min << " " << max << endl;

	cv::minMaxLoc(imA2_depth, &min, &max);
	cout << "imA2_depth min,max: " << min << " " << max << endl;
	//
	// DONE printing debug info on images

	/***********************************************
	*
	*  getting calibration parameters
	*
	* ********************************************/
	cout << "=========== calibration parameters ============\n";
	//
	// Intrinsic Calibration. Obtained from https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
	Eigen::Matrix3d K;
	double fx = 2536.420, fy = 2534.961, cx = 640, cy = 512;
	K << fx, 0., cx, 0., fy, cy, 0., 0., 1.;
	cout << "K\n" << K << endl;

	Eigen::Matrix3d K2;
	double fx2 = 2543.018, fy2 = 2542.821, cx2 = 640, cy2 = 512;
	K2 << fx2, 0., cx2, 0., fy2, cy2, 0., 0., 1.;
	cout << "K2\n" << K2 << endl;


	double trans_1to2[16];
	// R
	trans_1to2[0] = 9.3245144047860473e-01;
	trans_1to2[1] = -2.6234356966398651e-03;
	trans_1to2[2] = -3.6128579924281645e-01;

	trans_1to2[4] = -4.1590017795670826e-03;
	trans_1to2[5] = 9.9982944086546910e-01;
	trans_1to2[6] = -1.7994218039162582e-02;

	trans_1to2[8] = 3.6127138532351633e-01;
	trans_1to2[9] = 1.8281322812886434e-02;
	trans_1to2[10] = 9.3228149149419370e-01;

	// T
	trans_1to2[3] = 1.4711195376504131e+02;
	trans_1to2[7] = -5.2977913013145550e-01;
	trans_1to2[11] = 2.0838117412886817e+01;
	
	// last row
	trans_1to2[12] = 0.0;
	trans_1to2[13] = 0.0;
	trans_1to2[14] = 0.0;
	trans_1to2[15] = 1.0;

	Eigen::Matrix<double, 4, 4> TransformFromFirstCam;
	TransformFromFirstCam(0, 0) = trans_1to2[0];
	TransformFromFirstCam(0, 1) = trans_1to2[1];
	TransformFromFirstCam(0, 2) = trans_1to2[2];

	TransformFromFirstCam(1, 0) = trans_1to2[4];
	TransformFromFirstCam(1, 1) = trans_1to2[5];
	TransformFromFirstCam(1, 2) = trans_1to2[6];

	TransformFromFirstCam(2, 0) = trans_1to2[8];
	TransformFromFirstCam(2, 1) = trans_1to2[9];
	TransformFromFirstCam(2, 2) = trans_1to2[10];

	TransformFromFirstCam(0, 3) = trans_1to2[3];
	TransformFromFirstCam(1, 3) = trans_1to2[7];
	TransformFromFirstCam(2, 3) = trans_1to2[11];

	TransformFromFirstCam(3, 0) = trans_1to2[12];
	TransformFromFirstCam(3, 1) = trans_1to2[13];
	TransformFromFirstCam(3, 2) = trans_1to2[14];
	TransformFromFirstCam(3, 3) = trans_1to2[15];

	Eigen::Matrix<double, 4, 4> TransformFromFirstCamInv;
	Eigen::Matrix<double, 3, 3> rot_transpose = TransformFromFirstCam.topLeftCorner(3, 3).transpose();
	TransformFromFirstCamInv.topLeftCorner(3, 3) = rot_transpose;
	Eigen::Matrix<double, 3, 1> reverse_t = -rot_transpose * TransformFromFirstCam.topRightCorner(3, 1);
	TransformFromFirstCamInv(0, 3) = reverse_t[0];
	TransformFromFirstCamInv(1, 3) = reverse_t[1];
	TransformFromFirstCamInv(2, 3) = reverse_t[2];
	TransformFromFirstCamInv(3, 3) = 1.0;

	double trans_1to2_inv[16];
	// R
	trans_1to2_inv[0] = TransformFromFirstCamInv(0, 0);
	trans_1to2_inv[1] = TransformFromFirstCamInv(0, 1);
	trans_1to2_inv[2] = TransformFromFirstCamInv(0, 2);

	trans_1to2_inv[4] = TransformFromFirstCamInv(1, 0);
	trans_1to2_inv[5] = TransformFromFirstCamInv(1, 1);
	trans_1to2_inv[6] = TransformFromFirstCamInv(1, 2);

	trans_1to2_inv[8] = TransformFromFirstCamInv(2, 0);
	trans_1to2_inv[9] = TransformFromFirstCamInv(2, 1);
	trans_1to2_inv[10] = TransformFromFirstCamInv(2, 2);

	// T
	trans_1to2_inv[3] = TransformFromFirstCamInv(0, 3);
	trans_1to2_inv[7] = TransformFromFirstCamInv(1, 3);
	trans_1to2_inv[11] = TransformFromFirstCamInv(2, 3);

	// last row
	trans_1to2_inv[12] = TransformFromFirstCamInv(3, 0);
	trans_1to2_inv[13] = TransformFromFirstCamInv(3, 1);
	trans_1to2_inv[14] = TransformFromFirstCamInv(3, 2);
	trans_1to2_inv[15] = TransformFromFirstCamInv(3, 3);

	float  zScaling = 10000;
	cout << "zScaling = " << zScaling << endl;
	/***********************************************
	*
	*  Get 3D points using the depth image and
	*  intrinsic matrix
	* ********************************************/
	//
	// Get 3D points of imA as a 4xN matrix. ^aX. These 3d points are in frame-of-ref of imA.
	cout << "=========== Get 3D points ============\n";
	Eigen::MatrixXd a_X;
	get_aX(imA, imA_depth, K, zScaling, a_X);
	cout << "Total number of Edge points = " << a_X.cols() << endl;

	Eigen::MatrixXd all_a_X;
	get_AllaX(imA, imA_depth, K, zScaling, all_a_X);
	cout << "Total number of points = " << all_a_X.cols() << endl;

	{
		cv::Mat edgePointcloud = cv::Mat(a_X.cols(), 3, CV_32FC1);
		for (int i = 0; i < a_X.cols(); i++)
		{
			edgePointcloud.at<float>(i, 0) = a_X(0, i);
			edgePointcloud.at<float>(i, 1) = a_X(1, i);
			edgePointcloud.at<float>(i, 2) = a_X(2, i);
		}
		SavePointCloudToObj("./log/sceneEdgeCloud.obj", edgePointcloud, CV_32FC1);

		cv::Mat pointcloud = cv::Mat(all_a_X.cols(), 3, CV_32FC1);
		for (int i = 0; i < all_a_X.cols(); i++)
		{
			pointcloud.at<float>(i, 0) = all_a_X(0, i);
			pointcloud.at<float>(i, 1) = all_a_X(1, i);
			pointcloud.at<float>(i, 2) = all_a_X(2, i);
		}
		SavePointCloudToObj("./log/sceneCloud.obj", pointcloud, CV_32FC1);
	}

	////////////////
	cout << "=========== Get 3D points ============\n";
	Eigen::MatrixXd a_X2;
	get_aX(imA2, imA2_depth, K, zScaling, a_X2);
	cout << "Total number of Edge points = " << a_X.cols() << endl;

	Eigen::MatrixXd all_a_X2;
	get_AllaX(imA2, imA2_depth, K2, zScaling, all_a_X2);
	cout << "Total number of points = " << all_a_X2.cols() << endl;

	{
		cv::Mat edgePointcloud = cv::Mat(a_X2.cols(), 3, CV_32FC1);
		for (int i = 0; i < a_X2.cols(); i++)
		{
			edgePointcloud.at<float>(i, 0) = a_X2(0, i);
			edgePointcloud.at<float>(i, 1) = a_X2(1, i);
			edgePointcloud.at<float>(i, 2) = a_X2(2, i);
		}
		SavePointCloudToObj("./log/sceneEdgeCloud2.obj", edgePointcloud, CV_32FC1);

		cv::Mat pointcloud = cv::Mat(all_a_X2.cols(), 3, CV_32FC1);
		for (int i = 0; i < all_a_X2.cols(); i++)
		{
			pointcloud.at<float>(i, 0) = all_a_X2(0, i);
			pointcloud.at<float>(i, 1) = all_a_X2(1, i);
			pointcloud.at<float>(i, 2) = all_a_X2(2, i);
		}
		SavePointCloudToObj("./log/sceneCloud2.obj", pointcloud, CV_32FC1);
	}
	/***********************************************
	*
	* Get Distance Transform
	*
	* ********************************************/
	//
	// Distance Transform of edges of imB
	cout << "=========== Distance Transform ============\n";
	cv::Mat disTrans;
	get_distance_transform2(imB, disTrans);
	cv::imwrite("./log/Distance_Transform_Image.png", disTrans * 255);

	cv::Mat disTrans2;
	get_distance_transform2(imB2, disTrans2);
	cv::imwrite("./log/Distance_Transform_Image2.png", disTrans2 * 255);

	Eigen::MatrixXd e_disTrans;
	cv::cv2eigen(disTrans, e_disTrans);

	Eigen::MatrixXd e_disTrans2;
	cv::cv2eigen(disTrans2, e_disTrans2);
	/***********************************************
	*
	*  Initial Guess
	*
	* ********************************************/
	// Initial Guess
	cout << "===========  Initial Guess ============\n";
	Eigen::Matrix4d b_T_a_optvar = Eigen::Matrix4d::Identity();

	cout << "Initial Guess : " << PoseManipUtils::prettyprintMatrix4d(b_T_a_optvar) << endl;
	Eigen::MatrixXd b_u;
	reproject(a_X, b_T_a_optvar, K, b_u);
	s_overlay(imB, b_u, "./log/initial.png");

	Eigen::Matrix4d b_T_a_optvar2 = Eigen::Matrix4d::Identity();
	Eigen::MatrixXd b_u2;
	reproject(a_X2, b_T_a_optvar2, K, b_u2);
	s_overlay(imB2, b_u2, "./log/initial2.png");


	/***********************************************
	*
	*  Solve
	*
	* ********************************************/
	////////////////////////////////////////////////////////////////////////////
	///////////////////// Setup non-linear Least Squares ///////////////////////
	////////////////////////////////////////////////////////////////////////////
	cout << "===========  Solve ============\n";
	// Using `a_X` and `disTrans` setup the non-linear least squares problem
	cout << "e_disTrans.shape = " << e_disTrans.rows() << ", " << e_disTrans.cols() << endl;
	ceres::Grid2D<double, 1> grid(e_disTrans.data(), 0, e_disTrans.cols(), 0, e_disTrans.rows());
	ceres::BiCubicInterpolator< ceres::Grid2D<double, 1> > interpolated_imb_disTrans(grid);

	cout << "e_disTrans2.shape = " << e_disTrans2.rows() << ", " << e_disTrans2.cols() << endl;
	ceres::Grid2D<double, 1> grid2(e_disTrans2.data(), 0, e_disTrans2.cols(), 0, e_disTrans2.rows());
	ceres::BiCubicInterpolator< ceres::Grid2D<double, 1> > interpolated_imb_disTrans2(grid2);

	double b_quat_a[10], b_t_a[10]; // quaternion, translation
	PoseManipUtils::eigenmat_to_raw(b_T_a_optvar, b_quat_a, b_t_a);

	// Residues for each 3d points
	ceres::Problem problem;
	int count = 0;
	for (int i = 0; i< a_X.cols(); i += 10)
	{
		ceres::CostFunction * cost_function = EAResidue::Create(fx, fy, cx, cy, a_X(0, i), a_X(1, i), a_X(2, i), interpolated_imb_disTrans);
		problem.AddResidualBlock(cost_function, new CauchyLoss(1.), b_quat_a, b_t_a);
		count++;
	}

	for (int i = 0; i< a_X2.cols(); i += 10)
	{
		ceres::CostFunction * cost_function = EAResidueSecondCam::Create(fx2, fy2, cx2, cy2, a_X2(0, i), a_X2(1, i), a_X2(2, i), trans_1to2, trans_1to2_inv, interpolated_imb_disTrans2);
		problem.AddResidualBlock(cost_function, new CauchyLoss(1.), b_quat_a, b_t_a);
		count++;
	}
	std::cout << "-----> Use Point count = " << count << "\n";

	ceres::LocalParameterization * quaternion_parameterization = new ceres::QuaternionParameterization;
	problem.SetParameterization(b_quat_a, quaternion_parameterization);

	auto start1 = std::chrono::high_resolution_clock::now();
	// Run
	ceres::Solver::Options options;
	options.minimizer_progress_to_stdout = true;
	options.linear_solver_type = ceres::DENSE_QR;
	Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	auto finish1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed1 = finish1 - start1;
	double time = elapsed1.count();
	std::cout << "solve done" << ". " << "time = " << time << "\n";

	std::cout << summary.FullReport() << "\n";

	/***********************************************
	*
	*  Result
	*
	* ********************************************/
	cout << "===========  Result ============\n";
	PoseManipUtils::raw_to_eigenmat(b_quat_a, b_t_a, b_T_a_optvar);
	cout << "Final Guess : " << PoseManipUtils::prettyprintMatrix4d(b_T_a_optvar) << endl;

	reproject(a_X, b_T_a_optvar, K, b_u);
	s_overlay(imB, b_u, "./log/final.png");

	reproject(a_X2, b_T_a_optvar, TransformFromFirstCam, TransformFromFirstCamInv, K2, b_u2);
	s_overlay(imB2, b_u2, "./log/final2.png");


	{
		Eigen::MatrixXd b_X = b_T_a_optvar * a_X;

		cv::Mat edgePointcloud = cv::Mat(b_X.cols(), 3, CV_32FC1);
		for (int i = 0; i < b_X.cols(); i++)
		{
			edgePointcloud.at<float>(i, 0) = b_X(0, i);
			edgePointcloud.at<float>(i, 1) = b_X(1, i);
			edgePointcloud.at<float>(i, 2) = b_X(2, i);
		}
		SavePointCloudToObj("./log/sceneEdgeCloud_Final.obj", edgePointcloud, CV_32FC1);

		Eigen::MatrixXd b_X2 = TransformFromFirstCam * b_T_a_optvar * TransformFromFirstCamInv * a_X2;

		cv::Mat edgePointcloud2 = cv::Mat(b_X2.cols(), 3, CV_32FC1);
		for (int i = 0; i < b_X2.cols(); i++)
		{
			edgePointcloud2.at<float>(i, 0) = b_X2(0, i);
			edgePointcloud2.at<float>(i, 1) = b_X2(1, i);
			edgePointcloud2.at<float>(i, 2) = b_X2(2, i);
		}
		SavePointCloudToObj("./log/sceneEdgeCloud_Final2.obj", edgePointcloud2, CV_32FC1);
	}
}


int main()
{
	//edge_align_test1();
	//edge_align_test2();
	edge_align_test3();
    std::cout << "press ENTER to continue...\n";
    getchar();
    return 0;
}