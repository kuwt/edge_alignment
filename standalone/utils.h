#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

// Eigen
#include <Eigen/Dense>

// ceres
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>

string type2str(int type);
string mat_info( const cv::Mat& im );

void get_aX( const cv::Mat& imA, const cv::Mat& imA_depth, const Eigen::Matrix3d& K, const float zScaling, Eigen::MatrixXd& a_X );
void get_aX_mask( const cv::Mat& imA, const cv::Mat& mask, const cv::Mat& imA_depth, const Eigen::Matrix3d& K, const float zScaling, Eigen::MatrixXd& a_X );
void get_aX_canny(const cv::Mat& imA, const cv::Mat& imA_depth, const Eigen::Matrix3d& K, const float zScaling, Eigen::MatrixXd& a_X);

void get_AllaX(const cv::Mat& imA, const cv::Mat& imA_depth, const Eigen::Matrix3d& K, const float zScaling, Eigen::MatrixXd& a_X);
void get_distance_transform( const cv::Mat& input, cv::Mat& out_distance_transform );
void get_distance_transform2(const cv::Mat& input, cv::Mat& out_distance_transform);

void reproject( const Eigen::MatrixXd& a_X, const Eigen::Matrix4d& b_T_a, const Eigen::Matrix3d& K, Eigen::MatrixXd& b_u );
void reproject(const Eigen::MatrixXd& a_X, const Eigen::Matrix4d& b_T_a, const Eigen::Matrix4d& one_T_two, const Eigen::Matrix4d& two_T_one, const Eigen::Matrix3d& K, Eigen::MatrixXd& b_u);
void s_overlay( const cv::Mat& im, const Eigen::MatrixXd& uv, const char * win_name);


class EAResidue {
public:
    EAResidue(
        const double fx, const double fy, const double cx, const double cy,
        const Eigen::Vector4d& __a_X,
        const ceres::BiCubicInterpolator<ceres::Grid2D<double,1>>& __interpolated_a
    ): fx(fx), fy(fy), cx(cx), cy(cy),  a_X(__a_X), interp_a(__interpolated_a)
    {
        // cout << "---\n";
        // cout << "EAResidue.a_X: "<< a_X << endl;
        // cout << "fx=" << fx << "fy=" << fy << "cx=" << cx << "cy=" << cy << endl;

    }

    EAResidue(
        const double fx, const double fy, const double cx, const double cy,
        const double a_Xx, const double a_Xy, const double a_Xz,
        const ceres::BiCubicInterpolator<ceres::Grid2D<double,1>>& __interpolated_a
    ): fx(fx), fy(fy), cx(cx), cy(cy),   a_Xx(a_Xx),a_Xy(a_Xy),a_Xz(a_Xz), interp_a(__interpolated_a)
    {}

    template <typename T>
    bool operator()( const T* const quat, const T* const t, T* residue ) const {
        // b_quat_a, b_t_a to b_T_a
        Eigen::Quaternion<T> eigen_q( quat[0], quat[1], quat[2], quat[3] );
        Eigen::Matrix<T,4,4> b_T_a = Eigen::Matrix<T,4,4>::Zero();
        b_T_a.topLeftCorner(3,3) = eigen_q.toRotationMatrix();
        b_T_a(0,3) = t[0];
        b_T_a(1,3) = t[1];
        b_T_a(2,3) = t[2];
        b_T_a(3,3) =  T(1.0);



        // transform a_X
        Eigen::Matrix<T,4,1> b_X;
        Eigen::Matrix<T,4,1> templaye_a_X;
        // templaye_a_X << T(a_X(0)),T(a_X(1)),T(a_X(2)),T(1.0);
        // cout << "{{{{{{{{}}}}}}}}" << a_X << endl;
        // templaye_a_X(0) = T(a_X(0));
        // templaye_a_X(1) = T(a_X(1));
        // templaye_a_X(2) = T(2.0); //T(a_X(2));
        // templaye_a_X(3) = T(a_X(3));

        // cout << "{{{{{{{{}}}}}}}}" << a_Xx << ","<< a_Xy << ","<< a_Xz << "," << endl;
        templaye_a_X(0) = T(a_Xx);
        templaye_a_X(1) = T(a_Xy);
        templaye_a_X(2) = T(a_Xz);
        templaye_a_X(3) = T(1.0);
        b_X = b_T_a *templaye_a_X;


        // Perspective-Projection and scaling with K.
        if( b_X(2) < T(0.01) && b_X(2) > T(-0.01) )
            return false;
        T _u = T(fx) * b_X(0)/b_X(2) + T(cx);
        T _v = T(fy) * b_X(1)/b_X(2) + T(cy);


        // double __r;
        interp_a.Evaluate( _u, _v, &residue[0] );
        // residue[0] = _u*_u + _v*_v;

        // residue[0] = b_X(0) - t[0] ;//+ b_X(1)*b_X(1); //T(__r) ;
        return true;
    }

    static ceres::CostFunction* Create(
        const double fx, const double fy, const double cx, const double cy,
        //const Eigen::Vector4d __a_X,
        //const Eigen::Ref<const VectorXd>& __a_X,
        const double a_Xx, const double a_Xy, const double a_Xz,
        const ceres::BiCubicInterpolator<ceres::Grid2D<double,1>>& __interpolated_a  )
    {
        return( new ceres::AutoDiffCostFunction<EAResidue,1,4,3>
            (
                new EAResidue( fx, fy, cx, cy, a_Xx,a_Xy,a_Xz,__interpolated_a)
            )
            );
    }

private:
    const Eigen::Vector4d& a_X; // a_X
    const ceres::BiCubicInterpolator<ceres::Grid2D<double,1>>& interp_a; //
    // const Eigen::Matrix3d& K;
    double fx, fy, cx, cy;
    double a_Xx, a_Xy, a_Xz;

};

class EAResidueSecondCam {
public:
	EAResidueSecondCam(
		const double fx, const double fy, const double cx, const double cy,
		const Eigen::Vector4d& __a_X,
		const double *ptrans_1to2,
		const double *ptrans_1to2_inv,
		const ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>& __interpolated_a
	) : fx(fx), fy(fy), cx(cx), cy(cy), a_X(__a_X), interp_a(__interpolated_a)
	{
		// cout << "---\n";
		// cout << "EAResidue.a_X: "<< a_X << endl;
		// cout << "fx=" << fx << "fy=" << fy << "cx=" << cx << "cy=" << cy << endl;
		TransformFromFirstCam(0, 0) = ptrans_1to2[0];
		TransformFromFirstCam(0, 1) = ptrans_1to2[1];
		TransformFromFirstCam(0, 2) = ptrans_1to2[2];

		TransformFromFirstCam(1, 0) = ptrans_1to2[4];
		TransformFromFirstCam(1, 1) = ptrans_1to2[5];
		TransformFromFirstCam(1, 2) = ptrans_1to2[6];

		TransformFromFirstCam(2, 0) = ptrans_1to2[8];
		TransformFromFirstCam(2, 1) = ptrans_1to2[9];
		TransformFromFirstCam(2, 2) = ptrans_1to2[10];

		TransformFromFirstCam(0, 3) = ptrans_1to2[3];
		TransformFromFirstCam(1, 3) = ptrans_1to2[7];
		TransformFromFirstCam(2, 3) = ptrans_1to2[11];

		TransformFromFirstCam(3, 0) = ptrans_1to2[12];
		TransformFromFirstCam(3, 1) = ptrans_1to2[13];
		TransformFromFirstCam(3, 2) = ptrans_1to2[14];
		TransformFromFirstCam(3, 3) = ptrans_1to2[15];


		//
		TransformFromFirstCamInv(0, 0) = ptrans_1to2_inv[0];
		TransformFromFirstCamInv(0, 1) = ptrans_1to2_inv[1];
		TransformFromFirstCamInv(0, 2) = ptrans_1to2_inv[2];

		TransformFromFirstCamInv(1, 0) = ptrans_1to2_inv[4];
		TransformFromFirstCamInv(1, 1) = ptrans_1to2_inv[5];
		TransformFromFirstCamInv(1, 2) = ptrans_1to2_inv[6];

		TransformFromFirstCamInv(2, 0) = ptrans_1to2_inv[8];
		TransformFromFirstCamInv(2, 1) = ptrans_1to2_inv[9];
		TransformFromFirstCamInv(2, 2) = ptrans_1to2_inv[10];

		TransformFromFirstCamInv(0, 3) = ptrans_1to2_inv[3];
		TransformFromFirstCamInv(1, 3) = ptrans_1to2_inv[7];
		TransformFromFirstCamInv(2, 3) = ptrans_1to2_inv[11];

		TransformFromFirstCamInv(3, 0) = ptrans_1to2_inv[12];
		TransformFromFirstCamInv(3, 1) = ptrans_1to2_inv[13];
		TransformFromFirstCamInv(3, 2) = ptrans_1to2_inv[14];
		TransformFromFirstCamInv(3, 3) = ptrans_1to2_inv[15];

		
	}

	EAResidueSecondCam(
		const double fx, const double fy, const double cx, const double cy,
		const double a_Xx, const double a_Xy, const double a_Xz,
		const double *ptrans_1to2,
		const double *ptrans_1to2_inv,
		const ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>& __interpolated_a
	) : fx(fx), fy(fy), cx(cx), cy(cy), a_Xx(a_Xx), a_Xy(a_Xy), a_Xz(a_Xz), interp_a(__interpolated_a)
	{
		TransformFromFirstCam(0, 0) = ptrans_1to2[0];
		TransformFromFirstCam(0, 1) = ptrans_1to2[1];
		TransformFromFirstCam(0, 2) = ptrans_1to2[2];

		TransformFromFirstCam(1, 0) = ptrans_1to2[4];
		TransformFromFirstCam(1, 1) = ptrans_1to2[5];
		TransformFromFirstCam(1, 2) = ptrans_1to2[6];

		TransformFromFirstCam(2, 0) = ptrans_1to2[8];
		TransformFromFirstCam(2, 1) = ptrans_1to2[9];
		TransformFromFirstCam(2, 2) = ptrans_1to2[10];

		TransformFromFirstCam(0, 3) = ptrans_1to2[3];
		TransformFromFirstCam(1, 3) = ptrans_1to2[7];
		TransformFromFirstCam(2, 3) = ptrans_1to2[11];

		TransformFromFirstCam(3, 0) = ptrans_1to2[12];
		TransformFromFirstCam(3, 1) = ptrans_1to2[13];
		TransformFromFirstCam(3, 2) = ptrans_1to2[14];
		TransformFromFirstCam(3, 3) = ptrans_1to2[15];
		
		//
		TransformFromFirstCamInv(0, 0) = ptrans_1to2_inv[0];
		TransformFromFirstCamInv(0, 1) = ptrans_1to2_inv[1];
		TransformFromFirstCamInv(0, 2) = ptrans_1to2_inv[2];

		TransformFromFirstCamInv(1, 0) = ptrans_1to2_inv[4];
		TransformFromFirstCamInv(1, 1) = ptrans_1to2_inv[5];
		TransformFromFirstCamInv(1, 2) = ptrans_1to2_inv[6];

		TransformFromFirstCamInv(2, 0) = ptrans_1to2_inv[8];
		TransformFromFirstCamInv(2, 1) = ptrans_1to2_inv[9];
		TransformFromFirstCamInv(2, 2) = ptrans_1to2_inv[10];

		TransformFromFirstCamInv(0, 3) = ptrans_1to2_inv[3];
		TransformFromFirstCamInv(1, 3) = ptrans_1to2_inv[7];
		TransformFromFirstCamInv(2, 3) = ptrans_1to2_inv[11];

		TransformFromFirstCamInv(3, 0) = ptrans_1to2_inv[12];
		TransformFromFirstCamInv(3, 1) = ptrans_1to2_inv[13];
		TransformFromFirstCamInv(3, 2) = ptrans_1to2_inv[14];
		TransformFromFirstCamInv(3, 3) = ptrans_1to2_inv[15];
	}

	template <typename T>
	bool operator()(const T* const quat, const T* const t, T* residue) const {
		// b_quat_a, b_t_a to b_T_a
		Eigen::Quaternion<T> eigen_q(quat[0], quat[1], quat[2], quat[3]);
		Eigen::Matrix<T, 4, 4> b_T_a = Eigen::Matrix<T, 4, 4>::Zero();
		b_T_a.topLeftCorner(3, 3) = eigen_q.toRotationMatrix();
		b_T_a(0, 3) = t[0];
		b_T_a(1, 3) = t[1];
		b_T_a(2, 3) = t[2];
		b_T_a(3, 3) = T(1.0);

		Eigen::Matrix<T, 4, 4> b_T_a_SecCam = Eigen::Matrix<T, 4, 4>::Zero();
		b_T_a_SecCam = TransformFromFirstCam * b_T_a * TransformFromFirstCamInv;

		// transform a_X
		Eigen::Matrix<T, 4, 1> b_X;
		Eigen::Matrix<T, 4, 1> templaye_a_X;

		templaye_a_X(0) = T(a_Xx);
		templaye_a_X(1) = T(a_Xy);
		templaye_a_X(2) = T(a_Xz);
		templaye_a_X(3) = T(1.0);


		b_X = b_T_a_SecCam * templaye_a_X;


		// Perspective-Projection and scaling with K.
		if (b_X(2) < T(0.01) && b_X(2) > T(-0.01))
		{
			return false;
		}
		T _u = T(fx) * b_X(0) / b_X(2) + T(cx);
		T _v = T(fy) * b_X(1) / b_X(2) + T(cy);


		// double __r;
		interp_a.Evaluate(_u, _v, &residue[0]);

		return true;
	}

	static ceres::CostFunction* Create(
		const double fx, const double fy, const double cx, const double cy,
		const double a_Xx, const double a_Xy, const double a_Xz,
		const double *ptrans_1to2,
		const double *ptrans_1to2_inv,
		const ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>& __interpolated_a)
	{
		return(new ceres::AutoDiffCostFunction<EAResidueSecondCam, 1, 4, 3>
			(
				new EAResidueSecondCam(fx, fy, cx, cy, a_Xx, a_Xy, a_Xz, ptrans_1to2, ptrans_1to2_inv, __interpolated_a)
				)
			);
	}

private:
	const Eigen::Vector4d& a_X; // a_X
	const ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>& interp_a; //
																		  // const Eigen::Matrix3d& K;
	double fx, fy, cx, cy;
	double a_Xx, a_Xy, a_Xz;

	Eigen::Matrix<double, 4, 4> TransformFromFirstCam;
	Eigen::Matrix<double, 4, 4> TransformFromFirstCamInv;
};
