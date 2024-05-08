// amp_eigen_tensor.h

#ifndef AMP_EIGEN_TENSOR_H
#define AMP_EIGEN_TENSOR_H

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

Eigen::MatrixXd wigerdjx(const double &s, const int &lens);
// wignerd(s,theta=pi/2) # lens=2s+1;
Eigen::MatrixXcd wigerDx(const int &i, const Eigen::MatrixXd &wigerdj, const double &alpha, const double &beta, const double &gamma);
// wignerD(j, alpha, beta, gamma) # i=2j+1;
// input:wigerdj=wignerd(j,theta=pi/2)
double cgcoeff(const double &a, const double &m1, const double &b, const double &m2, const double &c, const double &m);
// cgcoeff(a,m1,b,m2,c,m) # <a,m1,b,m2|c,m>
std::complex<double> sphericalHarmonic(const int &l, const int &m, const double &theta, const double &phi);
// sphericalHarmonic(l,m,theta,phi) # Ylm(theta,phi)
void xyzToangle(const double &px, const double &py, const double &pz, double &theta, double &phi);
// xyzToangle(px,py,pz,theta,phi) # Convert unit space vector(px,py,pz) to angles in spherical coordinates(theta,phi)
Eigen::Tensor<std::complex<double>, 3> PWFA(Eigen::Vector4d &p1, const double &mu1, const double &s1, Eigen::Vector4d &p2, const double &mu2, const double &s2, const double &s, const double &S, const int &L);
// PWFA(p1,mu1,s1,p2,mu2,s2,s,S,L) # s1,s2,s,S are spin, L is orbital angular momentum
Eigen::Tensor<std::complex<double>, 3> PWFB(Eigen::Vector4d &p1, const double &mu1, const double &s1, Eigen::Vector4d &p2, const double &mu2, const double &s2, const double &s, const double &S, const int &L);
// PWFB(p1,mu1,s1,p2,mu2,s2,s,S,L) # s1,s2,s,S are spin, L is orbital angular momentum
#endif