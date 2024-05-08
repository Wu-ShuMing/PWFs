#include "amp_eigen_tensor.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

double fac1(const int &a, const int &b)
{
    double result = 1.0;
    for (int i = b + 1; i <= a; ++i)
    {
        result *= i;
    }
    return result;
}
double fac2(const int &a, const int &b, const int &c)
{
    double result = 1.0;
    if (a >= b)
    {
        result = fac1(a, b);
        for (int i = c; i > 1; --i)
        {
            result /= i;
        }
        return result;
    }
    else
    {
        result = fac1(b, a);
        for (int i = c; i > 1; --i)
        {
            result *= i;
        }
        return 1. / result;
    }
}
double fac(const double &j, const double &m1, const double &m2, const int &s)
{
    int xtem1 = j + m1, xtem2 = j + m2, xtem3 = j - m1, xtem4 = j - m2;
    int ytem1 = j + m2 - s, ytem2 = j - m1 - s, ytem3 = m1 - m2 + s, ytem4 = s;
    int xtem[] = {xtem1, xtem2, xtem3, xtem4};
    int ytem[] = {ytem1, ytem2, ytem3, ytem4};
    std::sort(xtem, xtem + 4);
    std::sort(ytem, ytem + 4);
    double tem1 = fac2(xtem[0], ytem[2], ytem[1]);
    double tem2 = fac2(xtem[1], ytem[2], ytem[1]);
    double tem3 = fac2(xtem[2], ytem[3], ytem[0]);
    double tem4 = fac2(xtem[3], ytem[3], ytem[0]);
    double result = tem1 * tem2 * tem3 * tem4;
    return std::sqrt(result);
}
double wigerd0j(const double &j, const double &m1, const double &m2)
{
    const int x1 = m2 - m1, x2 = j + m2, x3 = j - m1;
    const int smin = std::max(0, x1), smax = std::min(x2, x3);
    double ctem = 1.0 / sqrt(2);
    double fs = fac(j, m1, m2, smax);
    int sign = ((smax - x1) & 1) ? -1 : 1;
    double sumtem = sign * fs;
    for (int s = smax - 1; s >= smin; s--)
    {
        sign = ((s - x1) & 1) ? -1 : 1;
        fs = fs * (s + 1) * (s + 1 - x1) / ((x2 - s) * (x3 - s));
        sumtem += sign * fs;
    }
    double coftem = std::pow(ctem, x2 + x3 - x1);
    return coftem * sumtem;
}

Eigen::MatrixXd wigerdjx(const double &s, const int &lens)
{
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(lens, lens);
    for (int sx = 0; sx < lens; sx++)
    {
        for (int sy = 0; sy < lens; sy++)
        {
            result(sx, sy) = wigerd0j(s, sx - s, sy - s);
        }
    }
    return result;
}

Eigen::MatrixXcd wigerDx(const int &i, const Eigen::MatrixXd &wigerdj, const double &alpha, const double &beta, const double &gamma)
{
    Eigen::VectorXcd v1 = Eigen::VectorXcd::Zero(i), v2 = Eigen::VectorXcd::Zero(i), v3 = Eigen::VectorXcd::Zero(i);
    for (int j = 0; j < i; ++j)
    {
        double mj = (2 * j - i + 1) / 2.0;
        std::complex<double> resultx1(std::cos(mj * (M_PI / 2 + alpha)), std::sin(-mj * (M_PI / 2 + alpha)));
        std::complex<double> resultx2(std::cos(mj * (M_PI / 2 - gamma)), std::sin(mj * (M_PI / 2 - gamma)));
        std::complex<double> resultx3(std::cos(mj * (beta / 2)), std::sin(-mj * (beta / 2)));
        v1(j) = resultx1;
        v2(j) = resultx2;
        v3(j) = resultx3;
    }

    Eigen::MatrixXcd mat1 = v1 * v3.transpose();
    Eigen::MatrixXcd mat2 = v2 * v3.transpose();
    Eigen::MatrixXcd result1 = mat1.array() * wigerdj.array();
    Eigen::MatrixXcd result2 = mat2.array() * wigerdj.array();
    Eigen::MatrixXcd result = result1 * result2.transpose();

    return result;
}
double bmcoeff(const unsigned int &n, const unsigned int &k)
{
    double sumtem = 1.0;
    int imax = (k < n - k) ? k : n - k;
    for (int i = 1; i <= imax; ++i)
    {
        sumtem = sumtem * (n - imax + i) / i;
    }
    return sumtem;
}
double cgcoeff(const double &a, const double &m1, const double &b, const double &m2, const double &c, const double &m)
{
    if (a + b < c || b + c < a || a + c < b || std::abs(m1) > a || std::abs(m2) > b || std::abs(m) > c || m1 + m2 != m)
    {
        return 0.0;
    }
    else
    {
        int j = a + b + c;
        double ftem = std::sqrt(bmcoeff(2 * a, j - 2 * c) * bmcoeff(2 * b, j - 2 * c) / bmcoeff(j + 1, j - 2 * c) / bmcoeff(2 * a, a - m1) / bmcoeff(2 * b, b - m2) / bmcoeff(2 * c, c - m));
        int kmin = std::max(0.0, std::max(b - m1 - c, a + m2 - c));
        int kmax = std::min(a + b - c, std::min(a - m1, b + m2));
        int sign = (kmin & 1) ? -1 : 1;
        double fk = bmcoeff(j - 2 * c, kmin) * bmcoeff(j - 2 * b, a - m1 - kmin) * bmcoeff(j - 2 * a, b + m2 - kmin);
        double sumtem = sign * fk;
        for (int k = kmin + 1; k <= kmax; ++k)
        {
            sign = (k & 1) ? -1 : 1;
            fk = fk * (j - 2 * c - k + 1) * (a - m1 - k + 1) * (b + m2 - k + 1) / (k * (j - 2 * b - a + m1 + k) * (j - 2 * a - b - m2 + k));
            sumtem += sign * fk;
        }
        return ftem * sumtem;
    }
}
std::complex<double> sphericalHarmonic(const int &l, const int &m, const double &theta, const double &phi)
{
    int mabs = std::abs(m);
    if (mabs > l)
    {
        return std::complex<double>(0.0, 0.0);
    }
    else
    {
        double legtem = std::sqrt((2 * l + 1) * std::tgamma(l - mabs + 1) / (4 * M_PI * std::tgamma(l + mabs + 1)));
        legtem *= std::assoc_legendre(l, mabs, std::cos(theta));
        int sign = (m >= 0 && m & 1) ? -1 : 1;
        legtem *= sign;
        std::complex<double> result(legtem * std::cos(m * phi), legtem * std::sin(m * phi));
        return result;
    }
}

void xyzToangle(const double &px, const double &py, const double &pz, double &theta, double &phi)
{
    double pxy = px * px + py * py;
    if (pxy <= 1e-16)
    {
        theta = 0.0;
        phi = 0.0;
    }
    else
    {
        theta = std::acos(pz);
        phi = std::atan2(py, px);
        if (phi < 0) {
            phi += 2*M_PI;
        }
    }
}

//

Eigen::Tensor<std::complex<double>, 3> amp0ls(const double &s1, const int &lens1, const double &s2, const int &lens2, const double &s, const int &lens, const double &theta, const double &phi, const double &S, const int &L)
{
    Eigen::Tensor<std::complex<double>, 3> result(lens, lens1, lens2);
    for (int i = 0; i < lens; i++)
    {
        for (int is1 = 0; is1 < lens1; is1++)
        {
            for (int is2 = 0; is2 < lens2; is2++)
            {
                double sig1 = is1 - s1, sig2 = is2 - s2, sigma = i - s;
                std::complex<double> xx = cgcoeff(s1, sig1, s2, sig2, S, sig1 + sig2) * cgcoeff(S, sig1 + sig2, L, sigma - sig1 - sig2, s, sigma) * sphericalHarmonic(L, sigma - sig1 - sig2, theta, -phi);
                result(i, is1, is2) = xx;
            }
        }
    }
    return result;
}

void MassiveTransAngle(Eigen::Vector4d p1, Eigen::Vector4d p2, double &theta, double &phi, double &psi)
{
    Eigen::Vector3d p1xyz = p1.tail(3), p2xyz = p2.tail(3);
    Eigen::Vector3d hat = p1xyz.cross(p2xyz);
    double q1 = p1xyz.norm(), q2 = p2xyz.norm(), mhat = hat.norm();
    std::complex<double> result;
    if (q1 * q2 * mhat <= 1e-16)
    {
        theta = 0.0, phi = 0.0, psi = 0.0;
    }
    else
    {
        double m1 = std::sqrt(p1(0) * p1(0) - q1 * q1);
        double m2 = std::sqrt(p2(0) * p2(0) - q2 * q2);
        double gamma1 = p1(0) / m1, gamma2 = p2(0) / m2;
        double cosx = -p1xyz.dot(p2xyz) / q1 / q2;
        Eigen::Vector3d nhat = hat / (q1 * q2 * std::sqrt(1 - cosx * cosx));
        xyzToangle(nhat(0), nhat(1), nhat(2), theta, phi);
        psi = std::acos(1 - ((1 - cosx * cosx) * (-1 + gamma1) * (-1 + gamma2)) / (1 + (cosx * q1 * q2) / (m1 * m2) + gamma1 * gamma2));
    }
}

void MasslessTransAngle(Eigen::Vector4d p1, Eigen::Vector4d p2, double &psi)
{
    Eigen::Vector3d p1xyz = p1.tail(3), p2xyz = p2.tail(3);
    double q1 = p1xyz.norm(), q2 = p2xyz.norm();
    if (q1 <= 1e-16)
    {
        psi = 0.0;
    }
    else
    {
        Eigen::Vector3d n1 = p1xyz / (-q1), n2 = p2xyz / q2;
        double n1x = n1(0), n1y = n1(1), n1z = n1(2);
        double n2x = n2(0), n2y = n2(1), n2z = n2(2);
        if (1 - n2z * n2z <= 1e-16)
        {
            if (1 - n1z * n1z <= 1e-16)
            {
                psi = 0.0;
            }
            else
            {
                if (n1y >= 0)
                {
                    psi = std::acos(n1x / std::sqrt(1 - n1z * n1z));
                }
                else
                {
                    psi = 2 * M_PI - std::acos(n1x / std::sqrt(1 - n1z * n1z));
                }
            }
        }
        else
        {
            double beta = q1 / p1(0), gamma = 1 / std::sqrt(1 - beta * beta), c3 = n1.dot(n2);
            double delta1 = n1z * (gamma - 1) + n2z * gamma * beta, delta2 = n2z + n1z * (c3 * (gamma - 1) + gamma * beta);
            double delta = std::sqrt((1 - n2z * n2z) * (std::pow(gamma + c3 * gamma * beta, 2) - delta2 * delta2));
            if (delta <= 1e-16)
            {
                psi = 0.0;
            }
            else
            {
                double c4 = n1x * n2y - n2x * n1y;
                std::complex<double> exptem(((n2z * c3 - n1z) * delta1 + (1 + c3 * beta) * gamma * (1 - n2z * n2z)) / delta, delta1 * c4 / delta);
                psi = -std::arg(exptem);
            }
        }
    }
}

Eigen::Tensor<std::complex<double>, 3> PWFA(Eigen::Vector4d &p1, const double &mu1, const double &s1, Eigen::Vector4d &p2, const double &mu2, const double &s2, const double &s, const double &S, const int &L)
{
    const int lens1 = 2 * s1 + 1, lens2 = 2 * s2 + 1, lens = 2 * s + 1;
    Eigen::Vector3d p1xyz = p1.tail(3), p2xyz = p2.tail(3);
    double m1 = (mu1 <= 1e-16) ? mu1 : (std::sqrt(p1(0) * p1(0) - p1xyz.dot(p1xyz)));
    double m2 = (mu2 <= 1e-16) ? mu2 : (std::sqrt(p2(0) * p2(0) - p2xyz.dot(p2xyz)));
    Eigen::Vector4d p0 = p1 + p2;
    double p12 = p1(0) * p2(0) - p1xyz.dot(p2xyz), m = std::sqrt(m1 * m1 + m2 * m2 + 2 * p12);
    double lam = (m * m + m1 * m1 - m2 * m2 + 2 * m * p1(0)) / (2. * m * (m + p1(0) + p2(0)));
    double qs = std::sqrt(std::pow(m, 4) + std::pow((m1 * m1 - m2 * m2), 2) - 2 * m * m * (m1 * m1 + m2 * m2)) / (2. * m);
    Eigen::Vector3d ns = (p1xyz - (p1xyz + p2xyz) * lam) / qs;
    double theta, phi;
    xyzToangle(ns(0), ns(1), ns(2), theta, phi);
    Eigen::Tensor<std::complex<double>, 3> result = amp0ls(s1, lens1, s2, lens2, s, lens, theta, phi, S, L);
    Eigen::MatrixXd wigerdj1 = wigerdjx(s1, lens1);
    Eigen::MatrixXd wigerdj2 = wigerdjx(s2, lens2);
    Eigen::MatrixXcd trans1 = Eigen::MatrixXcd::Zero(lens1, lens1);
    Eigen::MatrixXcd trans2 = Eigen::MatrixXcd::Zero(lens2, lens2);
    if (m1 <= 1e-16)
    {
        double psix1;
        MasslessTransAngle(p0, p1, psix1);
        trans1 = wigerDx(lens1, wigerdj1, phi, theta, -psix1);
    }
    else
    {
        double thetay1, phiy1, psiy1;
        MassiveTransAngle(p0, p1, thetay1, phiy1, psiy1);
        trans1 = wigerDx(lens1, wigerdj1, phiy1, thetay1, psiy1) * wigerDx(lens1, wigerdj1, 0.0, -thetay1, -phiy1);
    }
    if (m2 <= 1e-16)
    {
        double psix2;
        MasslessTransAngle(p0, p2, psix2);
        trans2 = wigerDx(lens2, wigerdj2, M_PI + phi, M_PI - theta, -psix2);
    }
    else
    {
        double thetay2, phiy2, psiy2;
        MassiveTransAngle(p0, p2, thetay2, phiy2, psiy2);
        trans2 = wigerDx(lens2, wigerdj2, phiy2, thetay2, psiy2) * wigerDx(lens2, wigerdj2, 0.0, -thetay2, -phiy2);
    }
    double factor = (std::pow(qs, L) / std::sqrt(2 * s + 1));
    Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>> ten1(trans1.data(), trans1.rows(), trans1.cols());
    Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>> ten2(trans2.data(), trans2.rows(), trans2.cols());
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
    Eigen::Tensor<std::complex<double>, 3> resultx = result.contract(ten1, product_dims);
    Eigen::Tensor<std::complex<double>, 3> resulty = resultx.contract(ten2, product_dims);
    return factor * resulty;
}

double fac1x(const int &s)
{
    double result = 2.0;
    for (int i = 1; i <= s; ++i)
    {
        result *= i / (i + 0.5);
    }
    return result;
}
double CDKer(const double &s, const double &sig, const double &kappa)
{
    if (static_cast<int>(2 * s) & 1)
    {
        double kmax = std::min((2 * s + 1) / 4, (2 * s - 1) / 4 + sig), kmin = std::max(-(2 * s + 1) / 4, sig - (2 * s - 1) / 4);
        int xtem1 = s - sig, xtem2 = s + sig;
        int ytem1 = s / 2 - kmax + 1 / 4., ytem2 = s / 2 + kmax + 1 / 4., ytem3 = s / 2 - kmax - 1 / 4. + sig, ytem4 = s / 2 + kmax - 1 / 4. - sig;
        double facy = (2 * s + 1) / std::pow(2, 2 * s + 1) * fac1x(s - 1 / 2);
        double fack = fac2(xtem1, ytem1, ytem4) * fac2(xtem2, ytem2, ytem3);
        double result = (std::pow(kappa, 2 * kmax - sig) - std::pow(kappa, -2 * kmax + sig)) / 2 * fack;
        for (double k = kmax - 1; k >= kmin; --k)
        {
            fack *= (s / 2 + k + 5 / 4.) * (s / 2 + k - sig + 3 / 4.) / (s / 2 - k + 1 / 4.) / (s / 2 - k + sig - 1 / 4.);
            result += (std::pow(kappa, 2 * k - sig) - std::pow(kappa, -2 * k + sig)) * fack / 2;
        }
        return facy * result;
    }
    else
    {
        double kmax = std::min(s / 2, s / 2 + sig), kmin = std::max(-s / 2, sig - s / 2);
        int xtem1 = s + sig, xtem2 = s - sig;
        int ytem1 = s / 2 + kmax, ytem2 = s / 2 - kmax, ytem3 = s / 2 + kmax - sig, ytem4 = s / 2 - kmax + sig;
        double facy = (2 * s + 1) / std::pow(2, 2 * s + 1) * fac1x(s);
        double fack = std::pow(kappa, 2 * kmax - sig) * fac2(xtem1, ytem1, ytem4) * fac2(xtem2, ytem2, ytem3);
        double result = fack;
        for (double k = kmax - 1; k >= kmin; --k)
        {
            fack *= (s / 2 + k + 1) * (s / 2 + k - sig + 1) / (s / 2 - k) / (s / 2 - k + sig) / kappa / kappa;
            result += fack;
        }
        return facy * result;
    }
}
Eigen::VectorXd CDKerV(const double &s, const int &lens, const double &kappa)
{
    Eigen::VectorXd v1 = Eigen::VectorXd::Zero(lens);
    for (int j = 0; j < lens; ++j)
    {
        v1(j) = CDKer(s, j - s, kappa);
    }
    return v1;
}

Eigen::Tensor<std::complex<double>, 3> PWFB(Eigen::Vector4d &p1, const double &mu1, const double &s1, Eigen::Vector4d &p2, const double &mu2, const double &s2, const double &s, const double &S, const int &L)
{
    const int lens1 = 2 * s1 + 1, lens2 = 2 * s2 + 1, lens = 2 * s + 1;
    Eigen::Vector3d p1xyz = p1.tail(3), p2xyz = p2.tail(3);
    double m1 = (mu1 <= 1e-16) ? mu1 : (std::sqrt(p1(0) * p1(0) - p1xyz.dot(p1xyz)));
    double m2 = (mu2 <= 1e-16) ? mu2 : (std::sqrt(p2(0) * p2(0) - p2xyz.dot(p2xyz)));
    Eigen::Vector4d p0 = p1 + p2;
    double p12 = p1(0) * p2(0) - p1xyz.dot(p2xyz), m = std::sqrt(m1 * m1 + m2 * m2 + 2 * p12);
    double lam = (m * m + m1 * m1 - m2 * m2 + 2 * m * p1(0)) / (2. * m * (m + p1(0) + p2(0)));
    double qs = std::sqrt(std::pow(m, 4) + std::pow((m1 * m1 - m2 * m2), 2) - 2 * m * m * (m1 * m1 + m2 * m2)) / (2. * m);
    Eigen::Vector3d ns = (p1xyz - (p1xyz + p2xyz) * lam) / qs;
    double theta, phi;
    xyzToangle(ns(0), ns(1), ns(2), theta, phi);
    Eigen::Tensor<std::complex<double>, 3> result = amp0ls(s1, lens1, s2, lens2, s, lens, theta, phi, S, L);
    Eigen::MatrixXd wigerdj1 = wigerdjx(s1, lens1);
    Eigen::MatrixXd wigerdj2 = wigerdjx(s2, lens2);
    Eigen::MatrixXcd trans1 = Eigen::MatrixXcd::Zero(lens1, lens1);
    Eigen::MatrixXcd trans2 = Eigen::MatrixXcd::Zero(lens2, lens2);
    Eigen::VectorXd CDkerfv1 = Eigen::VectorXd::Zero(lens1);
    Eigen::VectorXd CDkerfv2 = Eigen::VectorXd::Zero(lens2);
    Eigen::VectorXcd phasev1 = Eigen::VectorXcd::Zero(lens1), phasev2 = Eigen::VectorXcd::Zero(lens2);
    if (m1 <= 1e-16)
    {
        double psix1;
        MasslessTransAngle(p0, p1, psix1);
        CDkerfv1 = CDKerV(s1, lens1, qs);
        for (int s1x = 0; s1x < lens1; s1x++)
        {
            std::complex<double> xxtem(std::cos((s1x - s1) * psix1), std::sin((s1x - s1) * psix1));
            phasev1(s1x) = xxtem;
        }
        phasev1 = phasev1.array() * CDkerfv1.array();
        trans1 = wigerDx(lens1, wigerdj1, phi, theta, 0.0);
        trans1 = trans1.cwiseProduct(phasev1.transpose().replicate(lens1, 1));
        if (m2 <= 1e-16)
        {
            double psix2;
            MasslessTransAngle(p0, p2, psix2);
            trans2 = wigerDx(lens2, wigerdj2, M_PI + phi, M_PI - theta, 0.0);
            CDkerfv2 = CDKerV(s2, lens2, qs);
            for (int s2x = 0; s2x < lens2; s2x++)
            {
                std::complex<double> xxtem(std::cos((s2x - s2) * psix2), std::sin((s2x - s2) * psix2));
                phasev2(s2x) = xxtem;
            }
            phasev2 = phasev2.array() * CDkerfv2.array();
            trans2 = trans2.cwiseProduct(phasev2.transpose().replicate(lens2, 1));
        }
        else
        {
            double kappa2 = qs / m2 + std::sqrt(1 + qs * qs / m2 / m2);
            double thetay2, phiy2, psiy2;
            MassiveTransAngle(p0, p2, thetay2, phiy2, psiy2);
            trans2 = wigerDx(lens2, wigerdj2, M_PI + phi, M_PI - theta, 0.0);
            CDkerfv2 = CDKerV(s2, lens2, kappa2);
            trans2 = trans2.cwiseProduct(CDkerfv2.transpose().replicate(lens2, 1));
            trans2 = trans2 * wigerDx(lens2, wigerdj2, -M_PI - phi, M_PI - theta, 0.0).transpose();
            trans2 = trans2 * wigerDx(lens2, wigerdj2, phiy2, thetay2, psiy2);
            trans2 = trans2 * wigerDx(lens2, wigerdj2, 0.0, -thetay2, -phiy2);
        }
    }
    else
    {
        double kappa1 = qs / m1 + std::sqrt(1 + qs * qs / m1 / m1);
        double thetay1, phiy1, psiy1;
        MassiveTransAngle(p0, p1, thetay1, phiy1, psiy1);
        trans1 = wigerDx(lens1, wigerdj1, phi, theta, 0.0);
        CDkerfv1 = CDKerV(s1, lens1, kappa1);
        trans1 = trans1.cwiseProduct(CDkerfv1.transpose().replicate(lens1, 1));
        trans1 = trans1 * wigerDx(lens1, wigerdj1, -phi, theta, 0.0).transpose();
        trans1 = trans1 * wigerDx(lens1, wigerdj1, phiy1, thetay1, psiy1);
        trans1 = trans1 * wigerDx(lens1, wigerdj1, 0.0, -thetay1, -phiy1);
        if (m2 <= 1e-16)
        {
            double psix2;
            MasslessTransAngle(p0, p2, psix2);
            trans2 = wigerDx(lens2, wigerdj2, M_PI + phi, M_PI - theta, -0.0);
            CDkerfv2 = CDKerV(s2, lens2, qs);
            for (int s2x = 0; s2x < lens2; s2x++)
            {
                std::complex<double> xxtem(std::cos((s2x - s2) * psix2), std::sin((s2x - s2) * psix2));
                phasev2(s2x) = xxtem;
            }
            phasev2 = phasev2.array() * CDkerfv2.array();
            trans2 = trans2.cwiseProduct(phasev2.transpose().replicate(lens2, 1));
        }
        else
        {
            double kappa2 = qs / m2 + std::sqrt(1 + qs * qs / m2 / m2);
            double thetay2, phiy2, psiy2;
            MassiveTransAngle(p0, p2, thetay2, phiy2, psiy2);
            trans2 = wigerDx(lens2, wigerdj2, M_PI + phi, M_PI - theta, 0.0);
            CDkerfv2 = CDKerV(s2, lens2, kappa2);
            trans2 = trans2.cwiseProduct(CDkerfv2.transpose().replicate(lens2, 1));
            trans2 = trans2 * wigerDx(lens2, wigerdj2, -M_PI - phi, M_PI - theta, 0.0).transpose();
            trans2 = trans2 * wigerDx(lens2, wigerdj2, phiy2, thetay2, psiy2);
            trans2 = trans2 * wigerDx(lens2, wigerdj2, 0.0, -thetay2, -phiy2);
        }
    }
    double factor = (std::pow(qs, L) / std::sqrt(2 * s + 1));
    Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>> ten1(trans1.data(), trans1.rows(), trans1.cols());
    Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>> ten2(trans2.data(), trans2.rows(), trans2.cols());
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
    Eigen::Tensor<std::complex<double>, 3> resultx = result.contract(ten1, product_dims);
    Eigen::Tensor<std::complex<double>, 3> resulty = resultx.contract(ten2, product_dims);
    return factor * resulty;
}
