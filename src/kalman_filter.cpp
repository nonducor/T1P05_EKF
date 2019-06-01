#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Calculate the non-linear estimate
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  float px2 = px * px;
  float py2 = py * py;

  float rho = sqrt(px2 + py2);
  float phi = atan2(py, px);

  // Calculate rho_dot with protection. 1 cm is a reasonable cut-off, as most
  // automotive radars will not be able to detect something so near
  float rho_dot;
  if (rho > 1e-2) {
    rho_dot = (px * vx + py * vy) / rho;
  } else {
    // Assume 0 radial speed due to lack of better option.
    rho_dot = 0.0;
  }

  VectorXd z_pred = VectorXd(3);
  z_pred << rho, phi, rho_dot;

  VectorXd y = z - z_pred;
  if (y(1) > M_PI) {
    y(1) -= (2 * M_PI);
  } else if (y(1) < -M_PI) {
    y(1) += (2 * M_PI);
  }
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
