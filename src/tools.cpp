#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the inputs:
  if (estimations.size() == 0) {
    cout << "Error: Estimations size is 0!" << endl;
    return rmse;
  }
  if (estimations.size() != ground_truth.size()) {
    cout << "Estimations and ground_truth have to be the same size" << endl;
    return rmse;
  }

  // accumulate squared residuals
  VectorXd residual;
  for (std::vector<VectorXd>::size_type i = 0; i < estimations.size(); ++i) {
    residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse / estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  Hj << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // compute the Jacobian matrix
  float px_2 = px * px;
  float py_2 = py * py;
  float norm_p_2 = px_2 + py_2;
  float norm_p = sqrt(norm_p_2);

  // check division by zero
  if (fabs(norm_p_2) < 0.0001) {
    cout << "CalculateJacobian() - Error - Division by 0" << endl;
    return Hj;
  }

  float d_rho_d_px = px / norm_p;
  float d_rho_d_py = py / norm_p;
  float d_phi_d_px = -py / norm_p_2;
  float d_phi_d_py = px / norm_p_2;

  float norm_p_3 = norm_p_2 * norm_p;
  float d_rho_dot_d_px = py * (vx * py - vy * px) / norm_p_3;
  float d_rho_dot_d_py = px * (vy * px - vx * py) / norm_p_3;
  float d_rho_dot_d_vx = d_rho_d_px;
  float d_rho_dot_d_vy = d_rho_d_py;

  Hj << d_rho_d_px, d_rho_d_py, 0, 0,
        d_phi_d_px, d_phi_d_py, 0, 0,
        d_rho_dot_d_px, d_rho_dot_d_py, d_rho_dot_d_vx, d_rho_dot_d_vy;

  return Hj;
}
