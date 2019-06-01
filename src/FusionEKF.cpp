#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ << 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0;

  // Initialize some ekf_ values to matrices that will have only part of its values changed
  ekf_ = KalmanFilter();
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0;

  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0;

  // Acceleration uncertainty
  noise_ax_ = 9.0;
  noise_ay_ = 9.0;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   *
   * Also initialize if time moved backwards (our filter does not address that)
   *  or if more than 5 seconds passed since last measurement (it is probably not
   *  the same object anymore)
   */
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) * 1e-6;
  previous_timestamp_ = measurement_pack.timestamp_;

  if (!is_initialized_ or (dt < 0) or (dt > 5)) {
    if (!is_initialized_) {
      cout << "EKF: first measurement" << endl;
    } else {
      cout << "EKF: restarting" << endl;
    }
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    // Initialize the timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // Initialize to an arbitrary value that says we have no idea about the speed
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1.0, 0.0,    0.0,    0.0,
               0.0, 1.0,    0.0,    0.0,
               0.0, 0.0, 1000.0,    0.0,
               0.0, 0.0,    0.0, 1000.0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      float px = cos(measurement_pack.raw_measurements_(1)) * measurement_pack.raw_measurements_(0);
      float py = sin(measurement_pack.raw_measurements_(1)) * measurement_pack.raw_measurements_(0);
      ekf_.x_ << px,
                 py,
                 0.0,
                 0.0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_(0),
                 measurement_pack.raw_measurements_(1),
                 0.0,
                 0.0;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  // Update the F_ & Q_ matrices (they are dependent only on dt)
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  // Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // set the process covariance matrix Q
  ekf_.Q_(0, 0) = 0.25 * dt_4 * noise_ax_;
  ekf_.Q_(0, 2) = 0.50 * dt_3 * noise_ax_;
  ekf_.Q_(1, 1) = 0.25 * dt_4 * noise_ay_;
  ekf_.Q_(1, 3) = 0.50 * dt_3 * noise_ay_;
  ekf_.Q_(2, 0) = 0.50 * dt_3 * noise_ax_;
  ekf_.Q_(2, 2) =        dt_2 * noise_ax_;
  ekf_.Q_(3, 1) = 0.50 * dt_3 * noise_ay_;
  ekf_.Q_(3, 3) =        dt_2 * noise_ay_;

  ekf_.Predict();

  /**
   * Update
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl << endl;
}
