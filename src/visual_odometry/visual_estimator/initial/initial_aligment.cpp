#include "initial_alignment.h"

// 初始化陀螺仪bias
// https://blog.csdn.net/weixin_35488643/article/details/112535151/
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    // 1.构造Ax=b
    Matrix3d A; // A = J^t * J
    Vector3d b; // b = J^t * b_
    Vector3d delta_bg; // x: 陀螺仪bias的变化量
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    // 遍历所有frames, 每两个相邻的frame构成一个约束
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i); // 相邻的两个frame

        // 1.1.填充J和b_
        MatrixXd tmp_A(3, 3); // J: R对陀螺仪bias的偏导
        tmp_A.setZero();
        VectorXd tmp_b(3);    // b_ = (r^bk_bk+1)^-1 * (q^c0_bk)^-1 * (q^c0_bk+1)
        tmp_b.setZero();
        // q_ij = (q^c0_bk)^-1 * (q^c0_bk+1) 视觉得到的imu预积分
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        // J: R对陀螺仪bias的偏导
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        // b_ = (r^bk_bk+1)^-1 * q_ij, imu实际的预积分 与 视觉得到的imu预积分 的差异
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();

        // 1.2.填充A和b_
        A += tmp_A.transpose() * tmp_A; // A = J^t * J
        b += tmp_A.transpose() * tmp_b; // b = J^t * b_
    }
    // 2.LDLT分解求解Ax=b, 得到陀螺仪bias的增量
    delta_bg = A.ldlt().solve(b);
    ROS_INFO_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg; // 更新陀螺仪bias

    // 3.更新完bias之后, 还得重新进行预积分
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

// 在半径为g0的半球寻找切面的一组正交基
MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c; // 切面上的一组正交基
    Vector3d a = g0.normalized(); // 重力向量
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

// 重力细化 (固定重力的模长, 优化速度, 重力和尺度)
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm(); // 固定模长为9.81
    Vector3d lx, ly; // 半球切面的正交基, g0 = g0 + w1 * lx + w2 * ly
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1; // 固定重力g的模长, 维度由3变为2, x = [v^b0_b0, v^b1_b1, ... v^bn_bn, w1, w2, s]^t

    // Ax=b
    MatrixXd A{n_state, n_state}; // A = H^t * H
    A.setZero();
    VectorXd b{n_state};          // b = H^t * b_
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    // 迭代四次, 不断优化重力向量
    for(int k = 0; k < 4; k++)
    {
        // 1.在半径为g0的半球寻找切面的一组正交基
        MatrixXd lxly(3, 2); // 一组正交基 lxly = [lx, ly]
        lxly = TangentBasis(g0);

        int i = 0;
        // 遍历所有frames, 每两个相邻的frame构成一个约束
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i); // 相邻的两个frame

            // 2.填充H和b_
            MatrixXd tmp_A(6, 9); // H = [H1, H2]^t
            tmp_A.setZero();
            VectorXd tmp_b(6);    // b_ = [b1, b2]^t
            tmp_b.setZero();

            // 相邻两帧的间隔时间
            double dt = frame_j->second.pre_integration->sum_dt;

            // H1 = [-I*dt, 0, 0.5 * R^bk_c0 * dt^2 * lxly, R^bk_c0 * (p^c0_ck+1 - p^c0_ck)]^t
            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();                                                      // -I*dt
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;       // 0.5 * R^bk_c0 * dt^2 * lxly
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0; // R^bk_c0 * (p^c0_ck+1 - p^c0_ck)
            // b1 = α^bk_bk+1 + R^bk_c0 * R^c0_bk+1 * p^b_c - p^b_c - 0.5 * R^bk_c0 * dt^2 * g0
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            // H2 = [-I, R^bk_c0 * R^c0_bk+1, R^bk_c0 * dt * lxly, 0]^t
            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();                                            // -I
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;                // R^bk_c0 * R^c0_bk+1
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly; // R^bk_c0 * dt * lxly
            // b2 = β^bk_bk+1 - R^bk_c0 * dt * g0
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            // 3.填充A和b
            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A; // A = H^t * H
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b; // b = H^t * b_

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }

        // 4.LDLT分解求解Ax=b
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b); // x = [v^b0_b0, v^b1_b1, ... v^bn_bn, w1, w2, s]^t
        VectorXd dg = x.segment<2>(n_state - 3); // [w1, w2]^t, dg = w1 * lx + w2 * ly
        g0 = (g0 + lxly * dg).normalized() * G.norm(); // 更新重力向零, g0 = g0 + w1 * lx + w2 * ly
        //double s = x(n_state - 1);
    }
    g = g0;
}

// 初始化速度, 重力和尺度
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1; // 状态变量的数量 (速度, 重力, 尺度)

    // 1.构造Ax=b
    MatrixXd A{n_state, n_state}; // A = H^t * H
    A.setZero();
    VectorXd b{n_state};          // b = H^t * b_
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    // 遍历所有frames, 每两个相邻的frame构成一个约束
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i); // 相邻的两个frame

        // 1.1.填充H和b_
        MatrixXd tmp_A(6, 10); // H = [H1, H2]^t
        tmp_A.setZero();
        VectorXd tmp_b(6);     // b_ = [b1, b2]^t
        tmp_b.setZero();

        // 相邻两帧的间隔时间
        double dt = frame_j->second.pre_integration->sum_dt;

        // H1 = [-I*dt, 0, 0.5 * R^bk_c0 * dt^2, R^bk_c0 * (p^c0_ck+1 - p^c0_ck)]^t
        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();                                                      // -I*dt
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();              // 0.5 * R^bk_c0 * dt^2
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0; // R^bk_c0 * (p^c0_ck+1 - p^c0_ck)
        // b1 = α^bk_bk+1 + R^bk_c0 * R^c0_bk+1 * p^b_c - p^b_c
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;

        // H2 = [-I, R^bk_c0 * R^c0_bk+1, R^bk_c0 * dt, 0]^t
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();                                     // -I
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;         // R^bk_c0 * R^c0_bk+1
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity(); // R^bk_c0 * dt
        // b2 = β^bk_bk+1
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        // 1.2.填充A和b
        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A; // A = H^t * H
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b; // b = H^t * b_

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }

    // 2.LDLT分解求解Ax=b
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b); // x = [v^b0_b0, v^b1_b1, ... v^bn_bn, g^c0, s]^t
    double s = x(n_state - 1) / 100.0;  // 尺度
    ROS_INFO("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);      // 重力加速度
    ROS_INFO_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    // 3.重力细化 (固定重力的模长, 优化速度, 重力和尺度)
    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_INFO_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;
    else
        return true;
}

// 视觉惯导对齐
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    // 1.初始化陀螺仪bias
    solveGyroscopeBias(all_image_frame, Bgs);

    // 2.初始化速度, 重力和尺度
    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
