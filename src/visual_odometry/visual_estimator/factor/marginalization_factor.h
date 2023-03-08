#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

// 残差块信息: imu预积分因子, 视觉重投影误差, 边缘化先验因子
struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate(); // 计算残差块的residuals和jacobians

    ceres::CostFunction *cost_function;     // 残差块: imu预积分因子, 视觉重投影误差, 边缘化先验因子
    ceres::LossFunction *loss_function;     // 鲁棒核函数
    std::vector<double *> parameter_blocks; // 参数块<参数地址>: 以imu因子为例, <para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]>
    std::vector<int> drop_set; // parameter_blocks中待marg变量的索引: 以imu因子为例, <0, 1> 对应是的 <para_Pose[0], para_SpeedBias[0]>

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals; // 残差, IMU:15X1, 视觉:2X1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size;
    std::unordered_map<long, int> parameter_block_idx;
};

// 边缘化信息
class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;

    // 添加残差块信息: imu预积分因子, 视觉重投影误差, 边缘化先验因子
    // 设置 parameter_block_size 和 parameter_block_idx
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    // 计算每个残差对应的residuals和Jacobian, 并更新parameter_block_data
    void preMarginalize();
    // 开启多线程构建信息矩阵H和b, 同时从H,b中恢复出线性化雅克比和残差
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors; // 残差块信息: imu预积分因子, 视觉重投影误差, 边缘化先验因子
    int m, n; // m为要marg掉的变量个数, n为要保留下来的变量个数
    std::unordered_map<long, int> parameter_block_size;      // <参数地址, 内存大小>: 在信息矩阵H中的大小
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx;       // <参数地址, 参数位置>: 在信息矩阵H中的位置, marge变量在前, 保留变量在后

    std::unordered_map<long, double *> parameter_block_data; // <参数地址, 参数数据>

    // 用于 MarginalizationFactor 中计算residuals和Jacobian
    std::vector<int> keep_block_size;
    std::vector<int> keep_block_idx;
    std::vector<double *> keep_block_data;

    // 从边缘化先验因子H,b中恢复出线性化雅克比和残差
    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
