#include "lice/map_distance_field.h"
#include "lice/utils.h"
#include "lice/math_utils.h"
#include "happly/happly.h"

#include <iostream>
#include <eigen3/Eigen/Dense>





// Function to solve the linear system with Eigen's cholesky decomposition
inline VecX solveKinvY(const MatX& K, const VecX& Y)
{
    Eigen::LLT<Eigen::MatrixXd> lltOfA(K);
    return lltOfA.solve(Y);
}


GPCellHyperparameters::GPCellHyperparameters(const double _lengthscale, const double _sz, const bool _use_weights)
{
    lengthscale = _lengthscale;
    inv_lengthscale2 = 1.0/(_lengthscale*_lengthscale);
    l2 = _lengthscale*_lengthscale;
    sz2 = _sz*_sz;
    two_l_2 = 2.0*_lengthscale*_lengthscale;
    two_beta_l_2 = 4.0*_lengthscale*_lengthscale;
    inv_2_l_2 = 1.0/two_l_2;
    inv_2_beta_l_2 = 1.0/two_beta_l_2;
    use_weights = _use_weights;
}



Cell::Cell(Vec3 pt, const double t, MapDistField* map_dist_field, const float intensity)
    : sum_(pt)
    , intensity_sum_(intensity)
    , first_time_(t)
    , map_(map_dist_field)
    , count_(1)
{
}

Cell::~Cell()
{
    lockCell();
    resetAlpha();
    unlockCell();
}



void Cell::addPt(const Vec3& pt, const float intensity)
{
    sum_ += pt;
    intensity_sum_ += intensity;
    count_++;
}

void Cell::setCount(int count)
{
    Vec3 pt = getPt();
    sum_ = pt * count;
    count_ = count;
    resetAlpha();
}



Vec3 Cell::getPt() const
{
    return sum_/count_;
}

float Cell::getIntensity() const
{
    return intensity_sum_/count_;
}


GridIndex Cell::getIndex() const
{
    auto pt = getPt();
    GridIndex index = map_->getGridIndex(pt); // to update bounds
    return index;
}



void Cell::resetAlpha()
{
    if(alpha_block_ != nullptr)
    {
        delete alpha_block_;
        alpha_block_ = nullptr;
    }
}

void Cell::getNeighbors(std::unordered_set<Cell*>& neighbors)
{
    std::vector<CellPtr> neighbors_vec = map_->getNeighborCells(getPt());
    for(auto& neighbor : neighbors_vec)
    {
        neighbors.insert(neighbor);
    }
}


VecX Cell::getWeights(const MatX& pts) const
{
    const GPCellHyperparameters& hp = map_->cell_hyperparameters;
    if (!hp.use_weights)
    {
        return VecX::Ones(pts.rows()) * hp.sz2;
    }
    double max_count = pts.col(3).maxCoeff();
    VecX weights = (((1.0 + hp.sz2) * (1.0+((12.0*(pts.col(3)/max_count)).array()-6.0).exp()).inverse().matrix()).array() + hp.sz2).matrix();
    return weights;
}


void Cell::computeAlpha(bool clean_behind)
{
    lockCell();
    if(alpha_block_ == nullptr)
    {
        alpha_block_ = new AlphaBlock();
        MatX pts = getNeighborPts(true);
        MatX weights = getWeights(pts).asDiagonal();
        MatX K = kernelRQ(pts.block(0,0,pts.rows(),3), pts.block(0,0,pts.rows(),3)) + weights;
        VecX Y = VecX::Ones(pts.rows());
        alpha_block_->alpha = solveKinvY(K, Y);
        if(!clean_behind)
        {
            map_->cellToClean(getIndex());
        }
        alpha_block_->neighbor_pts = pts.block(0, 0, pts.rows(), 3);
    }
    unlockCell();
}

MatX Cell::getNeighborPts(bool with_count)
{
    std::unordered_set<Cell*> neighbors;
    getNeighbors(neighbors);
    MatX pts(neighbors.size(), (with_count ? 4 : 3));
    int i = 0;
    for(auto& neighbor : neighbors)
    {
        pts.block(i, 0, 1, 3) = neighbor->getPt().transpose();
        if (with_count)
        {
            pts(i, 3) = neighbor->getCount();
        }
        i++;
    }
    return pts;
}



MatX Cell::kernelRQ(const MatX& X1, const MatX& X2) const
{
    const GPCellHyperparameters& hp = map_->cell_hyperparameters;
    MatX K(X1.rows(), X2.rows());
    for(int i = 0; i < X1.rows(); i++)
    {
        for(int j = 0; j < X2.rows(); j++)
        {
            double temp = 1.0 + ((X1.row(i) - X2.row(j)).squaredNorm()*hp.inv_2_beta_l_2);
            K(i, j) = 1.0/(temp*temp);
        }
    }
    return K;
}



std::tuple<MatX, MatX, MatX, MatX> Cell::kernelRQAndDiff(const MatX& X1, const MatX& X2)
{
    const GPCellHyperparameters& hp = map_->cell_hyperparameters;
    MatX K(X1.rows(), X2.rows());
    MatX K_diff_1(X1.rows(), X2.rows());
    MatX K_diff_2(X1.rows(), X2.rows());
    MatX K_diff_3(X1.rows(), X2.rows());
    for(int i = 0; i < X1.rows(); i++)
    {
        for(int j = 0; j < X2.rows(); j++)
        {
            double dist2 = (X1.row(i) - X2.row(j)).squaredNorm();
            Vec3 diff = X2.row(j) - X1.row(i);
            double temp = 1.0+(dist2*hp.inv_2_beta_l_2);
            double k = 1.0/(temp*temp);
            K(i, j) = k;
            temp = k/temp;
            K_diff_1(i, j) = diff[0]*hp.inv_lengthscale2*temp;
            K_diff_2(i, j) = diff[1]*hp.inv_lengthscale2*temp;
            K_diff_3(i, j) = diff[2]*hp.inv_lengthscale2*temp;
        }
    }
    return {K, K_diff_1, K_diff_2, K_diff_3};
}


double Cell::revertingRQ(const double& occ) const
{
    const GPCellHyperparameters& hp = map_->cell_hyperparameters;
    if(occ <= 0)
    {
        return -1;
    }
    else if (occ >= 1)
    {
        return 0;
    }
    else
    {
        return std::sqrt((1.0/std::sqrt(occ) - 1) * hp.two_beta_l_2);
    }
}

std::pair<double, double> Cell::revertingRQAndDiff(const double& occ) const
{
    const GPCellHyperparameters& hp = map_->cell_hyperparameters;
    if(occ <= 0)
    {
        return {1000, 0};
    }
    else if (occ >= 1)
    {
        return {0, 0};
    }
    else
    {
        double sqrt_occ = std::sqrt(occ);
        double dist = std::sqrt((1.0/sqrt_occ - 1)*hp.two_beta_l_2);
        double d_dist_d_occ = -hp.l2/(dist*occ*sqrt_occ);

        //double temp = (std::pow(occ, -hp.inv_beta) - 1) * hp.two_beta_l_2;
        //double dist = std::sqrt(temp);
        //double d_dist_d_occ = -hp.two_beta_l_2*std::pow(occ, -hp.inv_beta-1)*hp.inv_beta/(2.0*dist);

        return {dist, d_dist_d_occ};
    }
}


void Cell::testKernelAndRevert()
{
    std::cout << "Testing kernel and reverting function" << std::endl;
    // Create random points around the cell center
    int count_A = 10;
    int count_B = 1;
    MatX pts_A = MatX::Random(count_A, 3);
    MatX pts_B = MatX::Random(count_B, 3);
    auto [K, K_diff_1, K_diff_2, K_diff_3] = kernelRQAndDiff(pts_B, pts_A);
    std::cout << "K:\n" << K << std::endl;
    MatX K_diff_1_num(count_B, count_A);
    MatX K_diff_2_num(count_B, count_A);
    MatX K_diff_3_num(count_B, count_A);
    double eps = 1e-3;

    MatX pts_B_plus_1 = pts_B;
    pts_B_plus_1(0, 0) += eps;
    MatX pts_B_plus_2 = pts_B;
    pts_B_plus_2(0, 1) += eps;
    MatX pts_B_plus_3 = pts_B;
    pts_B_plus_3(0, 2) += eps;

    K_diff_1_num = (kernelRQ(pts_B_plus_1, pts_A) - K) / eps;
    K_diff_2_num = (kernelRQ(pts_B_plus_2, pts_A) - K) / eps;
    K_diff_3_num = (kernelRQ(pts_B_plus_3, pts_A) - K) / eps;

    std::cout << "K_diff_1:\n" << K_diff_1 << std::endl;
    std::cout << "K_diff_1_num:\n" << K_diff_1_num << std::endl;
    std::cout << "K_diff_2:\n" << K_diff_2 << std::endl;
    std::cout << "K_diff_2_num:\n" << K_diff_2_num << std::endl;
    std::cout << "K_diff_3:\n" << K_diff_3 << std::endl;
    std::cout << "K_diff_3_num:\n" << K_diff_3_num << std::endl;

    std::cout << "Max diff K diff 1: " << (K_diff_1 - K_diff_1_num).cwiseAbs().maxCoeff() << std::endl;
    std::cout << "Max diff K diff 2: " << (K_diff_2 - K_diff_2_num).cwiseAbs().maxCoeff() << std::endl;
    std::cout << "Max diff K diff 3: " << (K_diff_3 - K_diff_3_num).cwiseAbs().maxCoeff() << std::endl;


    int N = 10;
    VecX occs = VecX::Random(N).array().abs();
    VecX dists(N);
    VecX dists_2(N);
    VecX d_dists(N);
    VecX d_dists_num(N);
    eps = 1e-4;
    for(int i = 0; i < N; i++)
    {
        auto [dist, d_dist] = revertingRQAndDiff(occs(i));
        dists(i) = dist;
        d_dists(i) = d_dist;
        dists_2(i) = revertingRQ(occs(i)+eps);
        d_dists_num(i) = (dists_2(i) - dist) / eps;
    }
    std::cout << "occs:\n" << occs.transpose() << std::endl;
    std::cout << "dists:\n" << dists.transpose() << std::endl;
    std::cout << "dists_2:\n" << dists_2.transpose() << std::endl;
    std::cout << "d_dists:\n" << d_dists.transpose() << std::endl;
    std::cout << "d_dists_num:\n" << d_dists_num.transpose() << std::endl;
    std::cout << "Max diff d_dists: " << (d_dists - d_dists_num).cwiseAbs().maxCoeff() << std::endl;


    double test_dist = 0.8341;
    double test_occ = kernelRQ(Vec3(test_dist, 0, 0).transpose(), Vec3::Zero().transpose())(0,0);
    double test_dist_revert = revertingRQ(test_occ);
    std::cout << "Test dist: " << test_dist << ", test occ: " << test_occ << ", test dist revert: " << test_dist_revert << std::endl;

}



double Cell::getDist(const Vec3& pt)
{
    computeAlpha();
    MatX k = kernelRQ(pt.transpose(), alpha_block_->neighbor_pts);

    double occ = (k*alpha_block_->alpha)[0];
    if (occ < 0)
    {
        return (pt - getPt()).norm();
    }
    double dist = revertingRQ(occ);
    if (dist < 0)
    {
        return (pt - getPt()).norm();
    }
    return dist;
}

std::pair<double, Vec3> Cell::getDistAndGrad(const Vec3& pt)
{
    computeAlpha();
    auto [k, k_diff_1, k_diff_2, k_diff_3] = kernelRQAndDiff(pt.transpose(), alpha_block_->neighbor_pts);

    double occ = (k*alpha_block_->alpha)[0];
    Vec3 occ_grad;
    occ_grad[0] = (k_diff_1*alpha_block_->alpha)[0];
    occ_grad[1] = (k_diff_2*alpha_block_->alpha)[0];
    occ_grad[2] = (k_diff_3*alpha_block_->alpha)[0];
    if (occ <= 0)
    {
        Vec3 temp_vec = pt - getPt();
        double temp_dist = (temp_vec).norm();
        Vec3 temp_grad = temp_vec / temp_dist;
        return {temp_dist, temp_grad};
    }
    //double dist = revertingRQ(occ);
    auto[dist, d_dist_d_occ] = revertingRQAndDiff(occ);
    if (dist < 0)
    {
        Vec3 temp_vec = pt - getPt();
        double temp_dist = (temp_vec).norm();
        Vec3 temp_grad = temp_vec / temp_dist;
        return {temp_dist, temp_grad};
    }
    return {dist, d_dist_d_occ*occ_grad};
}


std::vector<Vec3> Cell::getNormals(const std::vector<Vec3>& pts, bool clean_behind)
{
    std::vector<Vec3> normals(pts.size());
    for(size_t i = 0; i < pts.size(); i++)
    {
        Vec3 occ_grad;
        {
            computeAlpha(clean_behind);
            auto [k, k_diff_1, k_diff_2, k_diff_3] = kernelRQAndDiff(pts[i].transpose(), alpha_block_->neighbor_pts);

            occ_grad[0] = (k_diff_1*alpha_block_->alpha)[0];
            occ_grad[1] = (k_diff_2*alpha_block_->alpha)[0];
            occ_grad[2] = (k_diff_3*alpha_block_->alpha)[0];
        }

        normals[i] = occ_grad.normalized();

        if(clean_behind)
        {
            resetAlpha();
        }
    }
    return normals;
}
        

double Cell::getUncertaintyProxy(){
    computeAlpha();
    return std::abs(alpha_block_->alpha.sum() - map_->cell_hyperparameters.uncertainty_proxy_calib);
}






MapDistField::MapDistField(const MapDistFieldOptions& options):
    cell_size_(options.cell_size)
    , inv_cell_size_(1.0/options.cell_size)
    , cell_size_f_((float)options.cell_size)
    , half_cell_size_f_((float)options.cell_size/2.0)
    , opt_(options)
    , cell_hyperparameters(options.gp_lengthscale < 0 ? 2.0*options.cell_size : options.gp_lengthscale, options.gp_sigma_z, options.use_voxel_weights)
{
    hash_map_ = std::make_unique<HashMap<CellPtr> >();
    if(opt_.edge_field)
    {
        hash_map_edge_ = std::make_unique<ankerl::unordered_dense::set<CellPtr>>();
    }

    calibrateUncertaintyProxy();
}

MapDistField::~MapDistField()
{
    for(auto& pair : *hash_map_)
    {
        delete pair.second;
    }
}

void MapDistField::clear()
{
    for(auto& pair : *hash_map_)
    {
        delete pair.second;
    }
    hash_map_->clear();
    if(hash_map_edge_)
    {
        hash_map_edge_->clear();
    }
    free_space_cells_.clear();
    ioctree_.clear();
    ioctree_edge_.clear();
    num_cells_ = 0;
    path_length_ = 0.0;
    scan_counter_ = -1;
    prev_scan_.clear();
    prev_pose_ = Mat4::Identity();
    cells_to_clean_.clear();
}


void MapDistField::calibrateUncertaintyProxy()
{
    // Create a latice of points around 0 on the x y plane
    std::vector<Pointd> latice_points;
    double spacing = cell_size_ / 5.0;
    double max_range = cell_size_ * 5.0;
    double offset = opt_.min_range + 1.0;
    double save_min_range = opt_.min_range;
    opt_.min_range = 0.0; // To make sure all points are added
    for(double x = -max_range; x <= max_range; x += spacing)
    {
        for(double y = -max_range; y <= max_range; y += spacing)
        {
            latice_points.push_back(Pointd(x+offset, y, 0.0, 0.0));
            latice_points.back().type = 1;
        }
    }
    // Add the points to the map at time 0
    Mat4 pose = Mat4::Identity();
    addPts(latice_points, pose);
    // Query the alpha values for the cell at the origin
    CellPtr origin_cell = getClosestCell(Vec3(offset, 0.0, 0.0));
    VecX alpha = origin_cell->getAlpha();
    cell_hyperparameters.uncertainty_proxy_calib = alpha.sum();

    // Clear the map
    clear();
    opt_.min_range = save_min_range;
}

Mat4 MapDistField::registerPts(const std::vector<Pointd>& pts, const Mat4& pose, const double current_time, const bool approximate, const double loss_scale, const int max_iterations)
{
    if(current_time != last_time_register_)
    {
        cleanCells();
    }
    std::cout << "Registering points" << std::endl;
    Vec6 pose_correction_state = Vec6::Zero();
    
    int num_neighbors_save = num_neighbors_;
    num_neighbors_ = 1;

    ceres::Problem problem;

    StopWatch sw;
    sw.start();
    std::cout << "Computing weights" << std::endl;
    std::vector<double> weights(pts.size(), 1.0);
    if(opt_.use_temporal_weights)
    {
        #pragma omp parallel for num_threads(8)
        for(size_t i = 0; i < pts.size(); i++)
        {
            weights[i] = getMinTime(pts[i].vec3());
        }
        double min_time = std::numeric_limits<double>::max();
        for(size_t i = 0; i < pts.size(); i++)
        {
            min_time = std::min(min_time, weights[i]);
        }
        if(current_time - min_time <= 0)
        {
            std::cout << "MapDistField::registerPts: Warning: current_time <= min_time (probably something funky with points' timestamps). Using uniform weights." << std::endl;
            for(size_t i = 0; i < pts.size(); i++)
            {
                weights[i] = 1.0;
            }
        }
        else
        {
            double coeff = -9.0/(current_time - min_time);
            for(size_t i = 0; i < pts.size(); i++)
            {
                weights[i] = 10.0 + (weights[i] - min_time)*coeff;
            }
        }
    }
    sw.stop();
    sw.print("Time to compute weights");

    auto temp_pose = pose;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.max_num_iterations = max_iterations;
    options.function_tolerance = 1e-4;
    options.minimizer_progress_to_stdout = true;


    // Optimization with openMP in the cost function
    bool use_loss = (loss_scale > 0.0);
    RegistrationCostFunction* cost_function = new RegistrationCostFunction(pts, temp_pose, this, weights, loss_scale, !approximate, use_loss);
    problem.AddResidualBlock(cost_function, NULL, pose_correction_state.data());


    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;



    Mat3 R = expMap(pose_correction_state.segment<3>(3));
    Vec3 pos = pose_correction_state.segment<3>(0);

    std::cout << "\n\n\n\n----------------\nPose correction: \n" << pose_correction_state.transpose() << "----------------\n\n\n\n" << std::endl;

    // Log the pose correction
    std::string log_path = "/home/ced/ros2_ws/install/ffastllamaa/share/ffastllamaa/maps/localization_corrections.csv";
    if(scan_counter_ <= 0)
    {
        std::ofstream log_file;
        log_file.open(log_path);
        log_file << "scan_counter,px,py,pz,rx,ry,rz,num_points,approximate\n";
        log_file.close();
    }
    {
        std::ofstream log_file;
        log_file.open(log_path, std::ios::app);
        if(log_file.is_open())
        {
            log_file << scan_counter_ << "," 
                     << pos[0] << "," << pos[1] << "," << pos[2] << ","
                     << pose_correction_state[3] << "," << pose_correction_state[4] << "," << pose_correction_state[5] << ","
                     << pts.size() << "," << approximate << "\n";
            log_file.close();
        }
        else
        {
            std::cerr << "MapDistField::registerPts: Unable to open log file: " << log_path << std::endl;
        }
    }


    Mat4 pose_correction = Mat4::Identity();
    pose_correction.block<3,3>(0,0) = R;
    pose_correction.block<3,1>(0,3) = pos;

    num_neighbors_ = num_neighbors_save;
    last_time_register_ = current_time;

    return temp_pose*pose_correction;
}




void MapDistField::cellToClean(const GridIndex& index)
{
    clean_mutex_.lock();
    cells_to_clean_.insert(index);
    clean_mutex_.unlock();
}

void MapDistField::cleanCells()
{
    clean_mutex_.lock();
    for(auto& index : cells_to_clean_)
    {
        auto it = hash_map_->find(index);
        if(it == hash_map_->end())
        {
            continue;
        }
        it->second->resetAlpha();
    }
    cells_to_clean_.clear();
    clean_mutex_.unlock();
}


std::pair<ankerl::unordered_dense::set<GridIndex>, std::vector<bool> > MapDistField::getFreeSpaceCellsToRemove(const std::vector<Pointd>& scan, const std::vector<Vec3>& map_pts, const Mat4& pose_scan, const Mat4& pose_map)
{
    // Overall this function creates an image like structure of the scan points and then checks which map points are in the free space of the scan.
    // It returns the map points that are in the free space and a mask indicating which points in the map_pts vector should be removed.

    // Define the angular resolution for the image like structure
    const double ang_res = 0.02;
    const double ang_res_inv = 1.0/ang_res;

    // Get the min and max elevation and azimuth indices
    int min_el_idx = std::numeric_limits<int>::max();
    int max_el_idx = std::numeric_limits<int>::lowest();
    int min_az_idx = std::numeric_limits<int>::max();
    int max_az_idx = std::numeric_limits<int>::lowest();
    // Take the opportunity to store the indices and range, and if the points are valid
    std::vector<std::tuple<int, int, double> > raw_indices(scan.size());
    std::vector<bool> valid_pts(scan.size(), true);
    int valid_count = 0;
    for(size_t i = 0; i < scan.size(); i++)
    {
        if(!std::isfinite(scan[i].x) || !std::isfinite(scan[i].y) || !std::isfinite(scan[i].z))
        {
            valid_pts[i] = false;
            continue;
        }
        float range = scan[i].vec3f().norm();
        if( (range < opt_.min_range) || (range > opt_.max_range)) 
        {
            valid_pts[i] = false;
            continue;
        }
        Vec3 polar = toPolar(scan[i].vec3());
        int el_idx = std::floor(polar[1]*ang_res_inv);
        int az_idx = std::floor(polar[2]*ang_res_inv);
        min_el_idx = std::min(min_el_idx, el_idx);
        max_el_idx = std::max(max_el_idx, el_idx);
        min_az_idx = std::min(min_az_idx, az_idx);
        max_az_idx = std::max(max_az_idx, az_idx);
        raw_indices[i] = {el_idx, az_idx, polar[0]};
        valid_count++;
    }

    // Prepare the output variables
    ankerl::unordered_dense::set<GridIndex> map_pts_to_remove;
    std::vector<bool> mask_remove(map_pts.size(), false);

    // If there are valid points
    if (valid_count > 0)
    {
        
        // Create the image like structure and fill it with the minimum distance in each cell
        int nb_el = max_el_idx - min_el_idx + 1;
        int nb_az = max_az_idx - min_az_idx + 1;
        std::vector<std::vector<double> > dists(nb_el, std::vector<double>(nb_az, -1));
        for(size_t i = 0; i < scan.size(); i++)
        {
            if(!valid_pts[i]) continue;
            auto [el_idx, az_idx, r] = raw_indices[i];
            if((el_idx - min_el_idx < 1) || (el_idx - min_el_idx >= nb_el-1)) continue;
            double cell_dist = dists[el_idx-min_el_idx][az_idx-min_az_idx];
            if(cell_dist < 0)
            {
                dists[el_idx-min_el_idx][az_idx-min_az_idx] = r;
            }
            else
            {
                dists[el_idx-min_el_idx][az_idx-min_az_idx] = std::min(cell_dist, r);
            }
        }

        // Check the neighbor pixels and keep the minimum distance of every 3x3 neighborhood
        std::vector<std::vector<double> > dists_copy = dists;
        for(int el = 0; el < nb_el; el++)
        {
            for(int az = 0; az < nb_az; az++)
            {
                double min_dist = std::numeric_limits<double>::max();
                int count = 0;
                for(int i = std::max(0, el-1); i <= std::min(nb_el-1, el+1); i++)
                {
                    for(int j = std::max(0, az-1); j <= std::min(nb_az-1, az+1); j++)
                    {
                        min_dist = std::min(min_dist, dists_copy[i][j]);
                        count++;
                    }
                }
                if(count > 0)
                {
                    dists[el][az] = min_dist;
                }
                else
                {
                    dists[el][az] = -1;
                }
            }
        }

        
        // Transform the map points to the scan frame and get the indices in the image like structure
        Mat4 pose_inv = pose_scan.inverse() * pose_map;
        double dist_threshold = 1.0*cell_size_;
        double min_range_threshold = opt_.min_range + dist_threshold;
        for(size_t i = 0; i < map_pts.size(); i++)
        {
            const Vec3& pt = map_pts[i];
            Vec3 temp_pt = pose_inv.block<3,3>(0,0)*pt + pose_inv.block<3,1>(0,3);
            Vec3 polar = toPolar(temp_pt);
            int el_idx = std::floor(polar[1]*ang_res_inv) - min_el_idx;
            int az_idx = std::floor(polar[2]*ang_res_inv) - min_az_idx;
            if((el_idx < 0) || (el_idx >= nb_el)) continue;
            if((az_idx < 0) || (az_idx >= nb_az)) continue;

            if((dists[el_idx][az_idx] > 0) && (polar[0] > min_range_threshold) && (dists[el_idx][az_idx]-dist_threshold > polar[0]))
            {
                Vec3 world_pt = pose_map.block<3,3>(0,0)*pt + pose_map.block<3,1>(0,3);
                map_pts_to_remove.insert(getGridIndex(world_pt));
                mask_remove[i] = true;
            }
        }
    }
    return {map_pts_to_remove, mask_remove};
}


std::vector<Vec3> MapDistField::getNeighborPoints(const Vec3& pt, const double radius)
{
    std::vector<Vec3> neighbors;
    std::vector<PointSimple> neighbors_octree_;
    std::vector<double> neighbor_dists;
    ioctree_.radiusNeighbors(PointSimple{pt[0], pt[1], pt[2]}, radius, neighbors_octree_, neighbor_dists);
    for(auto& neighbor : neighbors_octree_)
    {
        GridIndex index = getGridIndex(neighbor);
        auto it = hash_map_->find(index);
        if(it == hash_map_->end())
        {
            continue;
        }
        neighbors.push_back(it->second->getPt());
    }

    //PhNeighborQuery query;
    //phtree_.for_each(query, improbable::phtree::FilterSphere({pt[0], pt[1], pt[2]}, radius, phtree_.converter()));
    //for(auto& neighbor : query.neighbors)
    //{
    //    neighbors.push_back(neighbor.second->getPt());
    //}
    return neighbors;
}


void MapDistField::addPts(const std::vector<Pointd>& pts, const Mat4& pose, const std::vector<double>& count)
{
    if(time_offset_ < 0.0)
    {
        time_offset_ = pts[0].t;
    }
    cleanCells();
    if (pts.size() == 0)
    {
        return;
    }
    if(scan_counter_ < 0)
    {
        has_color_ = pts[0].has_color;
    }

    scan_counter_++;
    StopWatch sw;

    if(prev_scan_.size() > 0)
    {
        path_length_ += (prev_pose_.block<3,1>(0,3) - pose.block<3,1>(0,3)).norm();
    }


    std::vector<Pointd> pts_to_add;

    // Check if free space carving is enabled
    if(opt_.free_space_carving && count.size() == 0 && scan_counter_ > 0)
    {
        sw.start();

        pts_to_add = freeSpaceCarving(pts, pose);

        sw.stop();
        sw.print("Time to remove points for carving");

    }
    else
    {
        pts_to_add = pts;
    }

    prev_scan_ = pts;
    prev_pose_ = pose;


    sw.reset();
    sw.start();

    // Project the points to the map frame
    double min_point_time = std::numeric_limits<double>::max();
    std::vector<PointSimple> pts_to_add_octree;
    pts_to_add_octree.reserve(pts_to_add.size()/8);
    std::vector<PointSimple> pts_to_add_octree_edge;
    pts_to_add_octree_edge.reserve(pts_to_add.size()/16);
    for (size_t i = 0; i < pts_to_add.size(); i++)
    {
        if(pts_to_add[i].type == kSkyPoint) continue;
        if(pts_to_add[i].type == kInvalidPoint) continue;
        
        min_point_time = std::min(min_point_time, (double)(pts_to_add[i].t));
        
        // Check if point is a number
        bool is_a_number = std::isfinite(pts_to_add[i].x) && std::isfinite(pts_to_add[i].y) && std::isfinite(pts_to_add[i].z);
        if(!is_a_number)
        {
            continue;
        }
        float range = pts_to_add[i].vec3f().norm();
        if( (range < opt_.min_range) || (range > opt_.max_range))
        {
            continue;
        }

        Vec3 temp_pt = pose.block<3,3>(0,0)*pts_to_add[i].vec3() + pose.block<3,1>(0,3);

        GridIndex index = getGridIndex(temp_pt);

        if(free_space_cells_.count(index) > 0)
        {
            continue;
        }

        CellPtr cell_ptr;
        if (hash_map_->count(index) == 0)
        {
            cell_ptr = new Cell(temp_pt, ((pts_to_add[i].t)-time_offset_)*1e-9, this, pts_to_add[i].i);
            if(count.size() > 0)
            {
                cell_ptr->setCount(count[i]);
            }
            hash_map_->insert({index, cell_ptr});
            num_cells_++;
            temp_pt = getCenterPt(index);
            //phtree_.emplace({temp_pt[0], temp_pt[1], temp_pt[2]}, cell_ptr);
            pts_to_add_octree.push_back(PointSimple{temp_pt[0], temp_pt[1], temp_pt[2]});

        }
        else
        {
            auto it = hash_map_->find(index);
            if(it == hash_map_->end())
            {
                continue;
            }
            it->second->addPt(temp_pt, pts_to_add[i].i);
        }

        if(opt_.edge_field && (pts_to_add[i].type == 2))
        {
            if (hash_map_edge_->count(cell_ptr) == 0)
            {
                //PointPh edge_point = {temp_pt[0], temp_pt[1], temp_pt[2]};
                //hash_map_edge_->insert({cell_ptr, edge_point});
                //phtree_edge_.emplace(edge_point, cell_ptr);
                hash_map_edge_->insert(cell_ptr);

                PointSimple edge_octree_point = {temp_pt[0], temp_pt[1], temp_pt[2]};
                pts_to_add_octree_edge.push_back(edge_octree_point);
            }
        }
    }

    // Insert the new points in the octree
    if(pts_to_add_octree.size() > 0)
    {
        if(ioctree_.size() == 0)
        {
            ioctree_.initialize(pts_to_add_octree);
        }
        else
        {
            ioctree_.update(pts_to_add_octree);
        }
    }
    // Insert the new edge points in the octree
    if(opt_.edge_field && pts_to_add_octree_edge.size() > 0)
    {
        if(ioctree_edge_.size() == 0)
        {
            ioctree_edge_.initialize(pts_to_add_octree_edge);
        }
        else
        {
            ioctree_edge_.update(pts_to_add_octree_edge);
        }
    }



    sw.stop();
    sw.print("Time to transform and add points");
    std::cout << "Number of cells in the map: " << num_cells_ << std::endl;
    std::cout << "Number in hash map: " << hash_map_->size() << std::endl;
    std::cout << "Number in ioctree: " << ioctree_.size() << std::endl;
    if(opt_.edge_field)
    {
        std::cout << "Number in edge map: " << hash_map_edge_->size() << std::endl;
        std::cout << "Number in ioctree edge: " << ioctree_edge_.size() << std::endl;
    }
}


std::vector<Pointd> MapDistField::freeSpaceCarving(const std::vector<Pointd>& pts, const Mat4& pose)
{

    // Get the map points that are in the carving radius
    std::vector<Vec3> map_points = getNeighborPoints(pose.block<3,1>(0,3), opt_.free_space_carving_radius);

    auto [map_pts_to_remove, mask_map_remove] = getFreeSpaceCellsToRemove(pts, map_points, pose, Mat4::Identity());

    std::vector<bool> temp_mask_remove;
    if((prev_scan_.size() > 0) && opt_.last_scan_carving)
    {
        int nb_points_to_remove = map_pts_to_remove.size();
        std::vector<Vec3> current_map_points;
        current_map_points.reserve(pts.size());
        for(auto& pt : pts)
        {
            current_map_points.push_back(pt.vec3());
        }
        auto [current_pts_to_remove, mask_current_remove] = getFreeSpaceCellsToRemove(prev_scan_, current_map_points, prev_pose_, pose);
        // Ugly copy, can be revised
        temp_mask_remove = mask_current_remove;


        std::vector<Vec3> prev_map_points;
        prev_map_points.reserve(prev_scan_.size());
        for(auto& pt : prev_scan_)
        {
            prev_map_points.push_back(pt.vec3());
        }
        auto [prev_pts_to_remove, mask_prev_remove] = getFreeSpaceCellsToRemove(pts, prev_map_points, pose, prev_pose_);
        for(auto& pt : prev_pts_to_remove)
        {
            map_pts_to_remove.insert(pt);
        }

        std::cout << "Number of points to remove from current scan: " << nb_points_to_remove << ", from last scan: " << map_pts_to_remove.size() - nb_points_to_remove << ", total map points: " << num_cells_ << std::endl; 
    }

    std::vector<Pointd> pts_to_add;
    pts_to_add.reserve(pts.size());
    if(temp_mask_remove.size() > 0)
    {
        for(size_t i = 0; i < temp_mask_remove.size(); i++)
        {
            if(!temp_mask_remove[i])
            {
                pts_to_add.push_back(pts[i]);
            }
        }
    }
    else
    {
        pts_to_add = pts;
    }


    // Remove the points from the map
    for(auto& map_index : map_pts_to_remove)
    {
        //free_space_cells_.insert(map_index);
        // For each cell to remove, also remove the cell from the neighbors
        //PhNeighborQuery query_to_remove;
        Vec3 map_pt = getCenterPt(map_index);
        std::vector<PointSimple> neighbors_octree_;
        std::vector<double> neighbor_dists;

        ioctree_.radiusNeighbors(PointSimple{map_pt[0], map_pt[1], map_pt[2]}, 2*cell_size_, neighbors_octree_, neighbor_dists);

        bool one = false;
        for(auto& neighbor_octree : neighbors_octree_)
        {
            GridIndex grid_index = getGridIndex(neighbor_octree);
            auto it = hash_map_->find(grid_index);
            if(it == hash_map_->end())
            {
                continue;
            }
            CellPtr neighbor_cell = it->second;
            if(opt_.edge_field && hash_map_edge_->count(neighbor_cell) > 0)
            {
                ioctree_edge_.boxWiseDelete(getCellBox(grid_index), true);
                hash_map_edge_->erase(neighbor_cell);
            }
            ioctree_.boxWiseDelete(getCellBox(grid_index), true);
            GridIndex neighbor_index = getGridIndex(neighbor_octree);
            free_space_cells_.insert(neighbor_index);
            hash_map_->erase(neighbor_index);
            num_cells_--;
            delete neighbor_cell;

            one = true;
            if((!opt_.over_reject) && one)
            {
                break;
            }
        }
    }
    return pts_to_add;
}



    
double MapDistField::getMinTime(const Vec3& pt)
{
    PointSimple query_point = {pt[0], pt[1], pt[2]};
    std::vector<double> neighbor_dists;
    std::vector<PointSimple> neighbors;
    ioctree_.knnNeighbors(query_point, 1, neighbors, neighbor_dists);
    if(neighbors.size() == 0)
    {
        return std::numeric_limits<double>::max();
    }
    GridIndex index = getGridIndex(neighbors[0]);
    auto it = hash_map_->find(index);
    if(it == hash_map_->end())
    {
        return std::numeric_limits<double>::max();
    }
    return it->second->getFirstTime();
}



std::vector<Pointd> MapDistField::getPts()
{
    std::vector<Pointd> pts;
    if(!hash_map_)
    {
        return pts;
    }
    pts.reserve(num_cells_);
    for (auto& pair : *hash_map_)
    {
        Vec3 pt = pair.second->getPt();
        int count = pair.second->getCount();
        float intensity = pair.second->getIntensity();
        pts.push_back(Pointd(pt[0], pt[1], pt[2], count, intensity));
        pts.back().type = (hash_map_edge_ && (hash_map_edge_->count(pair.second) > 0)) ? 2 : 1;
    }
    std::cout << "Number of cells: " << pts.size() << std::endl;
    return pts;
}

std::pair<std::vector<Pointd>, std::vector<Vec3> > MapDistField::getPtsAndNormals(bool clean_behind)
{
    std::vector<Pointd> pts;
    std::vector<Vec3> normals;
    if(!hash_map_)
    {
        return {pts, normals};
    }

    // Get cell in vector form for parallel processing
    std::vector<CellPtr> cells;
    cells.reserve(num_cells_);
    for(auto& pair : *hash_map_)
    {
        cells.push_back(pair.second);
    }


    pts.resize(num_cells_);
    normals.resize(num_cells_);
    #pragma omp parallel for num_threads(16)
    for(size_t i = 0; i < cells.size(); i++)
    {
        Vec3 pt = cells[i]->getPt();
        int count = cells[i]->getCount();
        pts[i] = Pointd(pt[0], pt[1], pt[2], 0.0, 0.0, count);
        normals[i] = cells[i]->getNormals({pt}, clean_behind)[0];
    }
    return {pts, normals};
}




double MapDistField::queryDistField(const Vec3& pt, const bool field, const int type)
{
    double dist = std::numeric_limits<double>::max();
    thuni::Octree& octree = (opt_.edge_field && (type == 2) && hash_map_edge_ && hash_map_edge_->size() > 0) ? ioctree_edge_ : ioctree_;
    if(octree.size() == 0)
    {
        return dist;
    }
    std::vector<double> neighbor_dists;
    std::vector<PointSimple> neighbors;
    octree.knnNeighbors(PointSimple{pt[0], pt[1], pt[2]}, num_neighbors_, neighbors, neighbor_dists);
    for(size_t i = 0; i < neighbors.size(); i++)
    {
        GridIndex index = getGridIndex(neighbors[i]);
        auto it = hash_map_->find(index);
        if(it == hash_map_->end())
        {
            continue;
        }
        CellPtr cell = it->second;
        
        if(field)
        {
            double temp_dist = cell->getDist(pt);
            dist = std::min(dist, temp_dist);
        }
        else
        {
            double temp_dist = (pt - cell->getPt()).norm();
            dist = std::min(dist, temp_dist);
        }
    }

    if(dist == std::numeric_limits<double>::max())
    {
        return 0.0; // No cell found, return 0 distance for not affecting optimization
    }
    return dist;
}

std::pair<double, double> MapDistField::queryDistFieldAndUncertaintyProxy(const Vec3& pt)
{
    double dist = std::numeric_limits<double>::max();
    double uncertainty_proxy = std::numeric_limits<double>::max();
    CellPtr best_cell = nullptr;
    std::vector<double> neighbor_dists;
    std::vector<PointSimple> neighbors;
    ioctree_.knnNeighbors(PointSimple{pt[0], pt[1], pt[2]}, num_neighbors_, neighbors, neighbor_dists);
    for(size_t i = 0; i < neighbors.size(); i++)
    {
        GridIndex index = getGridIndex(neighbors[i]);
        auto it = hash_map_->find(index);
        if(it == hash_map_->end())
        {
            continue;
        }
        CellPtr cell = it->second;
        double temp_dist = cell->getDist(pt);
        if(temp_dist < dist)
        {
            dist = temp_dist;
            best_cell = cell;
        }
    }
    if(best_cell)
    {
        uncertainty_proxy = best_cell->getUncertaintyProxy();
    }
    return {dist, uncertainty_proxy};
}


std::vector<double> MapDistField::queryDistField(const std::vector<Vec3>& pts, const bool field)
{
    std::vector<double> dists(pts.size());
    #pragma omp parallel for num_threads(12)
    for(size_t i = 0; i < pts.size(); i++)
    {
        dists[i] = queryDistField(pts[i], field);
    }
    return dists;
}



std::pair<double, Vec3> MapDistField::queryDistFieldAndGrad(const Vec3& pt, const bool field, const int type)
{
    double dist = std::numeric_limits<double>::max();
    Vec3 grad = Vec3::Zero();
    CellPtr best_cell = nullptr;
    std::vector<double> neighbor_dists;
    std::vector<PointSimple> neighbors;
    thuni::Octree& octree = (opt_.edge_field && (type == 2) && hash_map_edge_ && hash_map_edge_->size() > 0) ? ioctree_edge_ : ioctree_;
    if(octree.size() == 0)
    {
        return {dist, grad};
    }
    octree.knnNeighbors(PointSimple{pt[0], pt[1], pt[2]}, num_neighbors_, neighbors, neighbor_dists);
    for(size_t i = 0; i < neighbors.size(); i++)
    {
        GridIndex index = getGridIndex(neighbors[i]);
        auto it = hash_map_->find(index);
        if(it == hash_map_->end())
        {
            continue;
        }
        CellPtr cell = it->second;
        double temp_dist;
        if(field)
        {
            temp_dist = cell->getDist(pt);
        }
        else
        {
            temp_dist = (pt - cell->getPt()).norm();
        }
        if(temp_dist < dist)
        {
            dist = temp_dist;
            best_cell = cell;
        }
    }
    if(best_cell)
    {
        if(field)
        {
            std::tie(dist, grad) = best_cell->getDistAndGrad(pt);
        }
        else
        {
            grad = (pt - best_cell->getPt()).normalized();
        }
    }
    else
    {
        dist = 0.0; // No cell found, return 0 distance for not affecting optimization
    }
    return {dist, grad};
}





CellPtr MapDistField::getClosestCell(const Vec3& pt)
{
    CellPtr closest_cell = nullptr;
    double dist = std::numeric_limits<double>::max();
    std::vector<double> neighbor_dists;
    std::vector<PointSimple> neighbors;
    ioctree_.knnNeighbors(PointSimple{pt[0], pt[1], pt[2]}, num_neighbors_, neighbors, neighbor_dists);
    for(auto& neighbor : neighbors)
    {
        GridIndex index = getGridIndex(neighbor);
        auto it = hash_map_->find(index);
        if(it == hash_map_->end())
        {
            continue;
        }
        CellPtr cell = it->second;
        double temp_dist = (pt - cell->getPt()).norm();
        if(temp_dist < dist)
        {
            dist = temp_dist;
            closest_cell = cell;
        }
    }
    return closest_cell;
}




GridIndex MapDistField::getGridIndex(const Vec3& pos)
{
    return std::make_tuple(std::floor(pos[0]*inv_cell_size_), std::floor(pos[1]*inv_cell_size_), std::floor(pos[2]*inv_cell_size_));
}



GridIndex MapDistField::getGridIndex(const Vec2& pos)
{
    return std::make_tuple(std::floor(pos[0]*inv_cell_size_), std::floor(pos[1]*inv_cell_size_), 0);
}

GridIndex MapDistField::getGridIndex(const PointSimple& pos)
{
    return std::make_tuple(std::floor(pos.x*inv_cell_size_), std::floor(pos.y*inv_cell_size_), std::floor(pos.z*inv_cell_size_));
}

Vec3 MapDistField::getCenterPt(const GridIndex& index)
{
    return Vec3(std::get<0>(index)*cell_size_f_ + half_cell_size_f_, std::get<1>(index)*cell_size_f_ + half_cell_size_f_, std::get<2>(index)*cell_size_f_ + half_cell_size_f_);
}

thuni::BoxDeleteType MapDistField::getCellBox(const GridIndex& index)
{
    thuni::BoxDeleteType box;
    box.min[0] = std::get<0>(index)*cell_size_f_;
    box.min[1] = std::get<1>(index)*cell_size_f_;
    box.min[2] = std::get<2>(index)*cell_size_f_;
    box.max[0] = box.min[0] + cell_size_f_;
    box.max[1] = box.min[1] + cell_size_f_;
    box.max[2] = box.min[2] + cell_size_f_;
    return box;
}


std::vector<CellPtr> MapDistField::getNeighborCells(const Vec3& pt)
{
    std::vector<CellPtr> neighbors;
    double radius = (opt_.neighborhood_size+0.5)*cell_size_;
    std::vector<PointSimple> neighbors_octree_;
    std::vector<double> neighbor_dists;
    ioctree_.radiusNeighbors(PointSimple{pt[0], pt[1], pt[2]}, radius, neighbors_octree_, neighbor_dists);
    for(auto& neighbor : neighbors_octree_)
    {
        GridIndex index = getGridIndex(neighbor);
        auto it = hash_map_->find(index);
        if(it == hash_map_->end())
        {
            continue;
        }
        neighbors.push_back(it->second);
    }
    return neighbors;
}



void MapDistField::writeMap(const std::string& filename)
{
    std::cout << "Writing map to file: " << filename << std::endl;
    std::cout << "Querying points and normals ...." << std::endl;

    StopWatch sw;
    sw.start();

    // Query points and normals
    std::vector<Pointd> pts;
    std::vector<Vec3> normals;
    pts = getPts();

    //// Convert the points to Eigen format
    // Variables for the full map
    std::vector<Vec3> pts_eigen;
    pts_eigen.reserve(pts.size());
    std::vector<double> pts_counter;
    std::vector<double> pts_intensity;
    std::vector<double> pts_types;
    pts_counter.reserve(pts.size());
    std::vector<Vec3> normals_eigen;
    std::vector<std::array<unsigned char, 3> > colors;
    std::vector<Eigen::Vector3i> faces;
    for(size_t i = 0; i < pts.size(); i++)
    {
        bool valid = false;
        if(normals.size() > 0)
        {
            if(std::isfinite(normals[i][0]) && std::isfinite(normals[i][1]) && std::isfinite(normals[i][2]))
            {
                normals_eigen.push_back(normals[i].normalized());
                valid = true;
            }
        }
        else
        {
            valid = true;
        }

        if(valid)
        {
            pts_eigen.push_back(pts[i].vec3());
            pts_counter.push_back(pts[i].channel);
            pts_intensity.push_back(pts[i].i);
            pts_types.push_back(pts[i].type);
            if(has_color_)
                colors.push_back({pts[i].r, pts[i].g, pts[i].b});
        }
    }



    // Write the map
    writePly(filename, pts_eigen, normals_eigen, colors, pts_counter, pts_intensity, pts_types, faces);

    // Write the resolution
    std::string static_filename = filename;
    if(static_filename.size() > 4 && static_filename.substr(static_filename.size()-4,4) == ".ply")
    {
        static_filename = static_filename.substr(0, static_filename.size()-4) + ".info";
    }
    else
    {
        static_filename = static_filename + ".info";
    }
    std::ofstream info_file;
    info_file.open(static_filename);
    info_file << "cell_size " << cell_size_ << std::endl;
    info_file.close();


    sw.stop();
    sw.print("Time to query and write maps to file");
}



void MapDistField::writePly(const std::string& filename, const std::vector<Vec3>& pts, const std::vector<Vec3>& normals, const std::vector<std::array<unsigned char, 3> >& colors, const std::vector<double>& count, const std::vector<double>& intensity, const std::vector<double>& types, const std::vector<Eigen::Vector3i>& faces) const
{
    happly::PLYData ply_out;
    ply_out.addVertexPositions<double>(pts);
    if(normals.size() > 0)
    {
        ply_out.addVertexNormals(normals);
    }
    if(colors.size() > 0)
    {
        ply_out.addVertexColors(colors);
    }
    if(count.size() > 0)
    {
        ply_out.addVertexScalar(count, "count");
    }
    if(intensity.size() > 0)
    {
        ply_out.addVertexScalar(intensity, "intensity");
    }
    if(types.size() > 0)
    {
        ply_out.addVertexScalar(types, "type");
    }
    if(faces.size() > 0)
    {
        ply_out.addFaceIndices(faces);
    }
    ply_out.write(filename, happly::DataFormat::Binary);
}



void MapDistField::loadMap(const std::string& filename)
{
    // Read the PLY file
    StopWatch sw;
    sw.start();
    
    clear();

    std::cout << "Loading map from file: " << filename << std::endl;
    happly::PLYData ply_in(filename);

    std::vector<std::array<double, 3> > pts = ply_in.getVertexPositions();
    sw.stop();
    sw.print("Time to read points from file");
    std::cout << "Number of points in the map: " << pts.size() << std::endl;

    std::vector<double> types;
    if(ply_in.getElement("vertex").hasProperty("type"))
    {
        types = ply_in.getElement("vertex").getProperty<double>("type");
    }

    sw.reset();
    sw.start();
    std::vector<Pointd> pts_to_add;
    pts_to_add.reserve(pts.size());
    for(size_t i = 0; i < pts.size(); i++)
    {
        const auto& pt = pts[i];
        if(std::isfinite(pt[0]) && std::isfinite(pt[1]) && std::isfinite(pt[2]))
        {
            pts_to_add.push_back(Pointd(pt[0], pt[1], pt[2], int64_t(0)));
            // Round the type to the nearest integer
            if(types.size() > 0)
            {
                pts_to_add.back().type = std::lround(types[i]);
            }
            else
            {
                pts_to_add.back().type = 1;
            }
        }
    }

    addPts(pts_to_add, Mat4::Identity());

    sw.stop();
    sw.print("Time to load map");

}
    






RegistrationCostFunction::RegistrationCostFunction(const std::vector<Pointd>& pts, const Mat4& prior, MapDistField* map, const std::vector<double>& weights, const double cauchy_loss_scale, const bool use_field, const bool use_loss)
    : prior_(prior)
    , map_(map)
    , weights_(weights)
    , use_field_(use_field)
{
    pts_.reserve(pts.size());
    type_.reserve(pts.size());
    for(auto& pt : pts)
    {
        pts_.push_back(pt.vec3());
        type_.push_back(pt.type);
    }

    if(use_loss)
    {
        loss_function_ = std::make_unique<ceres::CauchyLoss>(cauchy_loss_scale);
    }
    else
    {
        loss_function_ = std::make_unique<ceres::TrivialLoss>();
    }
    set_num_residuals(pts.size());
    std::vector<int>* parameter_block_sizes = mutable_parameter_block_sizes();
    parameter_block_sizes->push_back(6);
}

void RegistrationCostFunction::setUseField(const bool use_field)
{
    use_field_ = use_field;
}

bool RegistrationCostFunction::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
    Eigen::Map<const Vec3> pos(parameters[0]);
    Eigen::Map<const Vec3> rot(parameters[0]+3);

    Mat3 R_prior = prior_.block<3,3>(0,0);
    Vec3 pos_prior = prior_.block<3,1>(0,3);

    Mat3 R = expMap(rot);

    Mat3 R_w = R_prior*R;
    Vec3 p_w = R_prior*pos + pos_prior;

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> pts(pts_[0].data(), 3, pts_.size());

    if(jacobians != NULL)
    {
        if( jacobians[0] != NULL)
        {
            MatX pts_corr = R*pts;
            pts_corr.colwise() += pos;
            
            MatX pts_w = R_prior*pts_corr;
            pts_w.colwise() += pos_prior;

            Mat3 J_rot = jacobianLefthandSO3(rot);

            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[0], pts_.size(), 6);

            #pragma omp parallel for num_threads(8)
            for(size_t i = 0; i < pts_.size(); i++)
            {
                Vec3 temp_pt = pts_w.col(i);
                auto [dist, grad] = map_->queryDistFieldAndGrad(temp_pt, use_field_, type_[i]);
                // Apply the loss function
                std::array<double, 3> temp;
                loss_function_->Evaluate(dist, temp.data());
                residuals[i] = temp[0] * weights_[i];

                Row3 d_dist_d_rot = -temp[1]*grad.transpose()*R_prior*(toSkewSymMat(pts_corr.col(i)-pos))*J_rot;
                Row3 d_dist_d_pos = temp[1]*grad.transpose()*R_prior;

                jacobian.block<1,3>(i, 0) = weights_[i] * d_dist_d_pos;
                jacobian.block<1,3>(i, 3) = weights_[i] * d_dist_d_rot;
            }
        }
    }
    else
    {
        MatX pts_w = R_w*pts;
        pts_w.colwise() += p_w;

        #pragma omp parallel for num_threads(8)
        for(size_t i = 0; i < pts_.size(); i++)
        {
            residuals[i] = map_->queryDistField(pts_w.col(i), use_field_, type_[i]);
            // Apply the loss function
            std::array<double, 3> temp;
            loss_function_->Evaluate(residuals[i], temp.data());
            residuals[i] = temp[0] * weights_[i];
        }

    }
    return true;
}



