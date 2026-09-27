#pragma once

#include "types.h"
#include <memory>
#include <atomic>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include "ankerl/unordered_dense.h"

#include "ioctree/octree2/Octree.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>


struct PointSimple
{
    double x;
    double y;
    double z;
};


template <typename V>
using HashMap = ankerl::unordered_dense::map<GridIndex, V>;

struct GPCellHyperparameters {
    double lengthscale;
    double inv_lengthscale2;
    double l2;
    double sz2;
    double two_l_2;
    double two_beta_l_2;
    double inv_2_l_2;
    double inv_2_beta_l_2;
    double uncertainty_proxy_calib = -1.0;
    bool use_weights = true;

    GPCellHyperparameters(const double lengthscale, const double sz, const bool use_weights = true);
};


class GpMapPublisher
{
    public:
        virtual void publishSubmapInfo(const std::string& filename, const Vec3& gravity) = 0;
};



class MapDistField;
class Cell;
typedef Cell* CellPtr;

// Buffers of one thread's cell fits (Cell::computeAlpha), kept from one fit to the next so that a
// fit only allocates what it stores: the octree search results, the neighbour cells, their counts
// and the kernel matrix
struct NeighborScratch
{
    std::vector<double*> octree_pts;
    std::vector<double> octree_dists;
    std::vector<CellPtr> cells;
    std::vector<double> counts;
    std::vector<double> kernel;
};

struct AlphaBlock
{
    VecX alpha;
    // Column-major: each coordinate of all the neighbours contiguous, so that the kernel over the
    // neighbours (fit and field queries) is computed a few neighbours at a time with vector instructions
    Eigen::Matrix<double, Eigen::Dynamic, 3> neighbor_pts;
};

class Cell {
    private:
        Vec3 sum_;
        float intensity_sum_ = 0.0;
        float first_time_ = -1.0;
        AlphaBlock* alpha_block_ = nullptr;
        MapDistField* map_;
        uint32_t count_;
        std::atomic_flag lock_ = ATOMIC_FLAG_INIT;


        void computeAlpha(bool clean_behind=false);

        MatX getNeighborPts(bool with_count=false);

        VecX getWeights(const MatX& pts) const;


        MatX kernelRQ(const MatX& X1, const MatX& X2) const;
        // `kernelRQ(X, X)`, which is what fitting a cell needs. The matrix is symmetric, so only
        // half of the pairs are evaluated and each result is written to both sides.
        MatX kernelRQSelf(const MatX& X) const;
        // The same into K (n x n), its lower triangle only (all the fit's Cholesky reads), a column
        // at a time from the columns of X (each coordinate contiguous)
        void kernelRQSelf(const Eigen::Matrix<double, Eigen::Dynamic, 3>& X, Eigen::Ref<MatX> K) const;
        std::tuple<MatX, MatX, MatX, MatX> kernelRQAndDiff(const MatX& X1, const MatX& X2);

        // Evaluate the GP occupancy (and its gradient) at a query point directly from the weights,
        // without building the kernel matrices
        double occupancy(const Vec3& pt) const;
        std::pair<double, Vec3> occupancyAndGrad(const Vec3& pt) const;



        double revertingRQ(const double& x) const;
        std::pair<double, double> revertingRQAndDiff(const double& x) const;

        inline void lockCell()
        {
            while(lock_.test_and_set(std::memory_order_acquire));
        }
        inline void unlockCell()
        {
            lock_.clear(std::memory_order_release);
        }

    public:
        Cell(Vec3 pt, const double first_time, MapDistField* map, const float intensity);

        ~Cell();

        void getNeighbors(std::unordered_set<Cell*>& neighbors);
        GridIndex getIndex() const;


        void resetAlpha();
        void addPt(const Vec3& pt, const float intensity);

        Vec3 getPt() const;
        float getIntensity() const;


        double getDist(const Vec3& pt);
        std::pair<double, Vec3> getDistAndGrad(const Vec3& pt);


        int getCount() const { return count_; }
        void setCount(int count);



        double getFirstTime() const { return first_time_; }

        std::vector<Vec3> getNormals(const std::vector<Vec3>& pts, bool clean_behind=false);

        void testKernelAndRevert();

        VecX getAlpha(bool clean_behind=false){
            computeAlpha(clean_behind);
            return alpha_block_->alpha;
        }

        double getUncertaintyProxy();

};



struct MapDistFieldOptions {
    double cell_size = 0.15;
    int neighborhood_size = 2;
    double gp_sigma_z = 0.05;
    double gp_lengthscale = -1.0;
    bool use_voxel_weights = true;
    bool use_temporal_weights = false;
    bool free_space_carving = false;
    double free_space_carving_radius = -1.0;
    double min_range = 0.0001;
    double max_range = std::numeric_limits<double>::max();
    bool edge_field = true;
    int num_threads = 8;
    std::string scan_folder = "";
    // Use the odometry pose given to registerPts as a prior and not only as an initial guess: a
    // residual pulls the pose correction back to zero, with one weight for the translation (1/m) and
    // one for the rotation (1/rad). The larger the weight, the more the odometry is trusted.
    bool use_odom_prior = false;
    double odom_prior_weight_pos = 1.0;
    double odom_prior_weight_rot = 1.0;
    // Weight the registration residuals with the per-point position covariance, which the scan has to
    // carry. The covariance is propagated through the distance field query, and the robust loss is
    // then applied to the resulting Mahalanobis distance instead of the euclidean one. The loss scale
    // is therefore a number of standard deviations, not a distance in meters. The map itself is taken
    // as deterministic: the points to register are the only source of uncertainty.
    bool use_point_covariances = false;
    // Recompute that propagation at every evaluation point of the solver (the surface normal the
    // covariance is projected onto changes as the pose moves), instead of freezing it at the pose
    // given as initial guess. Set it to false to get the frozen behaviour back.
    bool point_covariances_per_iteration = true;
    // Estimate a scale of the scan alongside the pose: the points are scaled around the origin of the
    // sensor before being registered. For a front-end whose reconstruction is not exactly metric, or
    // whose scale drifts. The estimate carries over from one scan to the next.
    bool use_scale_optimization = false;
    // Inverse of the standard deviation of the change of scale between two consecutive scans. The
    // larger it is, the more the scale is held at the value estimated for the previous scan. Nothing
    // else pins the scale down, so this is what makes it a slowly drifting quantity rather than a free
    // parameter of every registration.
    double scale_prior_weight = 100.0;
    // Largest number of cells allowed to hold a GP weight block at once, or -1 for no limit. Only
    // bites in a localization run: `addPts` releases every block when it runs, and a mapping run
    // reaches it on every scan it processes, so it never accumulates any. Dropping the blocks further than
    // `kMaxAlphaRange` from the pose bounds the footprint on a trajectory that travels, but not on a
    // map that fits inside that range, which is what this bounds instead. It costs rebuilding what
    // it evicts. A block is a few hundred bytes to a kilobyte, depending on `neighborhood_size`.
    int64_t max_num_alpha_cells = -1;
};


struct GravityFactorFunctor;


// GP weight blocks belonging to cells further than this from the current pose are released. They
// are rebuilt on demand if the trajectory comes back, so this only trades memory against
// recomputation. Like `max_num_alpha_cells`, it only bites in a localization run, where `addPts`
// never runs to release them.
inline constexpr double kMaxAlphaRange = 100.0;                // m

// Distance travelled between two range evictions. The sweep is a contiguous scan over the set of
// alpha-holding cells that never dereferences a cell it keeps, so it can afford to be frequent.
inline constexpr double kAlphaSweepTravelDistance = 10.0;      // m



class MapDistField {
    private:
        int num_neighbors_ = 2;

        ankerl::unordered_dense::set<GridIndex> free_space_cells_;

        std::unique_ptr<HashMap<CellPtr> > hash_map_;
        GpMapPublisher* publisher_;
        std::unique_ptr<ankerl::unordered_dense::set<CellPtr> > hash_map_edge_;
        const double cell_size_;
        const double inv_cell_size_;
        const float cell_size_f_;
        const float half_cell_size_f_;
        const int dim_ = 3;
        const int num_threads_;

        std::mutex clean_mutex_;
        // Cells that built a weight block during the current scan, one list per thread of the
        // OpenMP regions that call `computeAlpha`. Staging them per thread rather than inserting
        // straight into `cells_to_clean_` keeps a process-wide lock off a path that is taken once
        // per cell built, from inside those regions. Folded in by `mergeStagedCells`.
        std::vector<std::vector<GridIndex> > staged_cells_;
        // The cells currently holding an alpha block: what `cleanCells` releases when the map
        // changes, and what `evictFarAlphas` sweeps by range when it does not.
        ankerl::unordered_dense::set<GridIndex> cells_to_clean_;

        // Distance travelled since the last range eviction.
        double travel_since_sweep_ = 0.0;
        bool has_sweep_position_ = false;
        Vec3 last_sweep_position_ = Vec3::Zero();

        // Points a leaf holds before it splits. The default of 32 left the neighbourhood searches of
        // the cell fits 9 to 22% slower than 128, with a larger tree; 256 slows the nearest neighbour
        // searches down (oct_bench, on Boreas and Newer College maps with gp_map's workload). The
        // minimum extent, 0.01 m as by default, never binds: the points are cell centres.
        static constexpr size_t kOctreeBucketSize = 128;
        thuni::Octree ioctree_{kOctreeBucketSize, false, 0.01};
        thuni::Octree ioctree_edge_{kOctreeBucketSize, false, 0.01};

        std::vector<Pointd> prev_scan_;
        Mat4 prev_pose_;

        size_t num_cells_ = 0;

        double path_length_ = 0.0;

        std::tuple<std::vector<Vec3>, std::set<std::array<int, 2> > > getPointsAndEdges();

        int scan_counter_ = -1;

        bool has_color_ = false;
        
        int64_t last_time_register_ = -1;
        int64_t time_offset_ = -1;

        MapDistFieldOptions opt_;
        bool is_2d_ = false;

        Vec3 gravity_ = Vec3::Zero();

        // Scale of the scans, estimated by the registration when `use_scale_optimization` is set. It
        // carries over from one registration to the next, which is what the scale prior is relative
        // to. Deliberately not reset by `clear()`: the scale belongs to the front-end producing the
        // scans, not to the map, and a submap switch should not throw the estimate away.
        double scale_ = 1.0;

        void cleanCells();

        // Move what `cellToClean` staged per thread into `cells_to_clean_`. Called by everything
        // that reads that set, and only ever from outside a parallel region. Expects
        // `clean_mutex_` to be held.
        void mergeStagedCells();


        // Release the GP weight blocks of the cells further than `kMaxAlphaRange` from `position`,
        // every `kAlphaSweepTravelDistance` of travel. Nothing else bounds the memory when `addPts`
        // never runs, which is a localization run.
        void evictFarAlphas(const Vec3& position);

        std::pair<ankerl::unordered_dense::set<GridIndex>, std::vector<bool> > getFreeSpaceCellsToRemove(const std::vector<Pointd>& scan, const std::vector<Vec3>& map_pts, const Mat4& pose_scan, const Mat4& pose_map);

        std::vector<Vec3> getNeighborPoints(const Vec3& pt, const double radius);


        void writePly(const std::string& filename, const std::vector<Vec3>& pts, const std::vector<Vec3>& normals, const std::vector<std::array<unsigned char, 3>>& colors, const std::vector<double>& count, const std::vector<double>& intensity, const std::vector<double>& types, const std::vector<Eigen::Vector3i>& faces) const;


        void calibrateUncertaintyProxy();


    public:
        GPCellHyperparameters cell_hyperparameters;
        MapDistField(const MapDistFieldOptions& options, GpMapPublisher* publisher);

        ~MapDistField();

        void clear();

        void set2D(const bool is_2d){ is_2d_ = is_2d;}

        // `pts_cov` holds the position covariance of each point of `pts`, in the frame of the points.
        // It is only read when the `use_point_covariances` option is set, and the registration falls
        // back on unit weights if it does not have one covariance per point.
        // `disable_odom_prior` overrides the `use_odom_prior` option for this one call. It is meant for
        // a registration that is recovering from a bad prior pose, where anchoring the solution to that
        // pose is exactly the wrong thing to do.
        Mat4 registerPts(const std::vector<Pointd>& pts, const Mat4& prior, const int64_t current_time, const bool approximate=false, const double loss_scale=0.5, const int max_iterations=12, GravityFactorFunctor* gravity_factor = nullptr, const std::vector<Mat3>& pts_cov = std::vector<Mat3>(), const bool disable_odom_prior = false);

        void addPts(const std::vector<Pointd>& pts, const Mat4& pose, const std::vector<double>& count=std::vector<double>());
        std::vector<Pointd> getPts();
        std::pair<std::vector<Pointd>, std::vector<Vec3> > getPtsAndNormals(bool clean_behind=false);
        std::vector<double> queryDistField(const std::vector<Vec3>& query_pts, const bool field=true);
        std::pair<double, Vec3> queryDistFieldAndGrad(const Vec3& query_pts, const bool field=true, const int type=0);
        double queryDistField(const Vec3& query_pt, const bool field=true, const int type=0);
        std::pair<double, double> queryDistFieldAndUncertaintyProxy(const Vec3& query_pt);

        double getMinTime(const Vec3& pt);
        std::pair<double, double> getMinTimeAndProxyWeight(const Vec3& pt);

        void display(const double inf_resolution = 0.0);

        GridIndex getGridIndex(const Vec3& pt);
        GridIndex getGridIndex(const Vec2& pt);
        GridIndex getGridIndex(const PointSimple& pt);
        Vec3 getCenterPt(const GridIndex& index);
        thuni::BoxDeleteType getCellBox(const GridIndex& index);


        void writeMap(const std::string& filename);

        void loadMap(const std::string& filename);

        bool isInHash(const GridIndex& index) const{ return hash_map_->find(index) != hash_map_->end(); }

        CellPtr getClosestCell(const Vec3& pt);

        // The cell `pt` falls in, when it is occupied, and null otherwise. The cell centres form a
        // cubic lattice, whose Voronoi region is the cell itself, so that cell's centre is the
        // nearest centre of the whole map to `pt`: one lookup answers what a nearest neighbour
        // search in the octree would, exactly, whenever it does not come back empty. `edge` also
        // requires the cell to be one of the edge cells, which is what the edge octree holds.
        CellPtr cellContaining(const Vec3& pt, const bool edge);

        std::vector<CellPtr> getNeighborCells(const Vec3& pt);
        // The same into scratch.cells, with the scratch buffers reused for the search
        void getNeighborCells(const Vec3& pt, NeighborScratch& scratch);
        // The num_neighbors_ cells nearest to pt, from the edge octree if use_edge (thread-local
        // buffers for the search)
        void nearestCells(const Vec3& pt, const bool use_edge, std::vector<CellPtr>& cells);
        
        // Records that `index` now holds a weight block. Called from inside the OpenMP regions of
        // the registration, so it takes no lock: it appends to this thread's staging list.
        void cellToClean(const GridIndex& index);



        double getPathLength() const { return path_length_; }

        // Scale of the scans as last estimated by the registration, 1.0 when the estimation is off
        double getScale() const { return scale_; }
        // Carry an estimate over to another map, so that a submap switch does not throw it away
        void setScale(const double scale) { scale_ = scale; }

        std::vector<Pointd> freeSpaceCarving(const std::vector<Pointd>& pts, const Mat4& pose);

        // Inverse of the standard deviation of the distance residual of each point, obtained by
        // propagating the position covariance of the point through the distance field query. `pose`
        // is the transformation bringing the points, and their covariances, into the map frame.
        void computePointInvStd(const std::vector<Pointd>& pts, const std::vector<Mat3>& pts_cov, const Mat4& pose, const bool use_field, std::vector<double>& inv_std, const double scan_scale = 1.0);

        void setGravity(const Vec3& gravity) { gravity_ = gravity; }

        // Weights of the odometry prior, which the caller may want to set per registration rather than
        // once: the field block of the cost grows with the number of points of the scan while the
        // prior does not, so a prior that is to keep the same influence has to follow that number.
        void setOdomPriorWeights(const double weight_pos, const double weight_rot)
        {
            opt_.odom_prior_weight_pos = weight_pos;
            opt_.odom_prior_weight_rot = weight_rot;
        }

};



class RegistrationCostFunction: public ceres::CostFunction
{
    public:
        // `inv_std` points at the inverse of the standard deviation of the distance residual of each
        // point, obtained by propagating its position covariance through the distance field query.
        // When it is given, the loss function is applied to the Mahalanobis distance `dist*inv_std`
        // instead of the euclidean distance. A null pointer disables it, and the residuals are then
        // bit for bit what they were before the covariances existed.
        // It is a pointer and not a copy so that PointCovarianceCallback can refresh it between two
        // evaluations of the cost function; it has to outlive the cost function.
        // `use_scale` adds a second parameter block of size 1, the scale the points are multiplied by
        // before being registered. When it is false the cost function declares the pose block only,
        // exactly as it did before the scale existed.
        RegistrationCostFunction(const std::vector<Pointd>& pts, const Mat4& prior, MapDistField* map, const std::vector<double>& weights, const double cauchy_loss_scale=0.2, const bool use_field=true, const bool use_loss=true, const int num_threads=8, const std::vector<double>* inv_std=nullptr, const bool use_scale=false);

        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

        void setUseField(const bool use_field);

    private:
        std::vector<Vec3> pts_;
        std::vector<int> type_;
        const Mat4 prior_;
        MapDistField* map_;
        const std::vector<double>& weights_;
        const std::vector<double>* inv_std_ = nullptr;
        bool use_mahalanobis_ = false;
        bool use_scale_ = false;
        bool use_field_;
        int num_threads_ = 8;

        std::unique_ptr<ceres::LossFunction> loss_function_;


};

// Refreshes the Mahalanobis scaling of the registration residuals at every evaluation point of the
// solver. The covariance of a point is projected onto the surface normal, which changes as the pose
// moves, so the scaling is a function of the pose being evaluated.
//
// Doing it here rather than inside RegistrationCostFunction::Evaluate is what makes it consistent:
// ceres evaluates the cost function either with or without its jacobian, and only the latter has the
// gradient of the field at hand. This callback runs before either kind of evaluation, with the
// parameter block already set to the point about to be evaluated, so both see the same scaling.
//
// Note that the jacobian still ignores the derivative of the scaling itself (it would need the
// hessian of the distance field): the scaling is constant *within* an evaluation, which makes this an
// iteratively reweighted least squares scheme.
class PointCovarianceCallback: public ceres::EvaluationCallback
{
    public:
        // `scan_scale` points at the scale parameter block when it is optimised, so that the field is
        // queried where the scaled points actually are; null when the scale is not estimated.
        PointCovarianceCallback(MapDistField* map, const std::vector<Pointd>& pts, const std::vector<Mat3>& pts_cov, const Mat4& prior, const Vec6& pose_correction, const bool use_field, std::vector<double>& inv_std, const double* scan_scale = nullptr)
        : map_(map)
        , pts_(pts)
        , pts_cov_(pts_cov)
        , prior_(prior)
        , pose_correction_(pose_correction)
        , use_field_(use_field)
        , inv_std_(inv_std)
        , scan_scale_(scan_scale)
        {
        }

        virtual void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point) override;

    private:
        MapDistField* map_;
        const std::vector<Pointd>& pts_;
        const std::vector<Mat3>& pts_cov_;
        const Mat4 prior_;
        const Vec6& pose_correction_;
        const bool use_field_;
        std::vector<double>& inv_std_;
        const double* scan_scale_ = nullptr;
};

// Keeps the scale estimated for the current scan close to the one estimated for the previous scan.
// Nothing else pins the scale down: without this prior it would be a free multiplicative parameter of
// every registration, and it would happily absorb whatever the pose cannot explain. The weight is the
// inverse of the standard deviation of the scale change between two consecutive scans.
struct ScalePriorFunctor {
    ScalePriorFunctor(const double previous_scale, const double weight)
    : previous_scale_(previous_scale)
    , weight_(weight)
    {
    }

    template<typename T>
    bool operator()(const T* const scale, T* residuals) const
    {
        residuals[0] = T(weight_)*(scale[0] - T(previous_scale_));
        return true;
    }

    double previous_scale_;
    double weight_;
};

// Anchors the registration to the pose given as prior (the odometry): registerPts optimises a
// correction that is applied on top of that pose, so penalising the correction itself is a prior on
// the odometry. The translation and the rotation get their own weight, as they are not in the same
// units, and they are the inverse of the standard deviation the odometry is trusted with.
struct OdomPriorFunctor {
    OdomPriorFunctor(const double weight_pos, const double weight_rot)
    : weight_pos_(weight_pos)
    , weight_rot_(weight_rot)
    {
    }

    template<typename T>
    bool operator()(const T* const pose_correction, T* residuals) const
    {
        for(int i = 0; i < 3; ++i)
        {
            residuals[i] = T(weight_pos_) * pose_correction[i];
            residuals[i+3] = T(weight_rot_) * pose_correction[i+3];
        }
        return true;
    }

    double weight_pos_;
    double weight_rot_;
};

struct GravityFactorFunctor {
    GravityFactorFunctor(const Vec3& gravity_current, const Vec3& gravity_target, const double angle_stdev = 1.0)
    : gravity_current_(gravity_current)
    , gravity_target_(gravity_target)
    , angle_stdev_(angle_stdev)
    {
        pose_base_(3,3) = -1.0;
    }

    template<typename T>
    bool operator()(const T* const pose_correction, T* residuals) const
    {
        if(pose_base_(3,3) < 0.0)
        {
            throw std::runtime_error("GravityFactorFunctor not properly initialized with pose base");
        }
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> rot_correction(pose_correction + 3);
        Eigen::Matrix<T, 3, 3> rot_base = pose_base_.block(0,0,3,3).template cast<T>();
        Eigen::Matrix<T, 3, 3> rot_correction_mat;
        ceres::AngleAxisToRotationMatrix(rot_correction.data(), rot_correction_mat.data());
        Eigen::Matrix<T, 3, 3> rot_corrected = rot_base * rot_correction_mat;

        Eigen::Matrix<T, 3, 1> gravity_corrected = rot_corrected * gravity_current_.template cast<T>();

        // Compute the angle between the corrected gravity and the target gravity
        T cos_angle = gravity_corrected.dot(gravity_target_.template cast<T>()) / (gravity_corrected.norm() * gravity_target_.norm());
        if (cos_angle > T(1.0)) cos_angle = T(1.0);
        if (cos_angle < T(-1.0)) cos_angle = T(-1.0);
        T angle_diff = ceres::acos(cos_angle);

        // Scale the residual by the angle stdev
        residuals[0] = angle_diff / T(angle_stdev_);
        return true;
    }

    void setPoseBase(const Mat4& pose_base)
    {
        pose_base_ = pose_base;
    }
    



    private:
        Vec3 gravity_current_;
        Vec3 gravity_target_;
        Mat4 pose_base_;
        double angle_stdev_ = 1.0;

};
