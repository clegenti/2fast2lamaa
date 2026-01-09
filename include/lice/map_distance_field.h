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

class MapDistField;

struct AlphaBlock
{
    VecX alpha;
    MatX neighbor_pts;
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
        std::tuple<MatX, MatX, MatX, MatX> kernelRQAndDiff(const MatX& X1, const MatX& X2);
        

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

typedef Cell* CellPtr;



struct MapDistFieldOptions {
    double cell_size = 0.15;
    int neighborhood_size = 2;
    double gp_sigma_z = 0.05;
    double gp_lengthscale = -1.0;
    bool use_voxel_weights = true;
    bool use_temporal_weights = false;
    bool free_space_carving = false;
    double free_space_carving_radius = -1.0;
    bool over_reject = false;
    double min_range = 0.0001;
    double max_range = std::numeric_limits<double>::max();
    bool last_scan_carving = false;
    bool edge_field = true;
    std::string scan_folder = "";
};




class MapDistField {
    private:
        int num_neighbors_ = 2;

        ankerl::unordered_dense::set<GridIndex> free_space_cells_;

        std::unique_ptr<HashMap<CellPtr> > hash_map_;
        std::unique_ptr<ankerl::unordered_dense::set<CellPtr> > hash_map_edge_;
        const double cell_size_;
        const double inv_cell_size_;
        const float cell_size_f_;
        const float half_cell_size_f_;
        const int dim_ = 3;

        std::mutex clean_mutex_;
        ankerl::unordered_dense::set<GridIndex> cells_to_clean_;

        thuni::Octree ioctree_;
        thuni::Octree ioctree_edge_;

        std::vector<Pointd> prev_scan_;
        Mat4 prev_pose_;

        size_t num_cells_ = 0;

        double path_length_ = 0.0;

        std::tuple<std::vector<Vec3>, std::set<std::array<int, 2> > > getPointsAndEdges();

        int scan_counter_ = -1;

        bool has_color_ = false;
        
        double last_time_register_ = -1.0;
        int64_t time_offset_ = -1;

        MapDistFieldOptions opt_;

        void cleanCells();

        std::pair<ankerl::unordered_dense::set<GridIndex>, std::vector<bool> > getFreeSpaceCellsToRemove(const std::vector<Pointd>& scan, const std::vector<Vec3>& map_pts, const Mat4& pose_scan, const Mat4& pose_map);

        std::vector<Vec3> getNeighborPoints(const Vec3& pt, const double radius);


        void writePly(const std::string& filename, const std::vector<Vec3>& pts, const std::vector<Vec3>& normals, const std::vector<std::array<unsigned char, 3>>& colors, const std::vector<double>& count, const std::vector<double>& intensity, const std::vector<double>& types, const std::vector<Eigen::Vector3i>& faces) const;


        void calibrateUncertaintyProxy();


    public:
        GPCellHyperparameters cell_hyperparameters;
        MapDistField(const MapDistFieldOptions& options);

        ~MapDistField();

        void clear();

        Mat4 registerPts(const std::vector<Pointd>& pts, const Mat4& prior, const double current_time, const bool approximate=false, const double loss_scale=0.5, const int max_iterations=12);

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

        std::vector<CellPtr> getNeighborCells(const Vec3& pt);
        
        void cellToClean(const GridIndex& index);


        double getPathLength() const { return path_length_; }

        std::vector<Pointd> freeSpaceCarving(const std::vector<Pointd>& pts, const Mat4& pose);

};



class RegistrationCostFunction: public ceres::CostFunction
{
    public:
        RegistrationCostFunction(const std::vector<Pointd>& pts, const Mat4& prior, MapDistField* map, const std::vector<double>& weights, const double cauchy_loss_scale=0.2, const bool use_field=true, const bool use_loss=true);
        
        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

        void setUseField(const bool use_field);

    private:
        std::vector<Vec3> pts_;
        std::vector<int> type_;
        const Mat4 prior_;
        MapDistField* map_;
        const std::vector<double>& weights_;
        bool use_field_;

        std::unique_ptr<ceres::LossFunction> loss_function_;


};
