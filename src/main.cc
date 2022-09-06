#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <cnpy.h>
#include <gflags/gflags.h>

using namespace std;

DEFINE_string(input_align_feat, "../data/align_feat.npy", "input feat data file");
DEFINE_string(input_align_rois, "../data/align_rois.npy", "input rois data file");
DEFINE_int32(col_rois, 5, "4: (x1, y1, x2, y2), 5: (batch_id, x1, y1, x2, y2)");

template<typename T>
struct PreCalc {
    // left_top, right_top, left_bottom, right_bottom
    int pos1, pos2, pos3, pos4;
    T w1, w2, w3, w4;
};

template<typename T>
void pre_calc_for_bilinear(const int h, const int w, const int pool_h, const int pool_w, int b_grid_h, int b_grid_w,
                           T start_y, T start_x, T b_size_h, T b_size_w, vector<PreCalc<T>> &pre_calc) {
    int idx = 0;
    for (int ph = 0; ph < pool_h; ++ph) {
        for (int pw = 0; pw < pool_w; ++pw) {
            for (int iy = 0; iy < b_grid_h; ++iy) {
                const T yy = start_y + ph * b_size_h + static_cast<T>(iy + 0.5f) * b_size_h / static_cast<T>(b_grid_h);
                for (int ix = 0; ix < b_grid_w; ++ix) {
                    const T xx =
                            start_x + pw * b_size_w + static_cast<T>(ix + 0.5f) * b_size_w / static_cast<T>(b_grid_w);
                    T x = xx, y = yy;
                    // situation 1: out of range
                    if (y < -1.0 || y > h || x < -1.0 || x > w) {
                        PreCalc<T> pc{0, 0, 0, 0, 0, 0, 0, 0};
                        pre_calc[idx] = pc;
                        idx += 1;
                        continue;
                    }
                    // not exceed 1.0
                    y = y <= 0 ? 0 : (y >= h - 1 ? h - 1 : y);
                    x = x <= 0 ? 0 : (x >= w - 1 ? w - 1 : x);
                    int y_low = (int) y;
                    int x_low = (int) x;
                    int y_high = y_low >= h - 1 ? y_low : y_low + 1;
                    int x_high = x_low >= w - 1 ? x_low : x_low + 1;
                    T ly = y - y_low, lx = x - x_low;
                    T hy = 1.0 - ly, hx = 1.0 - lx;
                    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                    // in the feature map's position and correspond weights
                    PreCalc<T> pc;
                    pc.pos1 = y_low * w + x_low;
                    pc.pos2 = y_low * w + x_high;
                    pc.pos3 = y_high * w + x_low;
                    pc.pos4 = y_high * w + x_high;
                    pc.w1 = w1, pc.w2 = w2, pc.w3 = w3, pc.w4 = w4;
                    pre_calc[idx] = pc;
                    idx += 1;
                } // b_grid_w
            } // b_grid_h
        } // pool_w
    } // pool_h
}

template<typename T>
void roi_align_forward(const vector<T> &feat, const vector<T> &rois, const vector<int32_t> &feat_size,
                       const vector<int32_t> &rois_size, const T &scale, const int ratio,vector<T> &out) {
    const int n_rois = rois_size[0], col_rois = rois_size[1], pool_h = rois_size[2], pool_w = rois_size[3];
    const int channel = feat_size[1], h = feat_size[2], w = feat_size[3];
    // #pragma omp parallel for
    for (int n = 0; n < n_rois; ++n) {
        int idx_n = n * channel * pool_h * pool_w;
        // rois data
        int offset_rois = col_rois * n;
        int roi_batch_idx = 0;
        if (col_rois == 5) {
            roi_batch_idx = rois[offset_rois];
            ++offset_rois;
        }
        // Do not using rounding; this implementation detail is critical
        T start_x = rois[offset_rois] * scale;
        T start_y = rois[offset_rois+1] * scale;
        T end_x = rois[offset_rois+2] * scale;
        T end_y = rois[offset_rois+3] * scale;

        // Force malformed ROIs to be 1x1
        T roi_w = std::max(end_x - start_x, (T) 1.);
        T roi_h = std::max(end_y - start_y, (T) 1.);
        T bin_size_w = roi_w / static_cast<T>(pool_w);
        T bin_size_h = roi_h / static_cast<T>(pool_h);

        // We use roi_bin_grid to sample the grid and mimic integral
        int bin_grid_h = (ratio > 0) ? ratio : std::ceil(roi_h / pool_h);
        int bin_grid_w = (ratio > 0) ? ratio : std::ceil(roi_w / pool_w);
        // We do average (integral) pooling inside a bin
        const T count = bin_grid_h * bin_grid_w;
        // get each bin's corresponding position and weights
        std::vector<PreCalc<T>> pre_calc(count * pool_h * pool_w);
        pre_calc_for_bilinear(h, w, pool_h, pool_w, bin_grid_h, bin_grid_w, start_y, start_x, bin_size_h, bin_size_w,
                              pre_calc);
        // map to feature map
        for (int c = 0; c < channel; ++c) {
            int idx_nc = idx_n + c * pool_w * pool_h;
            int offset_feat = (roi_batch_idx * channel + c) * h * w;
            int pre_calc_idx = 0;
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int idx = idx_nc + ph * pool_w + pw;
                    T output_val = 0.;
                    for (int iy = 0; iy < bin_grid_h; ++iy) {
                        for (int ix = 0; ix < bin_grid_w; ++ix) {
                            PreCalc<T> pc = pre_calc[pre_calc_idx];
                            output_val += pc.w1 * feat[offset_feat+pc.pos1] + pc.w2 * feat[offset_feat+pc.pos2] +
                                          pc.w3 * feat[offset_feat+pc.pos3] + pc.w4 * feat[offset_feat+pc.pos4];
                            pre_calc_idx += 1;
                        }
                    }
                    output_val /= count;
                    out.push_back(output_val);
                    std::cout << output_val << "\t";
                }  // for pw
                cout<<endl;
            } // for ph
        } // for c
    }  // for rois_n
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    cnpy::NpyArray arr_feat = cnpy::npy_load(FLAGS_input_align_feat);
    cnpy::NpyArray arr_rois = cnpy::npy_load(FLAGS_input_align_rois);
    float* loaded_feat = arr_feat.data<float>();
    float* loaded_rois = arr_rois.data<float>();
    //init feat
    std::vector<float> feat;
    for (int i = 0; i < arr_feat.num_vals; ++i) {
            feat.push_back(loaded_feat[i]);
    }
    vector<int32_t> feat_size;
    for (int i = 0; i < arr_feat.shape.size(); i++) {
        feat_size.push_back(arr_feat.shape[i]);
    }

    //init rois
    std::vector<float> rois;
    for (int i = 0; i < arr_rois.num_vals; ++i) {
            rois.push_back(loaded_rois[i]);
    }

    int pool_h = 4, pool_w = 4, sample=1;
    float scale = 1;
    const vector<int32_t> rois_size = {sample, FLAGS_col_rois, pool_h, pool_w};

    std::vector<float> output;
    roi_align_forward<float>(feat, rois, feat_size, rois_size, static_cast<float>(scale), sample,
                      output);
    return 0;
}
