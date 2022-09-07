template<class T>
inline void add(const T &val, T *address) {
    // 这个函数就很简单了 就是累加值    
    *address += val;
}
 
template<typename T>
void bilinear_interpolate_gradient(const int h, const int w, T y, T x, PreCalc<T> &pc) {
    if (y < -1.0 || y > h || x < -1.0 || x > w) {
        pc = {-1, -1, -1, -1, 0., 0., 0., 0.};
        return;
    }
    // 计算该样本点对应的4个用于双线性插值的点的位置和权重
    // not exceed 1.0
    y = y <= 0 ? 0 : (y >= h - 1 ? h - 1 : y);
    x = x <= 0 ? 0 : (x >= w - 1 ? w - 1 : x);
    int y_low = (int) y;
    int x_low = (int) x;
    int y_high = y_low >= h - 1 ? y_low : y_low + 1;
    int x_high = x_low >= w - 1 ? x_low : x_low + 1;
    // 得到四个点在feature_map上的位置和计算的权重
    pc.pos1 = y_low * w + x_low;
    pc.pos2 = y_low * w + x_high;
    pc.pos3 = y_high * w + x_low;
    pc.pos4 = y_high * w + x_high;
    T ly = y - y_low, lx = x - x_low;
    T hy = 1.0 - ly, hx = 1.0 - lx;
    pc.w1 = hy * hx, pc.w2 = hy * lx, pc.w3 = ly * hx, pc.w4 = ly * lx;
}
 
 
template<typename T>
void roi_align_backward(int total, const T *rois, T *grad_out, const T &scale, const vector<int64_t> feat_size,
                        const int pool_h, const int pool_w, const int rois_col, const int sample, T *grad_in) {
    // total=nxcxphxpw
    auto channel = feat_size[0], h = feat_size[1], w = feat_size[2];
    // 从idx 反推 n c pool_h pool_w   
    // 我们可以从forward看出 output是个数组  长度为  n * c * h * w
    for (int idx = 0; idx < total; ++idx) {
        int pw = idx % pool_w;
        int ph = (idx / pool_w) % pool_h;
        int c = (idx / pool_h / pool_w) % channel;
        int n = idx / pool_h / pool_w / channel;
        // 这里和forward是一致的
        const T *offset_rois = rois + n * rois_col;
        int roi_batch_idx = 0;
        if (rois_col == 5) {
            roi_batch_idx = offset_rois[0];
            ++offset_rois;
        }
        // Do not using rounding; this implementation detail is critical
        // 这里和forward是一致的 将roi的坐标映射到feature_map特征图上
        T start_x = offset_rois[0] * scale;
        T start_y = offset_rois[1] * scale;
        T end_x = offset_rois[2] * scale;
        T end_y = offset_rois[3] * scale;
 
        // Force malformed ROIs to be 1x1
        // 这里和forward是一致的
        T roi_w = std::max(end_x - start_x, (T) 1.0);
        T roi_h = std::max(end_y - start_y, (T) 1.0);
        T b_size_h = roi_h / static_cast<T>(pool_h);
        T b_size_w = roi_w / static_cast<T>(pool_w);
        
        // 注意这里  grad_in是指针数组存储了输入梯度  长度为 n * c * h * w 对应feature_map中各值的梯度
        // offset_grad_in 指向了当前的特征图第n张图片 第c个通道的featuer_map的梯度起始位置
        T *offset_grad_in = grad_in + (roi_batch_idx * channel + c) * h * w;
        // 注意这里  grad_out是指针数组存储了输出梯度  长度为 n * c * pool_h * pool_w 对应roialign后feature_map中各值的梯度
        // offset_grad_in 指向了当前的特征图第n张图片 第c个通道的roialign后ffeatuer_map的梯度起始位置
        T *offset_grad_out = grad_out + (n * channel + c) * pool_h * pool_w;
         // grad_out_this_bin 表示指向了在roialign后梯度特征图上的具体位置
        T grad_out_this_bin = offset_grad_out[ph * pool_w + pw];
 
        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sample > 0) ? sample : std::ceil(roi_h / pool_h);
        int roi_bin_grid_w = (sample > 0) ? sample : std::ceil(roi_w / pool_w);
        // We do average (integral) pooling inside a bin
        const int count = roi_bin_grid_h * roi_bin_grid_w;
        PreCalc<T> pc;
        // 计算梯度反传  遍历grad_out_this_bin指向的位置的四个采样点
        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            const T y = start_y + ph * b_size_h +
                        static_cast<T>(iy + .5f) * b_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const T x = start_x + pw * b_size_w +
                            static_cast<T>(ix + .5f) * b_size_w / static_cast<T>(roi_bin_grid_w);
                // 得到用于计算每个采样点的值得 4个用于双线性差值的四个点的位置和权重信息
                bilinear_interpolate_gradient(h, w, y, x, pc);
                // 将梯度反传到拥有计算双线性差值的四个点
                T g1 = grad_out_this_bin * pc.w1 / count;
                T g2 = grad_out_this_bin * pc.w2 / count;
                T g3 = grad_out_this_bin * pc.w3 / count;
                T g4 = grad_out_this_bin * pc.w4 / count;
                // update grad_out
                if (pc.pos1 >= 0 && pc.pos2 >= 0 && pc.pos3 >= 0 && pc.pos4 >= 0) {
                    // 将梯度累加到对应输入位置  因为该点可能参与了多次计算所以是需要累加的
                    // 所有用到过该点的梯度度需要反传
                    add(g1, offset_grad_in + pc.pos1);
                    add(g2, offset_grad_in + pc.pos2);
                    add(g3, offset_grad_in + pc.pos3);
                    add(g4, offset_grad_in + pc.pos4);
                }
            }  // for ix
        }  // for iy
    }  // for
}
 
 

