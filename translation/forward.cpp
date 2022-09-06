/* -----------------------------begin for forward---------------------------------  */
template<typename T>
void pre_calc_for_bilinear(const int h, const int w, const int pool_h, const int pool_w, int b_grid_h, int b_grid_w,
                           T start_y, T start_x, T b_size_h, T b_size_w, vector<PreCalc<T>> &pre_calc) {
    int idx = 0;
    // 开始遍历每个bin
    for (int ph = 0; ph < pool_h; ++ph) {
        for (int pw = 0; pw < pool_w; ++pw) {
            for (int iy = 0; iy < b_grid_h; ++iy) {
                // 为没个bin采样四个点 其位置相对于每个bin的坐标位置为 (0.25, 0.25) (0.25, 0.75) (0.75, 0.25) (0.75, 0.75)
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
                    // x y 向下取整
                    int y_low = (int) y;
                    int x_low = (int) x;
                    int y_high = y_low >= h - 1 ? y_low : y_low + 1;
                    int x_high = x_low >= w - 1 ? x_low : x_low + 1;
                    // 这里就是双线性插值公式了 low 就是 f(0, 0) high 就是 f(1, 1)
                    // f(x,y)=f(0,0)(1-x)(1-y)+f(1,0)x(1-y)+f(0,1)(1-x)y+f(1,1)xy
                    T ly = y - y_low, lx = x - x_low;
                    T hy = 1.0 - ly, hx = 1.0 - lx;
                    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                    // in the feature map's position and correspond weights
                    PreCalc<T> pc;
                    // 这四个点就是对应的 f(0, 0) f(1, 0) f(0,1) f(1, 1)
                    // 应为这里是返回该点所在feature_map上的索引位置 不是坐标位置所以进行转换  y * width + x
                    // 至于为什么是y * width + x  就和feature_map的数据存储有关了  因为feature_map存储格式是一维数组
                    // 长度为 n * c * h * w  这里我们只需要考虑具体的某一张特征图  h * w 
                    // 对应矩阵 (h, w )中每一点转到一维数组的坐标就是 y * width + x
                    // 有人要问 n, c? 哪里去了  这很简单因为这里不需要计算 因为对应一个roi来说 在对应feature_map每个通道上的位置都是一样的
                    // 所以这里是把双线性插值的计算方法抽取出来公用 
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
void roi_align_forward(const T *feat, const T *rois, const vector<int64_t> &feat_size,
                       const vector<int64_t> &rois_size, const T &scale, const int ratio, T *out) {
    const int n_rois = rois_size[0], col_rois = rois_size[1], pool_h = rois_size[2], pool_w = rois_size[3];
    const int channel = feat_size[1], h = feat_size[2], w = feat_size[3];
    /***
     * n_rois 表示的是有多少个roi (region of interest)
     * col_rois 表示的是一个rois是几列组成的 如果是4列就是(x1, y1, x2, y2) 如果是5列(batch_id, x1, y1, x2, y2)
     * pool_h pool_w 池化后的高宽
     * channel 通道数 
     * h, w feature_maps的尺寸 也就是特征图的大小
     * 这里我们主要注意下输出的格式
     * T * out 可以理解为一个数组 数组长度 n * c * pool_h * pool_w
     * 可以理解为 (n, c, h, w)的矩阵reshape成(-1, 1) 这种存储数据格式是计算方便  因为我们会将这个大数组用连续内存存储
     * 因为我们会频繁的操作这些数据 所以用连续的内存块存储存取都更方便这样每一只需要指针移动一步就是读取下一个数据，
     * 这种存储结构在正向反向传播是更方便
     * 
     * 同理对于feature_maps也就是特征图也是上面的存储格式也存在 一个 n * c * h * w 的数组中
     ***/
    // #pragma omp parallel for
    for (int n = 0; n < n_rois; ++n) {
        // 知道了数据的存储格式也就是说 对于每一个roi我们需要分配  c * pool_h * pool_w 长度的数组
        // n * channel * pool_h * pool_w 表示数组的开始位置索引
        int idx_n = n * channel * pool_h * pool_w;
        // rois data
        // 这里的 rois是一个指针只想了存储rois数组的指针 也是上面类似的存储格式  存储在 col_rois * n_rois长度的数组中  
        // 每col_rois列表示一个roi的位置信息
        // 下面的代码表示指针在数组上移动col_rois个位置
        const T *offset_rois = rois + col_rois * n;
        int roi_batch_idx = 0;
        if (col_rois == 5) {
            // 如果col_rois为5 第一个为batch_id 
            roi_batch_idx = offset_rois[0];
            // offset_rois移动到下一个位置 跳过第一个位置batch_id
            ++offset_rois;
        }
        // Do not using rounding; this implementation detail is critical
        // 这里是将roi坐标(x1, y1, x2, y2)映射到特征图上
        T start_x = offset_rois[0] * scale;
        T start_y = offset_rois[1] * scale;
        T end_x = offset_rois[2] * scale;
        T end_y = offset_rois[3] * scale;
 
        // Force malformed ROIs to be 1x1
        // 计算roi映射到特征图后的的宽高
        T roi_w = std::max(end_x - start_x, (T) 1.);
        T roi_h = std::max(end_y - start_y, (T) 1.);
        // 这表示每个bin的大小  将特征图分成pool_w * pool_h个区间 每个区间就是一个bin
        T bin_size_w = roi_w / static_cast<T>(pool_w);
        T bin_size_h = roi_h / static_cast<T>(pool_h);
 
        // We use roi_bin_grid to sample the grid and mimic integral
        // 这个表示每个bin的采样个数  论文中是采样四个点(w 上两个  h上两个) 然后最大池化 这里是取得平均池化
        // 如果没有设置采样个数就用 roi_h / pool_h
        int bin_grid_h = (ratio > 0) ? ratio : std::ceil(roi_h / pool_h);
        int bin_grid_w = (ratio > 0) ? ratio : std::ceil(roi_w / pool_w);
        // We do average (integral) pooling inside a bin
        // 这里其实就是 4
        const T count = bin_grid_h * bin_grid_w;
        // get each bin's corresponding position and weights
        // 计算双线性差值 这里只计算每个bin的 四个采样点位置和权重 
        // 这里返回的是一个vector  vector长度为  pool_h * pool_w * 4  
        // 总共有 pool_h * pool_w 个bin  每个bin采样四个点   每个点都采用双线性插值计算该采样点值
        // vector中的数据格式  PreCalc<T> pc; 这里面存了 四个点在feature_map中的位置信息 和四个点的权重
        // 利用双线性插值公式可以计算出采样点的真的数组
        std::vector<PreCalc<T>> pre_calc(count * pool_h * pool_w);
        pre_calc_for_bilinear(h, w, pool_h, pool_w, bin_grid_h, bin_grid_w, start_y, start_x, bin_size_h, bin_size_w,
                              pre_calc);
        // map to feature map
        for (int c = 0; c < channel; ++c) {
            // 遍历通道  idx_nc 表示第 n个 roi 第c个通道所在数组起始位置
            int idx_nc = idx_n + c * pool_w * pool_h;
            int offset_feat = (roi_batch_idx * channel + c) * h * w;
            // pre_calc_idx用来计数遍历到哪个bin了
            int pre_calc_idx = 0;
            // 遍历 pool_h pool_w  也就是遍历每个bin
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    // 每个bin在返回out数组中的索引位置
                    int idx = idx_nc + ph * pool_w + pw;
                    T output_val = 0.;
                    // 这里是将四个采样点相加 除以count 也就是平均池化
                    for (int iy = 0; iy < bin_grid_h; ++iy) {
                        for (int ix = 0; ix < bin_grid_w; ++ix) {
                            // 取出该bin对应的双线性插值位置和权重计算输出值
                            PreCalc<T> pc = pre_calc[pre_calc_idx];
                            output_val += pc.w1 * feat[offset_feat+pc.pos1] + pc.w2 * offset_feat[pc.pos2] +
                                          pc.w3 * offset_feat[pc.pos3] + pc.w4 * offset_feat[pc.pos4];
                            pre_calc_idx += 1;
                        }
                    }
                    // 这里就是直接赋值
                    output_val /= count;
                    out[idx] = output_val;
                }  // for pw
            } // for ph
        } // for c
    }  // for rois_n
}
