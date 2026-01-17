// Core SIFT implementation with OpenMP acceleration
// - Keep math/algorithm identical to reference for correctness
// - Use thread-local buffers; avoid false sharing
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <tuple>
#include <cassert>
#include <omp.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "sift.hpp"
#include "image.hpp"



ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min,
                                            int num_octaves, int scales_per_octave)
{
    // assume initial sigma is 1.0 (after resizing) and smooth
    // the image with sigma_diff to reach requried base_sigma
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;

    // determine sigma values for blurring between adjacent scales
    float k = std::pow(2, 1.0/scales_per_octave);
    std::vector<float> sigma_vals {base_sigma};
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
    }

    // create a scale space pyramid of gaussian images
    // images in each octave are half the size of images in the previous one
    ScaleSpacePyramid pyramid = {
        num_octaves,
        imgs_per_octave,
        std::vector<std::vector<Image>>(num_octaves)
    };
    
    // Pre-allocate memory for better performance
    for (int i = 0; i < num_octaves; i++) {
        pyramid.octaves[i].reserve(imgs_per_octave);
    }
    
    for (int i = 0; i < num_octaves; i++) {
        pyramid.octaves[i].push_back(std::move(base_img));
        
        // Sequential push with repeated blur from last produced image (spec-consistent)
        for (int j = 1; j < sigma_vals.size(); j++) {
            const Image& prev_img = pyramid.octaves[i].back();
            pyramid.octaves[i].push_back(gaussian_blur(prev_img, sigma_vals[j]));
        }
        
        // prepare base image for next octave
        const Image& next_base_img = pyramid.octaves[i][imgs_per_octave-3];
        base_img = next_base_img.resize(next_base_img.width/2, next_base_img.height/2,
                                        Interpolation::NEAREST);
    }
    return pyramid;
}

// generate pyramid of difference of gaussians (DoG) images
ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid)
{
    ScaleSpacePyramid dog_pyramid = {
        img_pyramid.num_octaves,
        img_pyramid.imgs_per_octave - 1,
        std::vector<std::vector<Image>>(img_pyramid.num_octaves)
    };
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        dog_pyramid.octaves[i].reserve(dog_pyramid.imgs_per_octave);
        for (int j = 1; j < img_pyramid.imgs_per_octave; j++) {
            Image diff = img_pyramid.octaves[i][j];
            #pragma omp parallel for
            for (int pix_idx = 0; pix_idx < diff.size; pix_idx++) {
                diff.data[pix_idx] -= img_pyramid.octaves[i][j-1].data[pix_idx];
            }
            dog_pyramid.octaves[i].push_back(diff);
        }
    }
    return dog_pyramid;
}

bool point_is_extremum(const std::vector<Image>& octave, int scale, int x, int y)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    bool is_min = true, is_max = true;
    float val = img.get_pixel(x, y, 0), neighbor;

    for (int dx : {-1,0,1}) {
        for (int dy : {-1,0,1}) {
            neighbor = prev.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = next.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = img.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            if (!is_min && !is_max) return false;
        }
    }
    return true;
}

// fit a quadratic near the discrete extremum,
// update the keypoint (interpolated) extremum value
// and return offsets of the interpolated extremum from the discrete extremum
std::tuple<float, float, float> fit_quadratic(Keypoint& kp,
                                              const std::vector<Image>& octave,
                                              int scale)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    float g1, g2, g3;
    float h11, h12, h13, h22, h23, h33;
    int x = kp.i, y = kp.j;

    // gradient 
    g1 = (next.get_pixel(x, y, 0) - prev.get_pixel(x, y, 0)) * 0.5;
    g2 = (img.get_pixel(x+1, y, 0) - img.get_pixel(x-1, y, 0)) * 0.5;
    g3 = (img.get_pixel(x, y+1, 0) - img.get_pixel(x, y-1, 0)) * 0.5;

    // hessian
    h11 = next.get_pixel(x, y, 0) + prev.get_pixel(x, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h33 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (next.get_pixel(x+1, y, 0) - next.get_pixel(x-1, y, 0)
          -prev.get_pixel(x+1, y, 0) + prev.get_pixel(x-1, y, 0)) * 0.25;
    h13 = (next.get_pixel(x, y+1, 0) - next.get_pixel(x, y-1, 0)
          -prev.get_pixel(x, y+1, 0) + prev.get_pixel(x, y-1, 0)) * 0.25;
    h23 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;
    
    // invert hessian
    float hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
    float det = h11*h22*h33 - h11*h23*h23 - h12*h12*h33 + 2*h12*h13*h23 - h13*h13*h22;
    hinv11 = (h22*h33 - h23*h23) / det;
    hinv12 = (h13*h23 - h12*h33) / det;
    hinv13 = (h12*h23 - h13*h22) / det;
    hinv22 = (h11*h33 - h13*h13) / det;
    hinv23 = (h12*h13 - h11*h23) / det;
    hinv33 = (h11*h22 - h12*h12) / det;

    // find offsets of the interpolated extremum from the discrete extremum
    float offset_s = -hinv11*g1 - hinv12*g2 - hinv13*g3;
    float offset_x = -hinv12*g1 - hinv22*g2 - hinv23*g3;
    float offset_y = -hinv13*g1 - hinv23*g3 - hinv33*g3;

    float interpolated_extrema_val = img.get_pixel(x, y, 0)
                                   + 0.5*(g1*offset_s + g2*offset_x + g3*offset_y);
    kp.extremum_val = interpolated_extrema_val;
    return {offset_s, offset_x, offset_y};
}

bool point_is_on_edge(const Keypoint& kp, const std::vector<Image>& octave, float edge_thresh=C_EDGE)
{
    const Image& img = octave[kp.scale];
    float h11, h12, h22;
    int x = kp.i, y = kp.j;
    h11 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;

    float det_hessian = h11*h22 - h12*h12;
    float tr_hessian = h11 + h22;
    float edgeness = tr_hessian*tr_hessian / det_hessian;

    if (edgeness > std::pow(edge_thresh+1, 2)/edge_thresh)
        return true;
    else
        return false;
}

void find_input_img_coords(Keypoint& kp, float offset_s, float offset_x, float offset_y,
                                   float sigma_min=SIGMA_MIN,
                                   float min_pix_dist=MIN_PIX_DIST, int n_spo=N_SPO)
{
    kp.sigma = std::pow(2, kp.octave) * sigma_min * std::pow(2, (offset_s+kp.scale)/n_spo);
    kp.x = min_pix_dist * std::pow(2, kp.octave) * (offset_x+kp.i);
    kp.y = min_pix_dist * std::pow(2, kp.octave) * (offset_y+kp.j);
}

bool refine_or_discard_keypoint(Keypoint& kp, const std::vector<Image>& octave,
                                float contrast_thresh, float edge_thresh)
{
    int k = 0;
    bool kp_is_valid = false; 
    while (k++ < MAX_REFINEMENT_ITERS) {
        auto [offset_s, offset_x, offset_y] = fit_quadratic(kp, octave, kp.scale);

        float max_offset = std::max({std::abs(offset_s),
                                     std::abs(offset_x),
                                     std::abs(offset_y)});
        // find nearest discrete coordinates
        kp.scale += std::round(offset_s);
        kp.i += std::round(offset_x);
        kp.j += std::round(offset_y);
        if (kp.scale >= octave.size()-1 || kp.scale < 1)
            break;

        bool valid_contrast = std::abs(kp.extremum_val) > contrast_thresh;
        if (max_offset < 0.6 && valid_contrast && !point_is_on_edge(kp, octave, edge_thresh)) {
            find_input_img_coords(kp, offset_s, offset_x, offset_y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}

std::vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid, float contrast_thresh,
                                     float edge_thresh)
{
    std::vector<Keypoint> keypoints;
    // Parallel keypoint detection (doesn't affect descriptor precision)
    std::vector<std::vector<Keypoint>> thread_keypoints(omp_get_max_threads());
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < dog_pyramid.num_octaves; i++) {
            for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
                const std::vector<Image>& octave = dog_pyramid.octaves[i];
                const Image& img = octave[j];
                for (int x = 1; x < img.width-1; x++) {
                    for (int y = 1; y < img.height-1; y++) {
                        if (std::abs(img.get_pixel(x, y, 0)) < 0.8*contrast_thresh) {
                            continue;
                        }
                        if (point_is_extremum(octave, j, x, y)) {
                            Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                            bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh,
                                                                          edge_thresh);
                            if (kp_is_valid) {
                                thread_keypoints[thread_id].push_back(kp);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Combine results from all threads
    for (const auto& thread_kps : thread_keypoints) {
        keypoints.insert(keypoints.end(), thread_kps.begin(), thread_kps.end());
    }
    return keypoints;
}

    // calculate x and y derivatives for all images in the input pyramid
ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid)
{
    ScaleSpacePyramid grad_pyramid = {
        pyramid.num_octaves,
        pyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(pyramid.num_octaves)
    };
    for (int i = 0; i < pyramid.num_octaves; i++) {
        grad_pyramid.octaves[i].reserve(grad_pyramid.imgs_per_octave);
        int width = pyramid.octaves[i][0].width;
        int height = pyramid.octaves[i][0].height;
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            Image grad(width, height, 2);
            float gx, gy;
            #pragma omp parallel for collapse(2) private(gx, gy)
            for (int x = 1; x < grad.width-1; x++) {
                for (int y = 1; y < grad.height-1; y++) {
                    gx = (pyramid.octaves[i][j].get_pixel(x+1, y, 0)
                         -pyramid.octaves[i][j].get_pixel(x-1, y, 0)) * 0.5;
                    grad.set_pixel(x, y, 0, gx);
                    gy = (pyramid.octaves[i][j].get_pixel(x, y+1, 0)
                         -pyramid.octaves[i][j].get_pixel(x, y-1, 0)) * 0.5;
                    grad.set_pixel(x, y, 1, gy);
                }
            }
            grad_pyramid.octaves[i].push_back(grad);
        }
    }
    return grad_pyramid;
}

// convolve 6x with box filter
void smooth_histogram(float hist[N_BINS])
{
    float tmp_hist[N_BINS];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < N_BINS; j++) {
            int prev_idx = (j-1+N_BINS)%N_BINS;
            int next_idx = (j+1)%N_BINS;
            tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
        }
        for (int j = 0; j < N_BINS; j++) {
            hist[j] = tmp_hist[j];
        }
    }
}

std::vector<float> find_keypoint_orientations(Keypoint& kp, 
                                              const ScaleSpacePyramid& grad_pyramid,
                                              float lambda_ori, float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];

    // discard kp if too close to image borders 
    float min_dist_from_border = std::min({kp.x, kp.y, pix_dist*img_grad.width-kp.x,
                                           pix_dist*img_grad.height-kp.y});
    if (min_dist_from_border <= std::sqrt(2)*lambda_desc*kp.sigma) {
        return {};
    }

    float hist[N_BINS] = {};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3 * patch_sigma;
    int x_start = std::round((kp.x - patch_radius)/pix_dist);
    int x_end = std::round((kp.x + patch_radius)/pix_dist);
    int y_start = std::round((kp.y - patch_radius)/pix_dist);
    int y_end = std::round((kp.y + patch_radius)/pix_dist);

    // accumulate gradients in orientation histogram with thread-local histograms
    std::vector<float> local_hist[N_BINS];
    #pragma omp parallel
    {
        float local_hist_private[N_BINS] = {0};
        #pragma omp for collapse(2) nowait
        for (int x = x_start; x <= x_end; x++) {
            for (int y = y_start; y <= y_end; y++) {
                gx = img_grad.get_pixel(x, y, 0);
                gy = img_grad.get_pixel(x, y, 1);
                grad_norm = std::sqrt(gx*gx + gy*gy);
                weight = std::exp(-(std::pow(x*pix_dist-kp.x, 2)+std::pow(y*pix_dist-kp.y, 2))
                                  /(2*patch_sigma*patch_sigma));
                theta = std::fmod(std::atan2(gy, gx)+2*M_PI, 2*M_PI);
                bin = (int)std::round(N_BINS/(2*M_PI)*theta) % N_BINS;
                local_hist_private[bin] += weight * grad_norm;
            }
        }
        // Merge local histogram into global one
        #pragma omp critical
        {
            for (int i = 0; i < N_BINS; i++) {
                hist[i] += local_hist_private[i];
            }
        }
    }

    smooth_histogram(hist);

    // extract reference orientations
    float ori_thresh = 0.8, ori_max = 0;
    std::vector<float> orientations;
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] > ori_max) {
            ori_max = hist[j];
        }
    }
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] >= ori_thresh * ori_max) {
            float prev = hist[(j-1+N_BINS)%N_BINS], next = hist[(j+1)%N_BINS];
            if (prev > hist[j] || next > hist[j])
                continue;
            float theta = 2*M_PI*(j+1)/N_BINS + M_PI/N_BINS*(prev-next)/(prev-2*hist[j]+next);
            orientations.push_back(theta);
        }
    }
    return orientations;
}

void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y,
                       float contrib, float theta_mn, float lambda_desc)
{
    float x_i, y_j;
    const float hist_scale = 2*lambda_desc/N_HIST;
    const float ori_scale = N_ORI*0.5/M_PI;
    const float desc_scale = N_HIST*0.5/lambda_desc;
    
    for (int i = 1; i <= N_HIST; i++) {
        x_i = (i-(1+(float)N_HIST)/2) * hist_scale;
        if (std::abs(x_i-x) > hist_scale)
            continue;
        for (int j = 1; j <= N_HIST; j++) {
            y_j = (j-(1+(float)N_HIST)/2) * hist_scale;
            if (std::abs(y_j-y) > hist_scale)
                continue;
            
            float hist_weight = (1 - desc_scale*std::abs(x_i-x))
                               *(1 - desc_scale*std::abs(y_j-y));

            for (int k = 1; k <= N_ORI; k++) {
                float theta_k = 2*M_PI*(k-1)/N_ORI;
                float theta_diff = std::fmod(theta_k-theta_mn+2*M_PI, 2*M_PI);
                if (std::abs(theta_diff) >= 2*M_PI/N_ORI)
                    continue;
                float bin_weight = 1 - ori_scale*std::abs(theta_diff);
                hist[i-1][j-1][k-1] += hist_weight*bin_weight*contrib;
            }
        }
    }
}

void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI], std::array<uint8_t, 128>& feature_vec)
{
    int size = N_HIST*N_HIST*N_ORI;
    float *hist = reinterpret_cast<float *>(histograms);

    float norm = 0;
    for (int i = 0; i < size; i++) {
        norm += hist[i] * hist[i];
    }
    norm = std::sqrt(norm);
    float norm2 = 0;
    for (int i = 0; i < size; i++) {
        hist[i] = std::min(hist[i], 0.2f*norm);
        norm2 += hist[i] * hist[i];
    }
    norm2 = std::sqrt(norm2);
    for (int i = 0; i < size; i++) {
        float val = std::floor(512*hist[i]/norm2);
        feature_vec[i] = std::min((int)val, 255);
    }
}

void compute_keypoint_descriptor(Keypoint& kp, float theta,
                                 const ScaleSpacePyramid& grad_pyramid,
                                 float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    float histograms[N_HIST][N_HIST][N_ORI] = {0};

    //find start and end coords for loops over image patch
    float half_size = std::sqrt(2)*lambda_desc*kp.sigma*(N_HIST+1.)/N_HIST;
    int x_start = std::round((kp.x-half_size) / pix_dist);
    int x_end = std::round((kp.x+half_size) / pix_dist);
    int y_start = std::round((kp.y-half_size) / pix_dist);
    int y_end = std::round((kp.y+half_size) / pix_dist);

    float cos_t = std::cos(theta), sin_t = std::sin(theta);
    float patch_sigma = lambda_desc * kp.sigma;
    //accumulate samples into histograms (sequential for precision)
    for (int m = x_start; m <= x_end; m++) {
        for (int n = y_start; n <= y_end; n++) {
            // find normalized coords w.r.t. kp position and reference orientation
            float x = ((m*pix_dist - kp.x)*cos_t
                      +(n*pix_dist - kp.y)*sin_t) / kp.sigma;
            float y = (-(m*pix_dist - kp.x)*sin_t
                       +(n*pix_dist - kp.y)*cos_t) / kp.sigma;

            // verify (x, y) is inside the description patch
            if (std::max(std::abs(x), std::abs(y)) > lambda_desc*(N_HIST+1.)/N_HIST)
                continue;

            float gx = img_grad.get_pixel(m, n, 0), gy = img_grad.get_pixel(m, n, 1);
            float theta_mn = std::fmod(std::atan2(gy, gx)-theta+4*M_PI, 2*M_PI);
            float grad_norm = std::sqrt(gx*gx + gy*gy);
            float weight = std::exp(-(std::pow(m*pix_dist-kp.x, 2)+std::pow(n*pix_dist-kp.y, 2))
                                    /(2*patch_sigma*patch_sigma));
            float contribution = weight * grad_norm;

            update_histograms(histograms, x, y, contribution, theta_mn, lambda_desc);
        }
    }

    // build feature vector (descriptor) from histograms
    hists_to_vec(histograms, kp.descriptor);
}

std::vector<Keypoint> find_keypoints_and_descriptors(const Image& img, float sigma_min,
                                                     int num_octaves, int scales_per_octave, 
                                                     float contrast_thresh, float edge_thresh, 
                                                     float lambda_ori, float lambda_desc)
{
    assert(img.channels == 1 || img.channels == 3);

    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);
    
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves,
                                                                   scales_per_octave);
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    std::vector<Keypoint> tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    
    std::vector<Keypoint> kps;
    kps.reserve(tmp_kps.size() * 2); // Reserve space for efficiency

    // Keypoint orientation and descriptor computation with thread-local storage
    std::vector<std::vector<Keypoint>> thread_kps(omp_get_max_threads());
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        thread_kps[thread_id].reserve(tmp_kps.size() / omp_get_num_threads() + 10);
        
        #pragma omp for schedule(dynamic, 1) // Dynamic scheduling for load balancing
        for (size_t i = 0; i < tmp_kps.size(); i++) {
            Keypoint& kp_tmp = tmp_kps[i];
            std::vector<float> orientations = find_keypoint_orientations(kp_tmp, grad_pyramid,
                                                                         lambda_ori, lambda_desc);
            for (float theta : orientations) {
                Keypoint kp = kp_tmp;
                compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
                thread_kps[thread_id].push_back(kp);
            }
        }
    }
    
    // Combine results from all threads
    for (const auto& thread_result : thread_kps) {
        kps.insert(kps.end(), thread_result.begin(), thread_result.end());
    }
    return kps;
}

// MPI Parallel SIFT processing - simple approach: each process computes full SIFT independently
// This leverages OpenMP parallelism within each process
std::vector<Keypoint> find_keypoints_and_descriptors_parallel_mpi(const Image& img, int rank, int size,
                                                                 float sigma_min, int num_octaves,
                                                                 int scales_per_octave, float contrast_thresh,
                                                                 float edge_thresh, float lambda_ori, float lambda_desc)
{
    assert(img.channels == 1);

    // Each process computes the full SIFT independently
    // This leverages the existing OpenMP parallelism within each process
    std::vector<Keypoint> kps = find_keypoints_and_descriptors(img, sigma_min, num_octaves, 
                                                              scales_per_octave, contrast_thresh, 
                                                              edge_thresh, lambda_ori, lambda_desc);
    
    // Only rank 0 returns results (others are just for parallel processing)
    if (rank == 0) {
        return kps;
    } else {
        return std::vector<Keypoint>(); // Other processes return empty
    }
}

// Parallel SIFT processing with region splitting
std::vector<Keypoint> find_keypoints_and_descriptors_parallel(const Image& img, 
                                                             int rank, int size,
                                                             float sigma_min,
                                                             int num_octaves, 
                                                             int scales_per_octave, 
                                                             float contrast_thresh,
                                                             float edge_thresh,
                                                             float lambda_ori,
                                                             float lambda_desc)
{
    assert(img.channels == 1 || img.channels == 3);

    const Image& grayscale_input = img.channels == 1 ? img : rgb_to_grayscale(img);
    
    // All processes generate the same scale space pyramids
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(grayscale_input, sigma_min, num_octaves,
                                                                   scales_per_octave);
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    
    // Split keypoint detection by octaves with optimized processing
    std::vector<Keypoint> tmp_kps;
    tmp_kps.reserve(1000); // Pre-allocate reasonable size
    
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        // Each process handles different octaves
        if (i % size == rank) {
            const std::vector<Image>& octave = dog_pyramid.octaves[i];
            
            #pragma omp parallel
            {
                std::vector<Keypoint> local_tmp_kps;
                #pragma omp for schedule(dynamic)
                for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
                    const Image& img_scale = octave[j];
                    
                    // Process in blocks for better cache performance
                    const int block_size = 16;
                    for (int x_start = 1; x_start < img_scale.width-1; x_start += block_size) {
                        int x_end = std::min(x_start + block_size, img_scale.width-1);
                        for (int y_start = 1; y_start < img_scale.height-1; y_start += block_size) {
                            int y_end = std::min(y_start + block_size, img_scale.height-1);
                            
                            for (int x = x_start; x < x_end; x++) {
                                for (int y = y_start; y < y_end; y++) {
                                    float pixel_val = img_scale.get_pixel(x, y, 0);
                                    if (std::abs(pixel_val) < 0.8*contrast_thresh) {
                                        continue;
                                    }
                                    if (point_is_extremum(octave, j, x, y)) {
                                        Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                                        bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh,
                                                                                      edge_thresh);
                                        if (kp_is_valid) {
                                            local_tmp_kps.push_back(kp);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                #pragma omp critical
                {
                    tmp_kps.insert(tmp_kps.end(), local_tmp_kps.begin(), local_tmp_kps.end());
                }
            }
        }
    }
    
    // All processes generate gradient pyramid (needed for descriptor computation)
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    
    std::vector<Keypoint> kps;
    kps.reserve(tmp_kps.size() * 2);

    #pragma omp parallel
    {
        std::vector<Keypoint> local_kps;
        #pragma omp for
        for (size_t i = 0; i < tmp_kps.size(); i++) {
            Keypoint& kp_tmp = tmp_kps[i];
            std::vector<float> orientations = find_keypoint_orientations(kp_tmp, grad_pyramid,
                                                                         lambda_ori, lambda_desc);
            for (float theta : orientations) {
                Keypoint kp = kp_tmp;
                compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
                local_kps.push_back(kp);
            }
        }
        #pragma omp critical
        {
            kps.insert(kps.end(), local_kps.begin(), local_kps.end());
        }
    }
    return kps;
}

float euclidean_dist(std::array<uint8_t, 128>& a, std::array<uint8_t, 128>& b)
{
    float dist = 0;
    for (int i = 0; i < 128; i++) {
        int di = (int)a[i] - b[i];
        dist += di * di;
    }
    return std::sqrt(dist);
}

Image draw_keypoints(const Image& img, const std::vector<Keypoint>& kps)
{
    Image res(img);
    if (img.channels == 1) {
        res = grayscale_to_rgb(res);
    }
    for (auto& kp : kps) {
        draw_point(res, kp.x, kp.y, 5);
    }
    return res;
}